import json
import time
import queue
import threading
import numpy as np
import markdown
from markupsafe import Markup
from flask import Flask, render_template, request, jsonify, Response, send_file
from engine import orchestrator, ingestion
from engine import llm as llm_mod
from engine import chat_assistant
from engine import target_router
from config import REPORTS_DIR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

_run_queues = {}


@app.context_processor
def inject_ctx():
    import config
    return {"llm_ok": llm_mod.llm_available(),
            "provider": config.LLM_PROVIDER,
            "model": config.LLM_MODEL}


def _get_target():
    if request.form:
        t = request.form.get("target", "")
    else:
        try:
            t = (request.get_json(silent=True) or {}).get("target", "")
        except Exception:
            t = ""
    return (t or "").strip()


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


@app.route("/")
def index():
    return render_template("index.html",
                           targets=ingestion.list_known_targets())


@app.route("/run", methods=["POST"])
def run():
    target = _get_target()
    if not target:
        return render_template("report.html",
                               r={"error": "No target provided.",
                                  "target": ""})
    # --- Target routing: solar-system vs stellar vs unresolvable ---
    route = target_router.route_target(target)
    if route["track"] is None:
        return render_template("report.html",
                               r={"error": route["error"],
                                  "target": target,
                                  "detection": False})
    if route["track"] == "B":
        return render_template("report.html",
                               r={"error": (
                                   f"'{target}' is a solar-system body "
                                   "(Track B). Solar-system research track "
                                   "is not yet built. Use a stellar/exoplanet "
                                   "target (e.g. TIC 268684168, TOI 700)."
                               ),
                                  "target": target,
                                  "detection": False})
    # Track A: stellar/exoplanet. Proceed with the transit pipeline.
    try:
        result = orchestrator.run(target)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        fname = REPORTS_DIR / f"{target.replace(' ', '_')}_{stamp}.json"
        safe = {k: v for k, v in result.items() if k not in ("plot", "plots")}
        if result.get("paper"):
            safe["paper_md"] = result["paper"]["markdown"]
        else:
            safe["paper_md"] = None
        fname.write_text(json.dumps(_sanitize(safe), indent=2))
        return render_template("report.html", r=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("report.html",
                               r={"error": f"{type(e).__name__}: {e}",
                                  "target": target,
                                  "detection": False})


@app.route("/api/run", methods=["POST"])
def api_run():
    target = _get_target()
    if not target:
        return jsonify({"error": "no target"}), 400

    # --- Target routing ---
    route = target_router.route_target(target)
    if route["track"] is None:
        return jsonify({"error": route["error"], "target": target}), 400
    if route["track"] == "B":
        return jsonify({
            "error": (
                f"'{target}' is a solar-system body (Track B). "
                "Solar-system track not yet built."
            ),
            "target": target,
        }), 400

    try:
        result = orchestrator.run(target)
        out = {k: v for k, v in result.items() if k not in ("plot", "plots")}
        if result.get("paper"):
            out["paper_md"] = result["paper"]["markdown"]
        else:
            out["paper_md"] = None
        out["analysis"] = {k: v for k, v in out["analysis"].items()
                           if k != "flux_residuals"}
        return jsonify(_sanitize(out))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/run_stream", methods=["POST"])
def api_run_stream():
    """Start a pipeline run and stream progress via SSE."""
    target = _get_target()
    if not target:
        return jsonify({"error": "no target"}), 400

    run_id = f"{target.replace(' ', '_')}_{int(time.time())}"
    q = queue.Queue()
    _run_queues[run_id] = q

    def _progress_cb(stage, message):
        q.put({"type": "progress", "stage": stage, "message": message,
               "time": time.strftime("%H:%M:%S")})

    def _worker():
        try:
            _progress_cb("start", f"Pipeline started for {target}")
            result = orchestrator.run(target, progress_callback=_progress_cb)
            _progress_cb("done", "Pipeline complete! Generating report...")

            stamp = time.strftime("%Y%m%d-%H%M%S")
            fname = REPORTS_DIR / f"{target.replace(' ', '_')}_{stamp}.json"
            safe = {k: v for k, v in result.items() if k not in ("plot", "plots")}
            safe["paper_md"] = result["paper"]["markdown"]
            fname.write_text(json.dumps(_sanitize(safe), indent=2))

            q.put({"type": "result", "redirect": f"/report/{run_id}"})
            q.put(None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
            q.put(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    def _generate():
        while True:
            try:
                event = q.get(timeout=300)
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'timeout'})}\n\n"
                break
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
        _run_queues.pop(run_id, None)

    return Response(_generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/report/<run_id>")
def report_page(run_id):
    """Serve report from the latest JSON file matching run_id."""
    target_part = run_id.rsplit("_", 1)[0] if "_" in run_id else run_id
    reports = sorted(REPORTS_DIR.glob(f"{target_part}_*.json"), reverse=True)
    if not reports:
        return render_template("report.html",
                               r={"error": "Report not found.", "target": target_part})
    data = json.loads(reports[0].read_text())

    data["paper"] = {"markdown": data.pop("paper_md", None), "source": "llm"}
    data["plots"] = {}

    return render_template("report.html", r=data, run_id=run_id)


def _load_results_for_run(run_id: str) -> dict | None:
    """Load the saved results JSON for a run_id."""
    target_part = run_id.rsplit("_", 1)[0] if "_" in run_id else run_id
    reports = sorted(REPORTS_DIR.glob(f"{target_part}_*.json"), reverse=True)
    if not reports:
        return None
    data = json.loads(reports[0].read_text())
    data["paper"] = {"markdown": data.pop("paper_md", ""), "source": "llm"}
    data["plots"] = {}
    return data


@app.route("/api/paper/download/<run_id>")
def download_paper(run_id: str):
    """Download the research paper as a Markdown file."""
    data = _load_results_for_run(run_id)
    if data is None:
        return jsonify({"error": "report not found"}), 404
    paper_md = data.get("paper", {}).get("markdown", "")
    if not paper_md:
        return jsonify({"error": "no paper content"}), 404
    target = data.get("target", "report")
    fname = f"{target.replace(' ', '_')}_research_paper.md"
    return Response(
        paper_md,
        mimetype="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.route("/api/chat/send/<run_id>", methods=["POST"])
def chat_send(run_id: str):
    """Send a message to the AI research assistant."""
    data = _load_results_for_run(run_id)
    if data is None:
        return jsonify({"error": "report not found"}), 404

    body = request.get_json(silent=True) or {}
    question = (body.get("message") or "").strip()
    if not question:
        return jsonify({"error": "empty message"}), 400

    try:
        answer = chat_assistant.ask_assistant(question, data, run_id)
        return jsonify({"answer": answer})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/chat/history/<run_id>", methods=["GET"])
def chat_history(run_id: str):
    """Get chat history for a run."""
    history = chat_assistant.load_history(run_id)
    return jsonify({"history": history})


@app.route("/api/chat/clear/<run_id>", methods=["POST"])
def chat_clear(run_id: str):
    """Clear chat history for a run."""
    chat_assistant.clear_history(run_id)
    return jsonify({"status": "cleared"})


@app.template_filter("md")
def md_filter(text):
    return Markup(markdown.markdown(text or "", extensions=["tables", "fenced_code"]))


if __name__ == "__main__":
    app.run(debug=False, port=5001, use_reloader=False)
