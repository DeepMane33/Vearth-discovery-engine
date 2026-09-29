import json
import time
import queue
import threading
import numpy as np
import markdown
from markupsafe import Markup
from flask import Flask, render_template, request, jsonify, Response
from engine import orchestrator, ingestion
from engine import llm as llm_mod
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
    try:
        result = orchestrator.run(target)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        fname = REPORTS_DIR / f"{target.replace(' ', '_')}_{stamp}.json"
        safe = {k: v for k, v in result.items() if k not in ("plot", "plots")}
        safe["paper_md"] = result["paper"]["markdown"]
        fname.write_text(json.dumps(_sanitize(safe), indent=2))
        return render_template("report.html", r=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("report.html",
                               r={"error": f"{type(e).__name__}: {e}",
                                  "target": target})


@app.route("/api/run", methods=["POST"])
def api_run():
    target = _get_target()
    if not target:
        return jsonify({"error": "no target"}), 400
    try:
        result = orchestrator.run(target)
        out = {k: v for k, v in result.items() if k not in ("plot", "plots")}
        out["paper_md"] = result["paper"]["markdown"]
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
                event = q.get(timeout=120)
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

    data["paper"] = {"markdown": data.pop("paper_md", ""), "source": "llm"}
    data["plots"] = {}

    return render_template("report.html", r=data)


@app.template_filter("md")
def md_filter(text):
    return Markup(markdown.markdown(text or "", extensions=["tables", "fenced_code"]))


if __name__ == "__main__":
    app.run(debug=False, port=5000, use_reloader=False)
