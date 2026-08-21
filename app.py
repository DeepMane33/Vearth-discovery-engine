import json
import time
import numpy as np
import markdown
from flask import Flask, render_template, request, jsonify
from engine import orchestrator, ingestion
from engine import llm as llm_mod
from config import REPORTS_DIR

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024


@app.context_processor
def inject_ctx():
    import config
    return {"llm_ok": llm_mod.llm_available(),
            "provider": config.LLM_PROVIDER,
            "model": config.LLM_MODEL}


def _get_target():
    """Pull target from form data OR JSON body, safely."""
    if request.form:
        t = request.form.get("target", "")
    else:
        try:
            t = (request.get_json(silent=True) or {}).get("target", "")
        except Exception:
            t = ""
    return (t or "").strip()


def _sanitize(obj):
    """Recursively convert numpy types/arrays to JSON-safe Python natives."""
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
        safe = {k: v for k, v in result.items() if k != "plot"}
        safe["paper_md"] = result["paper"]["markdown"]
        fname.write_text(json.dumps(_sanitize(safe), indent=2))
        return render_template("report.html", r=result)
    except Exception as e:
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
        out = {k: v for k, v in result.items() if k != "plot"}
        out["paper_md"] = result["paper"]["markdown"]
        out["analysis"] = {k: v for k, v in out["analysis"].items()
                           if k != "flux_residuals"}
        return jsonify(_sanitize(out))
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.template_filter("md")
def md_filter(text):
    return markdown.markdown(text or "", extensions=["tables", "fenced_code"])


if __name__ == "__main__":
    app.run(debug=True, port=5000)
