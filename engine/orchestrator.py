"""Orchestrator - coordinates the 5 agents end-to-end."""
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine import ingestion, analyzer, critic, synthesizer


def run(target):
    """Run the full discovery pipeline for a target.
    Returns a dict with every stage's output plus a base64 plot."""
    lc = ingestion.fetch_tess_light_curve(target)
    analysis = analyzer.analyze(lc["time"], lc["flux"], metadata=lc.get("metadata"))
    validation = critic.validate(analysis, target)
    paper = synthesizer.write_paper(target, analysis, validation)
    plot = _plot(lc["time"], lc["flux"], analysis)

    return {
        "target": target,
        "ingestion": {"source": lc.get("source", "unknown"),
                      "n_points": int(len(lc["time"])),
                      "metadata": lc.get("metadata", {})},
        "analysis": analysis,
        "validation": validation,
        "paper": paper,
        "plot": plot,
    }


def _plot(time, flux, analysis):
    fig, ax = plt.subplots(figsize=(8, 3.0), dpi=100)
    f = np.asarray(flux, float)
    f = f / np.median(f)
    ax.scatter(time, f, s=2, color="#334155", alpha=0.7, label="flux")

    out = set(analysis.get("outlier_indices", []))
    if out:
        idx = np.array(sorted(out))
        ax.scatter(np.asarray(time)[idx],
                   np.asarray(flux)[idx] / np.median(flux),
                   s=14, color="#b91c1c", label="isolated outliers",
                   zorder=5)

    p = analysis.get("period_days")
    if p:
        ax.set_title(f"{analysis.get('depth_ppm')} ppm dip, "
                     f"P={p:.3f} d, S/N={analysis.get('snr'):.1f}",
                     fontsize=9)
    ax.set_xlabel("BJD - 2457000 (days)", fontsize=8)
    ax.set_ylabel("normalised flux", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    b = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b}"
