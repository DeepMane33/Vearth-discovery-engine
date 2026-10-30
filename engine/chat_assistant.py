"""AI Research Assistant Chat.

Answers questions about VEARTH research results using the actual
results JSON as context. Every answer is grounded in computed data
from the pipeline, never fabricated.

Chat history is persisted per run/session so nothing is ever lost.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from config import DATA_DIR
from engine import llm

CHAT_DIR = DATA_DIR / "chat_history"
CHAT_DIR.mkdir(parents=True, exist_ok=True)


def _chat_path(run_id: str) -> Path:
    return CHAT_DIR / f"{run_id}.json"


def load_history(run_id: str) -> list[dict]:
    """Load chat history for a run. Returns list of {role, content, timestamp}."""
    p = _chat_path(run_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(run_id: str, history: list[dict]) -> None:
    """Persist chat history for a run."""
    p = _chat_path(run_id)
    p.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def append_message(run_id: str, role: str, content: str) -> None:
    """Add a message to the chat history and persist."""
    history = load_history(run_id)
    history.append({
        "role": role,
        "content": content,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_history(run_id, history)


def clear_history(run_id: str) -> None:
    """Delete chat history for a run."""
    p = _chat_path(run_id)
    if p.exists():
        p.unlink()


def build_context(results_json: dict) -> str:
    """Build a plain-text summary of the results JSON for the LLM context.

    Includes only computed values so the assistant explains real results,
    not fabricated data.
    """
    target = results_json.get("target", "unknown")
    analysis = results_json.get("analysis", {})
    validation = results_json.get("validation", {})
    physics = results_json.get("physics", {})
    stellar = results_json.get("stellar_info", {})
    novelty = results_json.get("novelty", {})
    self_verify = results_json.get("self_verification", {})
    monte_carlo = results_json.get("monte_carlo", {})
    ingestion = results_json.get("ingestion", {})

    transit = analysis.get("transit", {})

    lines = []
    lines.append(f"TARGET: {target}")
    lines.append(f"DATA SOURCE: {ingestion.get('source', 'unknown')}")
    lines.append(f"N POINTS: {ingestion.get('n_points', 'unknown')}")
    lines.append("")
    lines.append("TRANSIT DETECTION (from BLS analysis):")
    lines.append(f"  Depth: {analysis.get('depth_ppm', 0)} ppm")
    lines.append(f"  SNR: {analysis.get('snr', 0)}")
    lines.append(f"  Period: {analysis.get('period_days', 'N/A')} days")
    lines.append(f"  T0: {transit.get('t0', 'N/A')}")
    lines.append(f"  Recurrences: {transit.get('n_transits', 0)}")
    lines.append(f"  Noise floor: {analysis.get('noise_floor_ppm', 0)} ppm")
    lines.append(f"  Reduced chi2: {analysis.get('reduced_chi2', 0)}")
    lines.append("")
    lines.append("PLANET PHYSICS (computed from depth + stellar parameters):")
    lines.append(f"  Planet radius: {physics.get('planet_radius_rearth', 0)} Earth radii")
    lines.append(f"  Planet radius: {physics.get('planet_radius_rjup', 0)} Jupiter radii")
    lines.append(f"  Estimated mass: {physics.get('estimated_mass_earth', 0)} Earth masses (ESTIMATED via Chen & Kipping 2017)")
    lines.append(f"  Bulk density: {physics.get('bulk_density_gcm3', 0)} g/cm3 (ESTIMATED)")
    lines.append(f"  Semi-major axis: {physics.get('semi_major_axis_au', 0)} AU")
    lines.append(f"  Equilibrium temperature: {physics.get('equilibrium_temp_k', 0)} K")
    lines.append(f"  In habitable zone: {physics.get('in_habitable_zone', False)}")
    lines.append(f"  Classification: {physics.get('planet_classification', 'unknown')}")
    lines.append(f"  Transit duration: {physics.get('transit_duration_hours', 0)} hours")
    lines.append(f"  Eccentricity: {physics.get('eccentricity_min', 0)} - {physics.get('eccentricity_max', 0)}")
    lines.append(f"  TSM (atmospheric metric, ESTIMATED): {physics.get('tsm', 0)} ({physics.get('tsm_tier', 'unknown')})")
    lines.append(f"  ESM (emission metric, ESTIMATED): {physics.get('esm', 0)} ({physics.get('esm_tier', 'unknown')})")
    lines.append(f"  Hill stability: {physics.get('hill_stability', {}).get('classification', 'unknown')}")
    lines.append(f"  Tidal locking: {physics.get('tidal_locking', {}).get('assessment', 'unknown')}")
    lines.append("")
    lines.append("STELLAR PARAMETERS:")
    lines.append(f"  Teff: {stellar.get('teff', 'N/A')} K")
    lines.append(f"  Radius: {stellar.get('radius_rsun', 'N/A')} Rsun")
    lines.append(f"  Mass: {stellar.get('mass_msun', 'N/A')} Msun")
    lines.append("")
    lines.append("VERIFICATION (15-test battery):")
    vt = validation.get("verification_tests", {})
    lines.append(f"  Passed: {vt.get('passed_count', 0)} / {vt.get('total_tests', 0)}")
    lines.append(f"  Average score: {vt.get('average_score', 0)}")
    for t in vt.get("tests", []):
        lines.append(f"  - {t.get('test_name', 'unknown')}: {'PASS' if t.get('passed') else 'FAIL'} - {t.get('details', '')[:120]}")
    lines.append("")
    lines.append(f"Critic hypothesis: {validation.get('hypothesis', 'N/A')}")
    lines.append(f"Critic veto: {validation.get('veto', True)}")
    lines.append(f"Critic confidence: {validation.get('confidence', 'unknown')}")
    lines.append(f"Critic reason: {validation.get('reason', 'N/A')}")
    lines.append("")
    lines.append(f"NOVELTY STATUS: {novelty.get('status', 'UNKNOWN')}")
    lines.append(f"NOVELTY CHECKED: {', '.join(novelty.get('checked', [])) if novelty.get('checked') else 'not run'}")
    lines.append("")
    lines.append(f"AI SELF-VERIFICATION VERDICT: {self_verify.get('final_verdict', 'N/A')}")
    lines.append(f"CONFIDENCE: {self_verify.get('confidence', 'N/A')}%")
    lines.append(f"SUMMARY: {self_verify.get('summary', 'N/A')}")
    lines.append("")
    lines.append(f"MONTE CARLO ITERATIONS: {monte_carlo.get('iterations', 0)}")
    if monte_carlo.get("iterations", 0) > 50:
        lines.append(f"  Period: {monte_carlo.get('period_mean', 'N/A')} +/- {monte_carlo.get('period_std', 'N/A')} days")
        lines.append(f"  Depth: {monte_carlo.get('depth_mean', 'N/A')} +/- {monte_carlo.get('depth_std', 'N/A')} ppm")

    return "\n".join(lines)


def ask_assistant(question: str, results_json: dict, run_id: str) -> str:
    """Answer a user question about the research results.

    Uses the LLM with the results JSON as context. Every number the LLM
    references comes from the computed results, not fabrication.
    """
    context = build_context(results_json)

    system = (
        "You are a research assistant helping a user understand VEARTH "
        "Discovery Engine results. You have access to the COMPLETE computed "
        "results JSON below. Answer the user's question using ONLY these "
        "computed values. "
        "CRITICAL RULES:\n"
        "1. Never invent numbers, citations, or data. If a value is not in "
        "the context, say it is not available.\n"
        "2. When explaining a calculation, show the formula and the values "
        "used from the context.\n"
        "3. Clearly mark ESTIMATED values (mass from mass-radius relation, "
        "TSM, ESM, bulk density) as estimates, not measurements.\n"
        "4. Remind the user that findings are candidates unless independently "
        "confirmed.\n"
        "5. Be concise but thorough. Use plain language.\n"
        "6. If asked about a specific number, point to exactly where it came "
        "from in the pipeline (e.g., 'BLS analysis', 'physics module', "
        "'15-test verification battery').\n"
    )

    user = (
        f"RESEARCH RESULTS CONTEXT:\n{context}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"Answer using only the computed values above."
    )

    result = llm.ask_llm(system, user, temperature=0.3, max_tokens=1500)
    answer = result["text"]

    # Persist the Q&A
    append_message(run_id, "user", question)
    append_message(run_id, "assistant", answer)

    return answer
