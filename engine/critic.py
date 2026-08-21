"""Hypothesis & Validation Agent (The Critic).

When the Analyzer flags an anomaly, this agent:
  1. drafts a scientific hypothesis
  2. retrieves relevant prior literature from the Vector DB (RAG)
  3. self-questions the hypothesis from multiple angles
  4. cross-checks each doubt against the retrieved literature
  5. decides whether to VETO the anomaly or ADVANCE it as a candidate
"""
import json
from engine import llm, vectorstore


SYSTEM = (
    "You are a rigorous astrophysicist peer reviewer. You NEVER overclaim. "
    "Given a light-curve anomaly and retrieved prior literature, you: "
    "(a) state one primary hypothesis, (b) enumerate skeptical self-"
    "questions attacking it from every angle (instrumental, astrophysical, "
    "statistical), (c) answer each using ONLY the literature provided and "
    "basic physics, (d) give a final ruling. Respond ONLY as compact JSON "
    "with keys: hypothesis, self_questions (list of {question, answer, "
    "resolved_true_or_false}), confidence (low|medium|high), veto (bool), "
    "reason."
)


def validate(analysis, target):
    transit = analysis.get("transit", {})
    depth = analysis.get("depth_ppm", 0)
    snr = analysis.get("snr", 0.0)
    n_tr = analysis.get("n_transits", 0)
    period = analysis.get("period_days")
    rp_rearth = analysis.get("planet_radius_rearth")
    r_star = analysis.get("stellar_radius_rsun")

    evidence = (
        f"Target: {target}\n"
        f"Detected dip depth: {depth} ppm\n"
        f"Transit S/N (box least squares): {snr:.2f}\n"
        f"Best-fit period: {period} days\n"
        f"Number of transits recurring at fixed phase: {n_tr}\n"
        f"Reduced chi-squared of flat model: {analysis.get('reduced_chi2', 0):.5f}\n"
        f"Noise floor: {analysis.get('noise_floor_ppm', 0)} ppm\n"
    )
    if rp_rearth:
        evidence += f"Planet radius (from Gaia stellar params): {rp_rearth:.2f} R⊕\n"
    if r_star:
        evidence += f"Stellar radius: {r_star:.3f} R☉\n"

    rag = vectorstore.query(
        f"transit dip depth {depth}ppm period {period} days false positive "
        f"eclipsing binary contamination systematic s/n {snr:.1f}", k=5)
    context = "\n\n--- PRIOR LITERATURE ---\n" + "\n".join(
        f"[{r['id']} score={r['score']:.3f}] {r['text']}" for r in rag)

    user = ("LIGHT-CURVE EVIDENCE:\n" + evidence + context +
            "\n\nApply full skeptical vetting. Output JSON only.")

    result = llm.ask_llm(SYSTEM, user, temperature=0.2, max_tokens=1600)
    parsed = llm.extract_json(result["text"])

    parsed.setdefault("hypothesis", "")
    parsed.setdefault("self_questions", [])
    parsed.setdefault("confidence", "low")
    parsed.setdefault("veto", True)
    parsed.setdefault("reason", "")

    if result["source"] == "fallback":
        veto = _rule_veto(snr, n_tr, depth, rp_rearth)
        parsed["veto"] = veto
        parsed["hypothesis"] = _fallback_hypothesis(snr, n_tr, depth, period, rp_rearth)
        parsed["self_questions"] = _fallback_questions(
            snr, n_tr, depth, period, analysis, rag, rp_rearth, r_star)
        parsed["confidence"] = _fallback_confidence(snr, n_tr, rp_rearth)
        parsed["reason"] = ("rule-based veto: insufficient S/N or recurrence"
                            if veto else
                            "rule-based advance: recurring dip clears basic vetting")

    parsed["rag_refs"] = [{"id": r["id"], "score": round(r["score"], 3)}
                          for r in rag]
    parsed["llm_source"] = result["source"]
    parsed["raw"] = result["text"]
    return parsed


def _rule_veto(snr, n_tr, depth, rp_rearth=None):
    if n_tr < 2:
        return True
    if snr < 7.0:
        return True
    if depth < 50 or depth > 50000:
        return True
    if rp_rearth and rp_rearth > 25:
        return True
    return False


def _fallback_confidence(snr, n_tr, rp_rearth=None):
    if snr >= 12 and n_tr >= 3 and (rp_rearth is None or rp_rearth < 20):
        return "high"
    if snr >= 7 and n_tr >= 2 and (rp_rearth is None or rp_rearth < 25):
        return "medium"
    return "low"


def _fallback_hypothesis(snr, n_tr, depth, period, rp_rearth=None):
    if snr < 7 or n_tr < 2:
        return ("The shallow dip lacks the signal-to-noise and recurrence "
                "required for a confident transit detection; it is more "
                "consistent with residual instrumental noise or stellar "
                "variability than with a transiting companion.")
    rp_str = f" (~{rp_rearth:.1f} R⊕)" if rp_rearth else ""
    return (f"The {depth} ppm dip repeating every {period} d over "
            f"{n_tr} recurrences is consistent with a single-body transit "
            f"by a {rp_str} companion, pending false-positive "
            f"vetting (centroid, odd-even, secondary eclipse).")


def _fallback_questions(snr, n_tr, depth, period, analysis, rag, rp_rearth=None, r_star=None):
    ids = [r["id"] for r in rag[:3]]
    qs = []
    qs.append({"question": "Is the signal real or a single-point outlier?",
               "answer": f"{n_tr} transits recur at fixed phase; a single "
                         f"cosmic-ray/hot pixel cannot explain periodicity, "
                         f"so the signal is not a lone outlier per "
                         f"{ids[0] if ids else 'noise lit'}.",
              "resolved_true_or_false": n_tr >= 2})
    qs.append({"question": "Could the dip be a detector systematic / roll?",
               "answer": "The best period is non-instrumental unless it "
                         "matches the TESS roll cadence (~13.7 d); here it "
                         f"is {period} d, so instrument roll is unlikely.",
              "resolved_true_or_false": not (period and abs(period - 13.7) < 0.5)})
    qs.append({"question": "Might it be a blended/background eclipsing binary?",
               "answer": "Depth is {} ppm — consistent with a small planet or "
                         "a diluted binary; centroid-offset data is required "
                         "to settle this per the contamination reference.".format(depth),
              "resolved_true_or_false": depth < 20000})
    qs.append({"question": "Is S/N statistically significant?",
               "answer": f"Box-LS S/N = {snr:.2f}; the standard planet threshold "
                         f"is ~7. This {'passes' if snr >= 7 else 'fails'} that bar.",
              "resolved_true_or_false": snr >= 7})
    qs.append({"question": "Could stellar variability mimic the dip?",
               "answer": "Starspots/flares are quasi-periodic or aperiodic and "
                         "do not produce a clean phase-locked box transit "
                         "repeating over many orbits.",
              "resolved_true_or_false": n_tr >= 3})
    if rp_rearth:
        if rp_rearth > 25:
            qs.append({"question": "Is the inferred radius physically plausible for a planet?",
                       "answer": f"Radius {rp_rearth:.1f} R⊕ exceeds the ~20 R⊕ brown dwarf/stellar boundary. "
                                 f"This is likely a low-mass star or brown dwarf, not a planet.",
                       "resolved_true_or_false": False})
        elif rp_rearth > 15:
            qs.append({"question": "Is the inferred radius physically plausible for a planet?",
                       "answer": f"Radius {rp_rearth:.1f} R⊕ is in the transition zone between giant planets "
                                 f"and brown dwarfs; mass measurement required for confirmation.",
                       "resolved_true_or_false": None})
        else:
            qs.append({"question": "Is the inferred radius physically plausible for a planet?",
                       "answer": f"Radius {rp_rearth:.1f} R⊕ is consistent with a planet "
                                 f"(super-Earth to Jupiter-size).",
                       "resolved_true_or_false": True})
    return qs
