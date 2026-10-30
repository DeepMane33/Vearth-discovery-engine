"""Hypothesis & Validation Agent (The Critic) - Professional Grade.

Runs the 10-test verification battery, cross-catalog matching,
and AI self-questioning to produce a rigorous validation verdict.
"""
import json
import numpy as np
from engine import llm, vectorstore, validation_tests


SYSTEM = (
    "You are a rigorous astrophysics peer reviewer at a top journal. "
    "You NEVER overclaim. Given a transit detection, its verification test "
    "results, and retrieved prior literature, you MUST: "
    "(a) state one primary hypothesis, "
    "(b) enumerate skeptical self-questions attacking it from every angle "
    "(instrumental, astrophysical, statistical, catalog), "
    "(c) answer each using ONLY the provided evidence and basic physics, "
    "(d) give a final ruling. "
    "Respond ONLY as compact JSON with keys: hypothesis, self_questions "
    "(list of {question, answer, resolved_true_or_false}), confidence "
    "(low|medium|high), veto (bool), reason. "
    "CRITICAL: If the catalog cross-match found a match, you MUST veto "
    "and state it is a KNOWN PLANET, not a new discovery."
)


def validate(analysis, target, time_arr=None, flux_arr=None, known_planets=None,
             stellar_info=None):
    transit = analysis.get("transit", {})
    depth = analysis.get("depth_ppm", 0)
    snr = analysis.get("snr", 0.0)
    n_tr = transit.get("n_transits", 0)
    period = analysis.get("period_days")
    rp_rearth = analysis.get("planet_radius_rearth")
    r_star = analysis.get("stellar_radius_rsun")

    test_results = validation_tests.run_all_tests(
        time_arr if time_arr is not None else np.array([]),
        flux_arr if flux_arr is not None else np.array([]),
        analysis,
        known_planets=known_planets,
        stellar_info=stellar_info,
    )

    rag = vectorstore.query(
        f"transit dip depth {depth}ppm period {period} days false positive "
        f"eclipsing binary contamination systematic s/n {snr:.1f}", k=5)

    test_summary = "\n".join(
        f"  [{t['test_name']}] {'PASS' if t['passed'] else 'FAIL'}: {t['details']}"
        for t in test_results["tests"]
    )

    evidence = (
        f"TARGET: {target}\n"
        f"DETECTED SIGNAL:\n"
        f"  Depth: {depth} ppm\n"
        f"  S/N: {snr:.2f}\n"
        f"  Period: {period} days\n"
        f"  Recurrences: {n_tr}\n"
        f"  Noise floor: {analysis.get('noise_floor_ppm', 0)} ppm\n\n"
        f"VERIFICATION TESTS ({test_results['passed_count']}/{test_results['total_tests']} passed):\n"
        f"{test_summary}\n\n"
        f"CATALOG CROSS-MATCH:\n"
    )
    if rp_rearth:
        evidence += f"Planet radius (from Gaia stellar params): {rp_rearth:.2f} R⊕\n"
    if r_star:
        evidence += f"Stellar radius: {r_star:.3f} R☉\n"

    if known_planets:
        if len(known_planets) > 0:
            names = [p.get("pl_name", "unknown") for p in known_planets[:5]]
            evidence += f"  KNOWN PLANETS NEARBY: {', '.join(names)}\n"
        else:
            evidence += "  No matching known planets found.\n"
    else:
        evidence += "  Catalog data not available.\n"

    context = "\n\n--- PRIOR LITERATURE ---\n" + "\n".join(
        f"[{r['id']} score={r['score']:.3f}] {r['text']}" for r in rag)

    user = ("LIGHT-CURVE EVIDENCE:\n" + evidence + context +
            "\n\nApply full skeptical vetting. Output JSON only.")

    result = llm.ask_llm(SYSTEM, user, temperature=0.2, max_tokens=2000)
    parsed = llm.extract_json(result["text"])

    parsed.setdefault("hypothesis", "")
    parsed.setdefault("self_questions", [])
    parsed.setdefault("confidence", "low")
    parsed.setdefault("veto", True)
    parsed.setdefault("reason", "")

    if result["source"] == "fallback":
        veto = _rule_veto(snr, n_tr, depth, test_results, rp_rearth)
        parsed["veto"] = veto
        parsed["hypothesis"] = _fallback_hypothesis(snr, n_tr, depth, period, test_results, rp_rearth)
        parsed["self_questions"] = _fallback_questions(
            snr, n_tr, depth, period, analysis, rag, test_results, known_planets, rp_rearth)
        parsed["confidence"] = _fallback_confidence(snr, n_tr, test_results, rp_rearth)
        parsed["reason"] = ("rule-based veto: failed verification tests or implausible radius"
                            if veto else
                            "rule-based advance: signal passes verification battery and radius vetting")

    parsed["rag_refs"] = [{"id": r["id"], "score": round(r["score"], 3)} for r in rag]
    parsed["llm_source"] = result["source"]
    parsed["raw"] = result["text"]
    parsed["verification_tests"] = test_results
    return parsed


def _rule_veto(snr, n_tr, depth, test_results, rp_rearth=None):
    if test_results.get("passed_count", 0) < 5:
        return True
    if snr < 7.0:
        return True
    if n_tr < 3:
        return True
    if depth < 50 or depth > 50000:
        return True
    if rp_rearth and rp_rearth > 25:
        return True
    return False


def _fallback_confidence(snr, n_tr, test_results, rp_rearth=None):
    passed = test_results.get("passed_count", 0)
    total = test_results.get("total_tests", 10)
    ratio = passed / total if total > 0 else 0

    if rp_rearth and rp_rearth > 25:
        return "low"
    if snr >= 12 and n_tr >= 3 and ratio >= 0.8 and (rp_rearth is None or rp_rearth < 20):
        return "high"
    if snr >= 7 and n_tr >= 2 and ratio >= 0.6 and (rp_rearth is None or rp_rearth < 25):
        return "medium"
    return "low"


def _fallback_hypothesis(snr, n_tr, depth, period, test_results, rp_rearth=None):
    passed = test_results.get("passed_count", 0)
    total = test_results.get("total_tests", 10)

    if snr < 7 or n_tr < 3:
        return (
            f"The signal (S/N={snr:.1f}, {n_tr} recurrences) lacks the "
            f"statistical significance required for a confident transit "
            f"detection. It is more consistent with residual noise or "
            f"stellar variability."
        )
    if passed < total * 0.6:
        return (
            f"The {depth} ppm dip at {period} d shows some transit-like "
            f"properties but fails {total - passed}/{total} verification "
            f"tests. The signal requires additional vetting before "
            f"classification."
        )
    rp_str = f" (~{rp_rearth:.1f} R⊕)" if rp_rearth else ""
    radius_note = ""
    if rp_rearth and rp_rearth > 25:
        radius_note = (f" However, the inferred radius of {rp_rearth:.1f} R⊕ "
                       f"exceeds the ~20 R⊕ brown dwarf/stellar boundary — "
                       f"this is likely a low-mass star, not a planet.")
    return (
        f"The {depth} ppm dip repeating every {period} d over {n_tr} "
        f"recurrences passes {passed}/{total} verification tests. "
        f"This is consistent with a transiting{rp_str} companion, "
        f"pending false-positive vetting (centroid, odd-even, secondary "
        f"eclipse).{radius_note}"
    )


def _find_test(test_results, test_name):
    """Find a test result by name instead of relying on index order."""
    for t in test_results.get("tests", []):
        if t.get("test_name") == test_name:
            return t
    return None


def _fallback_questions(snr, n_tr, depth, period, analysis, rag, test_results, known_planets, rp_rearth=None):
    ids = [r["id"] for r in rag[:3]]
    qs = []

    catalog_test = _find_test(test_results, "Catalog Cross-match")
    qs.append({
        "question": "Is this signal already in the NASA Exoplanet Archive?",
        "answer": (
            f"Catalog cross-match: {catalog_test['details']}"
            if catalog_test
            else "Catalog data not checked."
        ),
        "resolved_true_or_false": catalog_test.get("passed", True)
        if catalog_test else True,
    })

    qs.append({
        "question": "Is the signal-to-noise ratio sufficient for a real detection?",
        "answer": f"S/N = {snr:.2f}. Professional threshold is 7.0. This {'passes' if snr >= 7 else 'fails'} that bar.",
        "resolved_true_or_false": snr >= 7,
    })

    qs.append({
        "question": "Do multiple transits recur at the same orbital phase?",
        "answer": f"{n_tr} transits found recurring at fixed phase. A single dip cannot explain periodicity.",
        "resolved_true_or_false": n_tr >= 3,
    })

    odd_even = _find_test(test_results, "Odd-Even Depth")
    secondary = _find_test(test_results, "Secondary Eclipse")
    vshape = _find_test(test_results, "V-Shape Test")
    qs.append({
        "question": "Could the dip be from an eclipsing binary rather than a planet?",
        "answer": (
            f"Odd-even test: {odd_even['details'] if odd_even else 'not run'} "
            f"Secondary eclipse: {secondary['details'] if secondary else 'not run'} "
            f"V-shape: {vshape['details'] if vshape else 'not run'}"
        ),
        "resolved_true_or_false": all(
            t.get("passed", False) for t in [odd_even, secondary, vshape] if t
        ),
    })

    bootstrap = _find_test(test_results, "Bootstrap FAP")
    stellar_var = _find_test(test_results, "Stellar Variability")
    qs.append({
        "question": "Is the signal a noise artifact or stellar variability?",
        "answer": (
            f"Bootstrap FAP: {bootstrap['details'] if bootstrap else 'not run'} "
            f"Stellar variability: {stellar_var['details'] if stellar_var else 'not run'}"
        ),
        "resolved_true_or_false": (
            bootstrap.get("passed", True) if bootstrap else True
        ) and (
            stellar_var.get("passed", True) if stellar_var else True
        ),
    })

    if rp_rearth:
        if rp_rearth > 25:
            qs.append({
                "question": "Is the inferred radius physically plausible for a planet?",
                "answer": f"Radius {rp_rearth:.1f} R⊕ exceeds the ~20 R⊕ brown dwarf/stellar boundary. "
                          f"This is likely a low-mass star or brown dwarf, not a planet.",
                "resolved_true_or_false": False,
            })
        elif rp_rearth > 15:
            qs.append({
                "question": "Is the inferred radius physically plausible for a planet?",
                "answer": f"Radius {rp_rearth:.1f} R⊕ is in the transition zone between giant planets "
                          f"and brown dwarfs; mass measurement required for confirmation.",
                "resolved_true_or_false": None,
            })
        else:
            qs.append({
                "question": "Is the inferred radius physically plausible for a planet?",
                "answer": f"Radius {rp_rearth:.1f} R⊕ is consistent with a planet "
                          f"(super-Earth to Jupiter-size).",
                "resolved_true_or_false": True,
            })

    return qs
