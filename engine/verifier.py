"""AI Self-Dialog Verifier - Anti-Hallucination Engine (v3.0).

Implements multi-round self-questioning to prevent false discoveries.
The AI talks to itself to verify findings are real and not hallucinated.

v3.0 enhancements:
  - Includes atmospheric characterization metrics (TSM/ESM) in evidence
  - Includes orbital stability checks in verification
  - Includes stellar density consistency as evidence
  - More detailed physics evidence in each round
"""
from engine import llm, datasources


HYPOTHESIS_SYSTEM = (
    "You are an astrophysicist who has just analyzed a transit signal. "
    "State your finding as a clear, specific hypothesis. Be precise about "
    "numbers. Consider: planet radius, equilibrium temperature, atmospheric "
    "characterization potential (TSM/ESM), orbital stability, and whether "
    "the planet is in the habitable zone. Do NOT overclaim. If uncertain, say so."
)

SKEPTIC_SYSTEM = (
    "You are a skeptical peer reviewer. You are given a hypothesis about "
    "a transit detection. Your job is to attack it from every possible angle. "
    "Ask hard questions that could invalidate the claim. Consider: "
    "instrumental artifacts, astrophysical false positives, statistical issues, "
    "catalog matches, stellar density mismatch, atmospheric implausibility, "
    "orbital instability, and any other angle. Be thorough."
)

VERIFIER_SYSTEM = (
    "You are a fact-checker for astronomical discoveries. You verify each "
    "claim against the provided evidence (data, catalogs, literature, physics). "
    "For each claim, state: VERIFIED, UNVERIFIED, or REFUTED. "
    "Pay special attention to: stellar density consistency, atmospheric metrics, "
    "orbital stability, and multi-mission data. If REFUTED, explain why. Be strict."
)

CROSSCHECK_SYSTEM = (
    "You are the final arbiter. Given all verification rounds, determine "
    "the final verdict. Be extremely conservative. Only declare a potential "
    "new discovery if ALL checks pass. Consider atmospheric characterization "
    "potential and orbital dynamics in your assessment. Respond as JSON with: "
    "verdict (confirmed_known|candidate|rejected), confidence (0-100), "
    "summary, next_steps, habitability_assessment, jwst_recommended."
)


def self_dialog_verify(analysis, target, validation, stellar_info=None, physics_data=None):
    """Run multi-round AI self-verification.
    Returns dict with rounds and final verdict."""
    transit = analysis.get("transit", {})
    depth = analysis.get("depth_ppm", 0)
    snr = analysis.get("snr", 0)
    period = analysis.get("period_days")
    n_tr = transit.get("n_transits", 0)

    rounds = []

    round1 = _round1_hypothesis(target, depth, snr, period, n_tr, stellar_info, physics_data)
    rounds.append({"round": 1, "role": "hypothesis", **round1})

    round2 = _round2_skeptic(round1["response"], target, analysis, validation, physics_data)
    rounds.append({"round": 2, "role": "skeptic", **round2})

    round3 = _round3_verifier(round1["response"], round2["response"],
                               target, analysis, validation, physics_data)
    rounds.append({"round": 3, "role": "verifier", **round3})

    round4 = _round4_crosscheck(rounds, target, analysis, validation, physics_data)
    rounds.append({"round": 4, "role": "crosscheck", **round4})

    return {
        "rounds": rounds,
        "final_verdict": round4.get("verdict", "rejected"),
        "confidence": round4.get("confidence", 0),
        "summary": round4.get("summary", ""),
        "next_steps": round4.get("next_steps", ""),
        "habitability_assessment": round4.get("habitability_assessment", ""),
        "jwst_recommended": round4.get("jwst_recommended", ""),
    }


def _round1_hypothesis(target, depth, snr, period, n_tr, stellar_info, physics_data=None):
    stellar_str = ""
    if stellar_info:
        if stellar_info.get("teff"):
            stellar_str += f"Stellar Teff: {stellar_info['teff']} K\n"
        if stellar_info.get("radius_rsun"):
            stellar_str += f"Stellar radius: {stellar_info['radius_rsun']} Rsun\n"
        if stellar_info.get("mass_msun"):
            stellar_str += f"Stellar mass: {stellar_info['mass_msun']} Msun\n"
        if stellar_info.get("mission_summary"):
            missions = stellar_info["mission_summary"].get("available_missions", [])
            if missions:
                stellar_str += f"Available missions: {', '.join(missions)}\n"

    physics_str = ""
    if physics_data:
        rp = physics_data.get("planet_radius_rearth", 0)
        teq = physics_data.get("equilibrium_temp_k", 0)
        tsm = physics_data.get("tsm", 0)
        esm = physics_data.get("esm", 0)
        in_hz = physics_data.get("in_habitable_zone", False)
        mass_est = physics_data.get("estimated_mass_earth", 0)
        density = physics_data.get("bulk_density_gcm3", 0)
        hill = physics_data.get("hill_stability", {})
        tidal = physics_data.get("tidal_locking", {})
        classification = physics_data.get("planet_classification", "unknown")

        physics_str = (
            f"\nDERIVED PLANET PROPERTIES:\n"
            f"  Radius: {rp:.2f} Re ({classification})\n"
            f"  Estimated mass: {mass_est:.2f} Me\n"
            f"  Bulk density: {density:.2f} g/cm3\n"
            f"  Equilibrium temperature: {teq:.0f} K\n"
            f"  In habitable zone: {'Yes' if in_hz else 'No'}\n"
            f"  TSM: {tsm:.1f} (atmospheric characterization metric)\n"
            f"  ESM: {esm:.3f} (emission spectroscopy metric)\n"
            f"  Hill stability: {hill.get('classification', 'unknown')}\n"
            f"  Tidal locking: {tidal.get('assessment', 'unknown')}\n"
        )

    user = (
        f"TARGET: {target}\n"
        f"DETECTED SIGNAL:\n"
        f"  Transit depth: {depth} ppm\n"
        f"  Signal-to-noise: {snr:.2f}\n"
        f"  Orbital period: {period} days\n"
        f"  Number of transits: {n_tr}\n"
        f"{stellar_str}"
        f"{physics_str}\n"
        f"State your hypothesis about what this signal represents. "
        f"Be specific and precise. Include atmospheric characterization potential "
        f"and habitability assessment if relevant."
    )

    result = llm.ask_llm(HYPOTHESIS_SYSTEM, user, temperature=0.2, max_tokens=800)
    return {"response": result["text"], "source": result["source"]}


def _round2_skeptic(hypothesis, target, analysis, validation, physics_data=None):
    test_summary = ""
    vt = validation.get("verification_tests", {})
    if vt.get("tests"):
        test_summary = "\n".join(
            f"  [{t['test_name']}] {'PASS' if t['passed'] else 'FAIL'}: {t['details']}"
            for t in vt["tests"]
        )

    physics_summary = ""
    if physics_data:
        rp = physics_data.get("planet_radius_rearth", 0)
        teq = physics_data.get("equilibrium_temp_k", 0)
        tsm = physics_data.get("tsm", 0)
        esm = physics_data.get("esm", 0)
        in_hz = physics_data.get("in_habitable_zone", False)
        mass_est = physics_data.get("estimated_mass_earth", 0)
        density = physics_data.get("bulk_density_gcm3", 0)
        e_min = physics_data.get("eccentricity_min", 0)
        e_max = physics_data.get("eccentricity_max", 0)
        hill = physics_data.get("hill_stability", {})
        tidal = physics_data.get("tidal_locking", {})

        physics_summary = (
            f"\nPHYSICS EVIDENCE:\n"
            f"  Radius: {rp:.2f} Re, Mass (est): {mass_est:.2f} Me\n"
            f"  Density: {density:.2f} g/cm3\n"
            f"  Teq: {teq:.0f} K, In HZ: {in_hz}\n"
            f"  Eccentricity: {e_min:.3f}-{e_max:.3f}\n"
            f"  TSM: {tsm:.1f}, ESM: {esm:.3f}\n"
            f"  Hill: {hill.get('classification', 'unknown')}\n"
            f"  Tidal locking: {tidal.get('assessment', 'unknown')}\n"
        )

    user = (
        f"HYPOTHESIS TO CHALLENGE:\n{hypothesis}\n\n"
        f"EVIDENCE:\n"
        f"Target: {target}\n"
        f"Depth: {analysis.get('depth_ppm')} ppm\n"
        f"S/N: {analysis.get('snr'):.2f}\n"
        f"Period: {analysis.get('period_days')} d\n"
        f"Recurrences: {analysis.get('n_transits')}\n\n"
        f"VERIFICATION TESTS:\n{test_summary}\n"
        f"{physics_summary}\n"
        f"Ask at least 5 hard skeptical questions that could invalidate "
        f"this hypothesis. Consider atmospheric implausibility, orbital instability, "
        f"density mismatch, and any other angle. For each, explain why it matters."
    )

    result = llm.ask_llm(SKEPTIC_SYSTEM, user, temperature=0.3, max_tokens=1000)
    return {"response": result["text"], "source": result["source"]}


def _round3_verifier(hypothesis, skeptic_questions, target, analysis, validation, physics_data=None):
    physics_evidence = ""
    if physics_data:
        rp = physics_data.get("planet_radius_rearth", 0)
        teq = physics_data.get("equilibrium_temp_k", 0)
        tsm = physics_data.get("tsm", 0)
        esm = physics_data.get("esm", 0)
        in_hz = physics_data.get("in_habitable_zone", False)
        mass_est = physics_data.get("estimated_mass_earth", 0)
        density = physics_data.get("bulk_density_gcm3", 0)
        e_min = physics_data.get("eccentricity_min", 0)
        e_max = physics_data.get("eccentricity_max", 0)
        hill = physics_data.get("hill_stability", {})
        prob = physics_data.get("transit_probability", {})

        physics_evidence = (
            f"\nPHYSICS EVIDENCE FOR VERIFICATION:\n"
            f"  Planet radius: {rp:.2f} Re\n"
            f"  Estimated mass: {mass_est:.2f} Me\n"
            f"  Bulk density: {density:.2f} g/cm3\n"
            f"  Equilibrium temperature: {teq:.0f} K\n"
            f"  In habitable zone: {in_hz}\n"
            f"  TSM: {tsm:.1f}, ESM: {esm:.3f}\n"
            f"  Eccentricity bounds: {e_min:.3f}-{e_max:.3f}\n"
            f"  Hill stability: {hill.get('classification', 'unknown')}\n"
            f"  Transit probability: {prob.get('geometric_probability', 0)*100:.2f}%\n"
        )

    user = (
        f"HYPOTHESIS: {hypothesis}\n\n"
        f"SKEPTICAL QUESTIONS:\n{skeptic_questions}\n\n"
        f"EVIDENCE TO CHECK AGAINST:\n"
        f"Target: {target}\n"
        f"Depth: {analysis.get('depth_ppm')} ppm\n"
        f"S/N: {analysis.get('snr'):.2f}\n"
        f"Period: {analysis.get('period_days')} d\n"
        f"{physics_evidence}\n"
        f"For each skeptical question, verify, unverify, or refute it "
        f"using the evidence. Pay special attention to: stellar density consistency, "
        f"atmospheric metrics, orbital stability, and physical plausibility. Be strict."
    )

    result = llm.ask_llm(VERIFIER_SYSTEM, user, temperature=0.2, max_tokens=1200)
    return {"response": result["text"], "source": result["source"]}


def _round4_crosscheck(all_rounds, target, analysis, validation, physics_data=None):
    rounds_text = ""
    for r in all_rounds:
        rounds_text += f"\n--- ROUND {r['round']} ({r['role']}) ---\n{r['response']}\n"

    vt = validation.get("verification_tests", {})
    passed = vt.get("passed_count", 0)
    total = vt.get("total_tests", 10)

    physics_summary = ""
    if physics_data:
        rp = physics_data.get("planet_radius_rearth", 0)
        teq = physics_data.get("equilibrium_temp_k", 0)
        tsm = physics_data.get("tsm", 0)
        esm = physics_data.get("esm", 0)
        in_hz = physics_data.get("in_habitable_zone", False)
        classification = physics_data.get("planet_classification", "unknown")
        hill = physics_data.get("hill_stability", {})
        tidal = physics_data.get("tidal_locking", {})

        physics_summary = (
            f"\nPHYSICS SUMMARY:\n"
            f"  Classification: {classification}\n"
            f"  Radius: {rp:.2f} Re, Teq: {teq:.0f} K\n"
            f"  In HZ: {in_hz}\n"
            f"  TSM: {tsm:.1f} (JWST suitability)\n"
            f"  ESM: {esm:.3f} (emission detection)\n"
            f"  Hill stability: {hill.get('classification', 'unknown')}\n"
            f"  Tidal locking: {tidal.get('assessment', 'unknown')}\n"
        )

    user = (
        f"TARGET: {target}\n"
        f"ALL VERIFICATION ROUNDS:\n{rounds_text}\n\n"
        f"SUMMARY:\n"
        f"  Tests passed: {passed}/{total}\n"
        f"  S/N: {analysis.get('snr'):.2f}\n"
        f"  Period: {analysis.get('period_days')} d\n"
        f"{physics_summary}\n"
        f"Determine the final verdict. Be extremely conservative. "
        f"Only declare 'candidate' if the evidence strongly supports it. "
        f"Include a brief habitability assessment and JWST recommendation "
        f"in your response. Respond as JSON with: verdict, confidence, "
        f"summary, next_steps, habitability_assessment, jwst_recommended."
    )

    result = llm.ask_llm(CROSSCHECK_SYSTEM, user, temperature=0.1, max_tokens=800)
    parsed = llm.extract_json(result["text"])

    parsed.setdefault("verdict", "rejected")
    parsed.setdefault("confidence", 0)
    parsed.setdefault("summary", result["text"][:500])
    parsed.setdefault("next_steps", "Further analysis required.")
    parsed["source"] = result["source"]

    return parsed
