"""Orchestrator - Coordinates the VEARTH discovery pipeline (v3.0).

Stages:
  1. COLLECT     - Fetch TESS light curve + stellar parameters + multi-mission
  2. CROSS-MATCH - Check if target already has known planets
  3. ANALYZE     - Real BLS + transit fitting + physics (sequential multi-planet)
  4. VALIDATE    - 15-test verification battery
  5. SELF-VERIFY - AI self-dialog anti-hallucination
  6. NOVELTY     - Check against public catalogs (archive, TOI, SIMBAD, arXiv)
  7. PLOT        - Publication-quality figures
  8. SYNTHESIZE  - Write research paper
  8. SAVE        - Store results

v3.0 enhancements:
  - Sequential multi-planet detection (up to 5 planets)
  - Monte Carlo uncertainty propagation
  - Multi-mission data integration (TESS + Kepler + Gaia + TIC)
  - 15 validation tests
"""
import numpy as np
from engine import ingestion, analyzer, critic, verifier, synthesizer, plots
from engine import physics, novelty
from engine import ttv, flares, secondary_eclipse, ephemeris, single_transit


def _remove_signal_and_rediscover(time_arr, flux_arr, first_analysis=None,
                                  max_planets=5):
    """Sequentially detect and remove planet signals.

    After finding a transit, subtract it and search for more planets.
    Pass first_analysis (result of analyzer.analyze) to avoid
    recomputing the primary search. Returns list of per-planet dicts.
    """
    results = []
    t = np.copy(time_arr)
    f = np.copy(flux_arr)

    analysis = first_analysis if first_analysis is not None else analyzer.analyze(t, f)

    for planet_num in range(max_planets):
        if not analysis.get("is_anomaly"):
            break
        if analysis.get("snr", 0) < 5.0:
            break

        results.append({
            "planet_number": planet_num + 1,
            "analysis": analysis,
            "time_arr": t,
            "flux_arr": f,
        })

        transit = analysis.get("transit", {})
        p = transit.get("period")
        t0_found = transit.get("t0", 0.0)
        d = analysis.get("depth_ppm", 0) / 1e6

        if not p or p <= 0 or d <= 0:
            break

        phase = (t - t0_found) % p / p
        in_transit = (phase < 0.05) | (phase > 0.95)
        out_transit = ~in_transit

        if in_transit.sum() < 3:
            break

        model = np.ones_like(f)
        model[in_transit] = 1.0 - d
        f = f / model
        f = f / np.nanmedian(f)

        analysis = analyzer.analyze(t, f)

    return results


def _monte_carlo_uncertainty(time_arr, flux_arr, n_iterations=200,
                             nominal_period=None):
    """Bootstrap Monte Carlo for parameter uncertainty propagation.

    Perturbs the light curve within noise, re-runs a fast BLS search,
    returns posterior distributions for period, depth, duration.
    When nominal_period is given the fast search is restricted to a
    +/-30% window around it, which is where the posterior lives and is
    ~10x cheaper than a blind full-range search.
    """
    rng = np.random.default_rng(42)
    t = np.asarray(time_arr, dtype=float)
    f = np.asarray(flux_arr, dtype=float)
    med = np.nanmedian(f)
    if med and np.isfinite(med) and med != 0:
        f = f / med

    noise = np.std(f[np.abs(f - 1.0) < 0.01]) if np.sum(np.abs(f - 1.0) < 0.01) > 10 else 0.001

    periods = []
    depths = []
    snrs = []

    for _ in range(n_iterations):
        f_perturbed = f + rng.normal(0, noise, len(f))

        try:
            if nominal_period:
                ana = analyzer.analyze_fast_window(
                    t, f_perturbed,
                    center=nominal_period,
                    window_frac=0.3,
                )
            else:
                ana = analyzer.analyze(t, f_perturbed, fast=True)
            transit = ana.get("transit", {})
            if transit.get("period") and transit.get("period") > 0:
                periods.append(transit["period"])
                depths.append(ana.get("depth_ppm", 0))
                snrs.append(ana.get("snr", 0))
        except Exception:
            continue

    if len(periods) < 10:
        return {
            "iterations": len(periods),
            "period_mean": None,
            "period_std": None,
            "period_median": None,
            "period_1sigma": (None, None),
            "depth_mean": None,
            "depth_std": None,
            "depth_median": None,
            "depth_1sigma": (None, None),
            "snr_mean": None,
            "snr_std": None,
        }

    periods = np.array(periods)
    depths = np.array(depths)
    snrs = np.array(snrs)

    return {
        "iterations": len(periods),
        "period_mean": round(float(np.mean(periods)), 6),
        "period_std": round(float(np.std(periods)), 6),
        "period_median": round(float(np.median(periods)), 6),
        "period_1sigma": (
            round(float(np.percentile(periods, 16)), 6),
            round(float(np.percentile(periods, 84)), 6),
        ),
        "depth_mean": round(float(np.mean(depths)), 1),
        "depth_std": round(float(np.std(depths)), 1),
        "depth_median": round(float(np.median(depths)), 1),
        "depth_1sigma": (
            round(float(np.percentile(depths, 16)), 1),
            round(float(np.percentile(depths, 84)), 1),
        ),
        "snr_mean": round(float(np.mean(snrs)), 2),
        "snr_std": round(float(np.std(snrs)), 2),
    }


def run(target, progress_callback=None):
    """Run the full VEARTH discovery pipeline for a target.

    Args:
        target: Star/planet identifier (e.g., "TIC 268684168")
        progress_callback: Optional callable(stage_name, message) for progress updates
    """
    def _progress(stage, msg):
        if progress_callback:
            progress_callback(stage, msg)

    _progress("collect", "Fetching stellar parameters and light curve...")

    stellar_info = ingestion.fetch_stellar_info(target)

    try:
        from engine.datasources import multi_mission_summary
        mission_summary = multi_mission_summary(target)
        stellar_info["mission_summary"] = mission_summary
    except Exception:
        stellar_info["mission_summary"] = {"available_missions": ["TESS"]}

    lc = ingestion.fetch_tess_light_curve(target)

    # --- Data quality validation: fail loudly on bad or synthetic data ---
    n_points = int(len(lc["time"]))
    if n_points < 50:
        raise RuntimeError(
            f"Insufficient light curve points for '{target}': {n_points} "
            f"(minimum 50 required). Cannot proceed."
        )
    if lc.get("source") == "synthetic":
        raise RuntimeError(
            f"Synthetic light curve detected for '{target}'. "
            "Real TESS data is required; synthetic data is not allowed "
            "in the pipeline."
        )
    if lc.get("source", "unknown") == "unknown":
        raise RuntimeError(
            f"Unknown data source for '{target}'. Cannot verify data "
            "provenance; aborting."
        )

    _progress("crossmatch", "Checking for known planets in catalogs...")

    known_planets = None
    try:
        from engine.datasources import query_simbad_coordinates, get_known_planets_near
        coords = query_simbad_coordinates(target)
        if coords.get("ra") and coords.get("dec"):
            known_planets = get_known_planets_near(
                float(coords["ra"]), float(coords["dec"]), radius_deg=0.15)
    except Exception:
        known_planets = []

    _progress("analyze", "Running BLS transit search and multi-planet detection...")

    primary = analyzer.analyze(lc["time"], lc["flux"])

    multi_planet_results = _remove_signal_and_rediscover(
        lc["time"], lc["flux"],
        first_analysis=primary,
        max_planets=5,
    )

    if multi_planet_results:
        primary = multi_planet_results[0]["analysis"]

    _progress("physics", "Computing planet physics and atmospheric characterization...")

    physics_data = physics.compute_all_physics(primary, stellar_info)

    all_planets = []
    for pm in multi_planet_results:
        p_analysis = pm["analysis"]
        p_physics = physics.compute_all_physics(p_analysis, stellar_info)
        all_planets.append({
            "planet_number": pm["planet_number"],
            "analysis": {
                "depth_ppm": p_analysis.get("depth_ppm", 0),
                "snr": p_analysis.get("snr", 0),
                "period_days": p_analysis.get("period_days"),
                "n_transits": p_analysis.get("n_transits", 0),
            },
            "physics": p_physics,
            "vetoed": False,
        })

    _progress("montecarlo", "Running Monte Carlo uncertainty propagation...")

    nominal_period = primary.get("period_days") or (
        primary.get("transit", {}) or {}).get("period")
    mc_results = _monte_carlo_uncertainty(
        lc["time"], lc["flux"],
        n_iterations=200,
        nominal_period=nominal_period if (
            nominal_period and nominal_period > 0) else None,
    )

    _progress("validate", "Running 15-test verification battery...")

    validation = critic.validate(
        primary, target,
        time_arr=lc["time"],
        flux_arr=lc["flux"],
        known_planets=known_planets,
        stellar_info=stellar_info,
    )

    # --- Veto gating: short no-detection report if S/N or verdict fails ---
    snr = primary.get("snr", 0.0)
    depth_ppm = primary.get("depth_ppm", 0)
    n_transits = primary.get("transit", {}).get("n_transits", 0)
    tests_passed = validation.get("verification_tests", {}).get("passed_count", 0)
    tests_total = validation.get("verification_tests", {}).get("total_tests", 0)

    veto_reasons = []
    if snr < 7.0:
        veto_reasons.append(f"S/N {snr:.2f} below threshold 7.0")
    if n_transits < 3:
        veto_reasons.append(f"only {n_transits} transits (minimum 3 required)")
    if tests_passed < 5:
        veto_reasons.append(f"only {tests_passed}/{tests_total} tests passed")
    if validation.get("veto", True):
        veto_reasons.append(f"critic veto: {validation.get('reason', 'unspecified')}")

    if veto_reasons:
        _progress("done", "Pipeline complete: no detection.")
        return {
            "target": target,
            "detection": False,
            "veto_reasons": veto_reasons,
            "ingestion": {
                "source": lc.get("source", "unknown"),
                "n_points": n_points,
                "mission_summary": stellar_info.get("mission_summary", {}),
            },
            "stellar_info": stellar_info,
            "analysis": primary,
            "physics": None,
            "validation": validation,
            "self_verification": None,
            "paper": None,
            "plots": {},
            "known_planets": known_planets or [],
            "multi_planet_results": [],
            "monte_carlo": None,
            "novelty": None,
        }

    for pm in all_planets:
        if pm["planet_number"] > 1:
            p_analysis = pm["analysis"]
            vt_simple = {
                "snr": p_analysis.get("snr", 0),
                "n_transits": p_analysis.get("n_transits", 0),
                "depth_ppm": p_analysis.get("depth_ppm", 0),
            }
            pm["vetoed"] = (
                vt_simple["snr"] < 7.0 or
                vt_simple["n_transits"] < 3 or
                vt_simple["depth_ppm"] < 50 or
                vt_simple["depth_ppm"] > 50000
            )

    _progress("discovery", "Running Track A discovery modules...")

    # --- Track A discovery modules (each optional, wrapped in try/except) ---
    ttv_result = None
    try:
        ttv_result = ttv.compute_ttv(
            lc["time"], lc["flux"],
            period_days=primary.get("period_days"),
            t0=primary.get("transit", {}).get("t0", 0.0),
            duration_hours=primary.get("transit_duration_hours"),
        )
    except Exception as e:
        ttv_result = {"error": f"{type(e).__name__}: {e}"}

    flare_result = None
    try:
        flare_result = flares.detect_flares(lc["time"], lc["flux"])
    except Exception as e:
        flare_result = {"error": f"{type(e).__name__}: {e}"}

    eclipse_result = None
    try:
        eclipse_result = secondary_eclipse.search_secondary_eclipse(
            lc["time"], lc["flux"],
            period_days=primary.get("period_days"),
            t0=primary.get("transit", {}).get("t0", 0.0),
            duration_hours=primary.get("transit_duration_hours"),
        )
    except Exception as e:
        eclipse_result = {"error": f"{type(e).__name__}: {e}"}

    ephemeris_result = None
    try:
        ephemeris_result = ephemeris.update_ephemeris(
            lc["time"], lc["flux"],
            known_period_days=primary.get("period_days"),
            known_t0=primary.get("transit", {}).get("t0", 0.0),
            duration_hours=primary.get("transit_duration_hours"),
        )
    except Exception as e:
        ephemeris_result = {"error": f"{type(e).__name__}: {e}"}

    single_transit_result = None
    try:
        single_transit_result = single_transit.detect_single_transit(
            lc["time"], lc["flux"],
            sigma_threshold=6.0,
        )
    except Exception as e:
        single_transit_result = {"error": f"{type(e).__name__}: {e}"}

    _progress("novelty", "Running novelty check against public catalogs...")

    nominal_period = primary.get("period_days") or (
        primary.get("transit", {}) or {}).get("period")
    novelty_result = novelty.check_novelty(
        target,
        period_days=nominal_period,
        depth_ppm=primary.get("depth_ppm"),
        stellar_info=stellar_info,
        analysis=primary,
    )
    validation["novelty"] = novelty_result

    _progress("verify", "Running AI self-verification dialog...")

    self_verify = verifier.self_dialog_verify(
        primary, target, validation, stellar_info, physics_data)

    if self_verify.get("final_verdict") == "rejected":
        validation["veto"] = True
        validation["confidence"] = "low"
        validation["reason"] = (
            "AI self-verification rejected: " +
            self_verify.get("summary", "failed verification rounds")
        )
    elif self_verify.get("final_verdict") == "confirmed_known":
        validation["veto"] = True
        validation["confidence"] = "high"
        validation["reason"] = (
            "Signal matches a known planet in the Exoplanet Archive: " +
            self_verify.get("summary", "")
        )
    elif self_verify.get("final_verdict") == "candidate":
        if self_verify.get("confidence", 0) >= 60:
            validation["veto"] = False
            validation["confidence"] = "medium" if self_verify["confidence"] < 80 else "high"

    _progress("plot", "Generating publication-quality figures...")

    gen_plots = plots.generate_all_plots(lc["time"], lc["flux"], primary)

    if len(all_planets) > 1:
        gen_plots["multi_planet_comparison"] = plots.plot_multi_planet(all_planets)

    if mc_results.get("iterations", 0) > 50:
        gen_plots["mc_distributions"] = plots.plot_monte_carlo(mc_results)

    _progress("synthesize", "Writing IMRaD research paper...")

    paper = synthesizer.write_paper(
        target, primary, validation,
        stellar_info=stellar_info,
        physics_data=physics_data,
        plots=gen_plots,
        multi_planet_data=all_planets,
        mc_results=mc_results,
        novelty=novelty_result,
        retrieved_papers=[],  # No arXiv/ADS papers retrieved in transit pipeline yet
    )

    _progress("done", "Pipeline complete.")

    return {
        "target": target,
        "ingestion": {
            "source": lc.get("source", "unknown"),
            "n_points": int(len(lc["time"])),
            "mission_summary": stellar_info.get("mission_summary", {}),
        },
        "stellar_info": stellar_info,
        "analysis": primary,
        "physics": physics_data,
        "validation": validation,
        "self_verification": self_verify,
        "paper": paper,
        "plots": gen_plots,
        "known_planets": known_planets or [],
        "multi_planet_results": all_planets,
        "monte_carlo": mc_results,
        "ttv": ttv_result,
        "flares": flare_result,
        "secondary_eclipse": eclipse_result,
        "ephemeris": ephemeris_result,
        "single_transit": single_transit_result,
    }
