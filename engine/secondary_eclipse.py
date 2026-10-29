"""Secondary Eclipse / Phase Curve Search.

Phase-folds a light curve at a known period and searches for:
  - Secondary eclipse: a dip at phase 0.5 (planet behind the star)
  - Phase curve modulation: brightness variation across the orbit

All numbers come from code; the LLM never invents a detection.
Returns uncertainties and a plain-language limitation note.
"""

from __future__ import annotations

import numpy as np


def _fold(time, period, t0=0.0):
    return (time - t0) % period / period


def search_secondary_eclipse(time_arr, flux_arr, period_days, t0,
                             duration_hours=None, n_phase_bins=30):
    """Search for secondary eclipse and phase curve modulation.

    Args:
        time_arr: 1D array of time (days).
        flux_arr: 1D array of normalized flux (median ~1.0).
        period_days: orbital period (days).
        t0: primary transit mid-time (days). Secondary eclipse expected
            at t0 + period/2.
        duration_hours: expected eclipse duration (hours). If None,
            estimated as 0.1 * period_days.
        n_phase_bins: number of phase bins for the phase curve.

    Returns:
        dict with keys:
            secondary_eclipse: {depth_ppm, depth_uncertainty_ppm,
                phase_center, significance_sigma, detected}
            phase_curve: {amplitude_ppm, max_phase, min_phase,
                bins: [{phase, flux, flux_err}]}
            limitation: plain-language note
    """
    t = np.asarray(time_arr, dtype=float)
    f = np.asarray(flux_arr, dtype=float)

    if len(t) < 50 or period_days is None or period_days <= 0 or t0 is None:
        return _empty_result("insufficient data or invalid period/t0")

    # Normalize flux
    med = np.nanmedian(f)
    if med > 0:
        f = f / med
    else:
        return _empty_result("non-positive median flux")

    if duration_hours is None:
        duration_hours = max(0.1, 0.1 * period_days)
    duration_days = duration_hours / 24.0

    phase = _fold(t, period_days, t0)

    # ---- Secondary eclipse at phase 0.5 ----
    eclipse_phase = 0.5
    eclipse_mask = (np.abs(phase - eclipse_phase) < duration_days / period_days / 2.0)
    if eclipse_mask.sum() < 3:
        eclipse_result = {
            "depth_ppm": None,
            "depth_uncertainty_ppm": None,
            "phase_center": eclipse_phase,
            "significance_sigma": None,
            "detected": False,
            "n_points": int(eclipse_mask.sum()),
            "reason": "too few in-eclipse points",
        }
    else:
        in_eclipse = f[eclipse_mask]
        # Out-of-eclipse baseline: phases 0.35-0.45 and 0.55-0.65
        baseline_mask = (((phase > 0.35) & (phase < 0.45)) |
                         ((phase > 0.55) & (phase < 0.65)))
        if baseline_mask.sum() > 10:
            baseline_flux = float(np.median(f[baseline_mask]))
            baseline_noise = float(np.std(f[baseline_mask]))
        else:
            baseline_flux = 1.0
            baseline_noise = float(np.std(f))

        eclipse_depth = baseline_flux - float(np.median(in_eclipse))
        eclipse_depth_ppm = eclipse_depth * 1e6

        n_ecl = int(eclipse_mask.sum())
        depth_unc_ppm = (baseline_noise / np.sqrt(max(1, n_ecl))) * 1e6
        significance = (eclipse_depth / depth_unc_ppm) if depth_unc_ppm > 0 else 0.0

        eclipse_result = {
            "depth_ppm": round(eclipse_depth_ppm, 2),
            "depth_uncertainty_ppm": round(depth_unc_ppm, 2),
            "phase_center": round(eclipse_phase, 4),
            "significance_sigma": round(significance, 2),
            "detected": bool(significance >= 3.0 and eclipse_depth_ppm > 0),
            "n_points": n_ecl,
            "reason": "" if n_ecl >= 3 else "too few points",
        }

    # ---- Phase curve across full orbit ----
    bin_edges = np.linspace(0.0, 1.0, n_phase_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    phase_bins = []
    for i in range(n_phase_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() > 0:
            phase_bins.append({
                "phase": round(float(bin_centers[i]), 4),
                "flux": round(float(np.median(f[mask])), 6),
                "flux_err": round(float(np.std(f[mask]) / np.sqrt(mask.sum())), 7),
                "n_points": int(mask.sum()),
            })

    if len(phase_bins) > 5:
        fluxes = np.array([b["flux"] for b in phase_bins])
        amplitude_ppm = float((np.max(fluxes) - np.min(fluxes)) * 1e6)
        max_phase = float(bin_centers[int(np.argmax(fluxes))])
        min_phase = float(bin_centers[int(np.argmin(fluxes))])
    else:
        amplitude_ppm = 0.0
        max_phase = 0.0
        min_phase = 0.0

    phase_curve_result = {
        "amplitude_ppm": round(amplitude_ppm, 2),
        "max_phase": round(max_phase, 4),
        "min_phase": round(min_phase, 4),
        "bins": phase_bins,
    }

    return {
        "secondary_eclipse": eclipse_result,
        "phase_curve": phase_curve_result,
        "period_days": float(period_days),
        "t0": float(t0),
        "limitation": (
            "Secondary eclipse depth of ~tens of ppm is near the noise floor "
            "for single-band TESS data. A 3-sigma detection is only a "
            "candidate; phase curve shape can be affected by stellar "
            "variability (spots, rotation) and instrumental systematics. "
            "Confirm with multiple epochs and/or longer-wavelength data."
        ),
    }


def _empty_result(reason: str) -> dict:
    return {
        "secondary_eclipse": {
            "depth_ppm": None,
            "depth_uncertainty_ppm": None,
            "phase_center": 0.5,
            "significance_sigma": None,
            "detected": False,
            "n_points": 0,
            "reason": reason,
        },
        "phase_curve": {
            "amplitude_ppm": 0.0,
            "max_phase": 0.0,
            "min_phase": 0.0,
            "bins": [],
        },
        "period_days": None,
        "t0": None,
        "limitation": (
            "Secondary eclipse / phase curve could not be measured: " +
            reason + " Requires a phase-folded light curve with known period."
        ),
    }
