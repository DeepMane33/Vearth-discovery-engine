"""Transit Timing Variation (TTV) Detector.

Compares observed transit mid-times against a linear ephemeris
(period + reference transit time) and reports O-C (observed minus
calculated) residuals. Systematic deviations from a constant period
can indicate gravitational perturbations from additional planets.

All numbers come from code; the LLM is never asked to invent a TTV
signal. Returns uncertainties and a plain-language limitation note.
"""

from __future__ import annotations

import numpy as np


def compute_ttv(time_arr, flux_arr, period_days, t0, duration_hours=None,
                n_transits_max=50):
    """Detect individual transit events and compute O-C residuals.

    Args:
        time_arr: 1D array of time (days).
        flux_arr: 1D array of normalized flux.
        period_days: assumed orbital period (days).
        t0: reference transit mid-time (days).
        duration_hours: expected transit duration (hours). If None,
            estimated as 0.1 * period_days.
        n_transits_max: maximum number of transits to measure.

    Returns:
        dict with keys:
            transits: list of {transit_number, observed_time, calculated_time,
                oc_minutes, uncertainty_minutes}
            rms_oc_minutes: RMS of O-C residuals (minutes)
            max_oc_minutes: maximum |O-C| (minutes)
            n_transits_measured: int
            has_ttv_signal: bool (True if RMS > 3 * median uncertainty)
            limitation: plain-language note
    """
    t = np.asarray(time_arr, dtype=float)
    f = np.asarray(flux_arr, dtype=float)

    if len(t) < 10 or period_days is None or period_days <= 0:
        return _empty_result("insufficient data or invalid period")

    if t0 is None:
        t0 = float(t[0])

    if duration_hours is None:
        duration_hours = max(0.1, 0.1 * period_days)
    duration_days = duration_hours / 24.0

    # Normalize flux
    med = np.nanmedian(f)
    if med > 0:
        f = f / med

    # Find in-transit windows for each expected transit epoch
    t_min, t_max = float(np.min(t)), float(np.max(t))
    n_expected = int((t_max - t_min) / period_days) + 2
    n_expected = min(n_expected, n_transits_max)

    transits = []
    for epoch in range(-n_expected // 2, n_expected // 2 + 1):
        calc_time = t0 + epoch * period_days
        if calc_time < t_min - duration_days or calc_time > t_max + duration_days:
            continue

        # Window around expected transit
        window = duration_days * 1.5
        mask = (t >= calc_time - window) & (t <= calc_time + window)
        if mask.sum() < 5:
            continue

        t_win = t[mask]
        f_win = f[mask]

        # Find flux minimum within window (simple box minimum)
        min_idx = int(np.argmin(f_win))
        obs_time = float(t_win[min_idx])
        min_flux = float(f_win[min_idx])

        # Estimate uncertainty from local noise
        out_mask = (t_win < calc_time - 0.5 * duration_days) | (
            t_win > calc_time + 0.5 * duration_days)
        if out_mask.sum() > 5:
            noise = float(np.std(f_win[out_mask]))
        else:
            noise = float(np.std(f_win))

        # Time uncertainty ~ noise * duration / sqrt(N) scaled to minutes
        n_in = int(mask.sum())
        time_uncertainty_days = max(
            0.001, noise * duration_days / np.sqrt(max(1, n_in))
        )
        time_uncertainty_minutes = time_uncertainty_days * 1440.0

        oc_days = obs_time - calc_time
        oc_minutes = oc_days * 1440.0

        transits.append({
            "transit_number": int(epoch),
            "observed_time": round(obs_time, 6),
            "calculated_time": round(calc_time, 6),
            "oc_minutes": round(oc_minutes, 3),
            "uncertainty_minutes": round(time_uncertainty_minutes, 3),
            "min_flux": round(min_flux, 6),
        })

    if len(transits) < 2:
        return _empty_result("fewer than 2 transits measured")

    oc_values = np.array([tr["oc_minutes"] for tr in transits])
    uncertainties = np.array([tr["uncertainty_minutes"] for tr in transits])

    rms_oc = float(np.sqrt(np.mean(oc_values ** 2)))
    max_oc = float(np.max(np.abs(oc_values)))
    median_unc = float(np.median(uncertainties))

    # A TTV signal is suggested if RMS O-C exceeds 3x median uncertainty
    has_ttv = rms_oc > 3.0 * median_unc and median_unc > 0

    return {
        "transits": transits,
        "rms_oc_minutes": round(rms_oc, 3),
        "max_oc_minutes": round(max_oc, 3),
        "n_transits_measured": len(transits),
        "has_ttv_signal": bool(has_ttv),
        "period_days": float(period_days),
        "t0": float(t0),
        "limitation": (
            "TTV detection here uses a simple flux-minimum within each "
            "expected transit window, not a full transit model fit. "
            "Timing precision is limited by cadence and noise; systematic "
            "instrumental trends can mimic or hide TTVs. A detected RMS "
            "O-C is a candidate signal requiring follow-up photometry "
            "and dynamical modeling before claiming additional planets."
        ),
    }


def _empty_result(reason: str) -> dict:
    return {
        "transits": [],
        "rms_oc_minutes": None,
        "max_oc_minutes": None,
        "n_transits_measured": 0,
        "has_ttv_signal": False,
        "period_days": None,
        "t0": None,
        "limitation": (
            "TTV could not be measured: " + reason +
            " At least two well-observed transits are required."
        ),
    }
