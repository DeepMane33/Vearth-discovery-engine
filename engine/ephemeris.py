"""Orbital Ephemeris Updater.

Given a known planet's period and reference transit time, re-measures
transit mid-times from a new light curve and fits a linear ephemeris
(T = T0 + n * P). Reports updated period, reference time, and their
uncertainties, plus O-C residuals against the old ephemeris.

All numbers come from code; the LLM never invents an ephemeris update.
Returns uncertainties and a plain-language limitation note.
"""

from __future__ import annotations

import numpy as np


def _find_transit_times(time_arr, flux_arr, period_days, t0,
                        duration_hours=None, n_transits_max=50):
    """Measure individual transit mid-times via flux minimum in window."""
    t = np.asarray(time_arr, dtype=float)
    f = np.asarray(flux_arr, dtype=float)

    if len(t) < 10 or period_days is None or period_days <= 0:
        return []

    med = np.nanmedian(f)
    if med > 0:
        f = f / med

    if duration_hours is None:
        duration_hours = max(0.1, 0.1 * period_days)
    duration_days = duration_hours / 24.0

    t_min, t_max = float(np.min(t)), float(np.max(t))
    n_expected = int((t_max - t_min) / period_days) + 2
    n_expected = min(n_expected, n_transits_max)

    transits = []
    for epoch in range(-n_expected // 2, n_expected // 2 + 1):
        calc_time = t0 + epoch * period_days
        if calc_time < t_min - duration_days or calc_time > t_max + duration_days:
            continue

        window = duration_days * 1.5
        mask = (t >= calc_time - window) & (t <= calc_time + window)
        if mask.sum() < 5:
            continue

        t_win = t[mask]
        f_win = f[mask]
        min_idx = int(np.argmin(f_win))
        obs_time = float(t_win[min_idx])

        out_mask = (t_win < calc_time - 0.5 * duration_days) | (
            t_win > calc_time + 0.5 * duration_days)
        if out_mask.sum() > 5:
            noise = float(np.std(f_win[out_mask]))
        else:
            noise = float(np.std(f_win))

        n_in = int(mask.sum())
        time_uncertainty_days = max(
            0.001, noise * duration_days / np.sqrt(max(1, n_in))
        )

        transits.append({
            "epoch": int(epoch),
            "observed_time": round(obs_time, 6),
            "calculated_time": round(calc_time, 6),
            "oc_days": round(obs_time - calc_time, 6),
            "uncertainty_days": round(time_uncertainty_days, 7),
        })

    return transits


def fit_linear_ephemeris(transit_times):
    """Fit T = T0 + n*P to transit times using weighted least squares.

    Args:
        transit_times: list of dicts with 'epoch', 'observed_time',
            'uncertainty_days'.

    Returns:
        dict with T0, P, their uncertainties, and residuals, or None
        if fewer than 2 transits.
    """
    if len(transit_times) < 2:
        return None

    epochs = np.array([tt["epoch"] for tt in transit_times], dtype=float)
    times = np.array([tt["observed_time"] for tt in transit_times],
                     dtype=float)
    unc = np.array([tt["uncertainty_days"] for tt in transit_times],
                   dtype=float)
    unc = np.where(unc > 0, unc, 0.001)

    # Weighted linear fit: times = T0 + P * epochs
    w = 1.0 / unc ** 2
    sum_w = np.sum(w)
    sum_wx = np.sum(w * epochs)
    sum_wy = np.sum(w * times)
    sum_wxx = np.sum(w * epochs ** 2)
    sum_wxy = np.sum(w * epochs * times)

    delta = sum_w * sum_wxx - sum_wx ** 2
    if abs(delta) < 1e-12:
        return None

    P = (sum_w * sum_wxy - sum_wx * sum_wy) / delta
    T0 = (sum_wxx * sum_wy - sum_wx * sum_wxy) / delta

    # Uncertainties
    sigma_P_sq = sum_w / delta
    sigma_T0_sq = sum_wxx / delta

    # Chi-squared
    model = T0 + P * epochs
    residuals = times - model
    chi2 = np.sum(w * residuals ** 2)
    dof = max(1, len(transit_times) - 2)
    chi2_reduced = chi2 / dof

    return {
        "T0": round(float(T0), 6),
        "P": round(float(P), 6),
        "T0_uncertainty_days": round(float(np.sqrt(sigma_T0_sq)), 7),
        "P_uncertainty_days": round(float(np.sqrt(sigma_P_sq)), 7),
        "chi2_reduced": round(float(chi2_reduced), 4),
        "n_transits": len(transit_times),
        "residuals_days": [round(float(r), 6) for r in residuals],
        "epochs": [int(e) for e in epochs],
    }


def update_ephemeris(time_arr, flux_arr, known_period_days, known_t0,
                     duration_hours=None, n_transits_max=50):
    """Run full ephemeris update pipeline.

    Args:
        time_arr: 1D array of time (days).
        flux_arr: 1D array of normalized flux.
        known_period_days: previously published period (days).
        known_t0: previously published reference transit time (days).
        duration_hours: expected transit duration (hours).
        n_transits_max: maximum transits to measure.

    Returns:
        dict with old ephemeris, new ephemeris, O-C against old ephemeris,
        and improvement assessment.
    """
    transits = _find_transit_times(time_arr, flux_arr, known_period_days,
                                   known_t0, duration_hours, n_transits_max)

    if len(transits) < 2:
        return {
            "old_ephemeris": {
                "P": known_period_days,
                "T0": known_t0,
            },
            "new_ephemeris": None,
            "n_transits_measured": len(transits),
            "oc_against_old": [],
            "improved": False,
            "limitation": (
                "Ephemeris update requires at least 2 measured transits. "
                "Only "
                + str(len(transits))
                + " transit(s) were detected in this light curve."
            ),
        }

    fit = fit_linear_ephemeris(transits)

    # O-C against the OLD ephemeris
    oc_against_old = []
    for tt in transits:
        oc_against_old.append({
            "epoch": tt["epoch"],
            "oc_days": round(tt["oc_days"], 6),
            "oc_minutes": round(tt["oc_days"] * 1440.0, 3),
        })

    # Did the new ephemeris reduce RMS O-C vs old?
    old_rms = float(np.sqrt(np.mean(
        np.array([tt["oc_days"] for tt in transits]) ** 2
    )))
    new_rms = None
    if fit:
        new_rms = float(np.sqrt(np.mean(
            np.array(fit["residuals_days"]) ** 2
        )))
    improved = (new_rms is not None and new_rms < old_rms)

    return {
        "old_ephemeris": {
            "P": known_period_days,
            "T0": known_t0,
        },
        "new_ephemeris": fit,
        "n_transits_measured": len(transits),
        "oc_against_old": oc_against_old,
        "old_rms_oc_days": round(old_rms, 7),
        "new_rms_oc_days": round(new_rms, 7) if new_rms is not None else None,
        "improved": improved,
        "period_days": known_period_days,
        "t0": known_t0,
        "limitation": (
            "Ephemeris refinement uses simple flux-minimum transit timing, "
            "not a full transit model fit. Timing precision is limited by "
            "cadence and photometric noise. TTVs from additional planets "
            "will bias the linear fit; a high reduced chi-squared (>2) "
            "suggests the orbit is not strictly periodic. The updated "
            "ephemeris is valid only over the time baseline of this data."
        ),
    }


def ephemeris_from_known_planet(planet_name: str, period_days: float,
                                t0: float) -> dict:
    """Convenience wrapper: document a known planet's ephemeris for later
    comparison against new observations.

    Args:
        planet_name: planet identifier (e.g. "WASP-18 b").
        period_days: published orbital period (days).
        t0: published reference transit time (days).

    Returns:
        dict documenting the reference ephemeris.
    """
    return {
        "planet_name": planet_name,
        "reference_period_days": float(period_days),
        "reference_t0": float(t0),
        "source": "user-provided (literature value)",
        "note": (
            "These are literature values. Use update_ephemeris() with new "
            "TESS data to refine them."
        ),
    }
