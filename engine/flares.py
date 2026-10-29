"""Stellar Flare Detector.

Scans a light curve for sudden brightening events (flares): a rapid flux
increase followed by a slower exponential decay. Reports the number of
flares, peak amplitudes, and durations. All numbers come from code; the
LLM never invents flare events.

Returns uncertainties and a plain-language limitation note.
"""

from __future__ import annotations

import numpy as np


def detect_flares(time_arr, flux_arr, sigma_threshold=6.0,
                  min_duration_points=3, max_duration_points=200,
                  rise_points_max=10):
    """Detect stellar flares in a normalized light curve.

    Args:
        time_arr: 1D array of time (days).
        flux_arr: 1D array of normalized flux (median ~1.0).
        sigma_threshold: detection threshold in units of local noise (sigma).
        min_duration_points: minimum points above threshold for a valid flare.
        max_duration_points: maximum points above threshold (rejects trends).
        rise_points_max: max points allowed for the initial rise (enforces
            rapid onset typical of flares).

    Returns:
        dict with keys:
            flares: list of {peak_time, peak_flux, amplitude_relative,
                duration_days, noise_sigma, start_index, end_index}
            n_flares: int
            median_noise: float (relative flux units)
            limitation: plain-language note
    """
    t = np.asarray(time_arr, dtype=float)
    f = np.asarray(flux_arr, dtype=float)

    if len(t) < 50:
        return _empty_result("light curve too short")

    # Normalize to median = 1
    med = np.nanmedian(f)
    if med > 0:
        f = f / med
    else:
        return _empty_result("non-positive median flux")

    # Estimate noise robustly from the full light curve using MAD.
    # Exclude the brightest 5% of points to avoid flare contamination.
    f_sorted = np.sort(f[np.isfinite(f)])
    n_keep = max(10, int(len(f_sorted) * 0.95))
    quiet = f_sorted[:n_keep]
    med_quiet = np.median(quiet)
    mad = np.median(np.abs(quiet - med_quiet))
    global_noise = 1.4826 * mad if mad > 0 else 0.001
    global_noise = max(global_noise, 0.0001)

    # Find points exceeding threshold (relative to median=1 baseline)
    above = (f - 1.0) > (sigma_threshold * global_noise)

    # Group contiguous above-threshold points into candidate flares
    flares = []
    i = 0
    while i < len(f):
        if above[i]:
            start = i
            while i < len(f) and above[i]:
                i += 1
            end = i - 1
            duration_pts = end - start + 1

            if duration_pts < min_duration_points or duration_pts > max_duration_points:
                continue

            # Peak within the candidate region
            region = f[start:end + 1]
            peak_idx_rel = int(np.argmax(region))
            peak_idx = start + peak_idx_rel
            peak_flux = float(region[peak_idx_rel])
            peak_time = float(t[peak_idx])

            # Amplitude relative to local baseline
            baseline = 1.0
            amplitude = peak_flux - baseline

            # Rise time: points from start to peak
            rise_pts = peak_idx - start + 1

            # Require rapid rise (not a slow trend)
            if rise_pts > rise_points_max:
                continue

            flares.append({
                "peak_time": round(peak_time, 6),
                "peak_flux": round(peak_flux, 6),
                "amplitude_relative": round(amplitude, 6),
                "duration_days": round(float(t[end] - t[start]), 6),
                "noise_sigma": round(float(global_noise), 6),
                "start_index": int(start),
                "end_index": int(end),
                "rise_points": int(rise_pts),
            })
        else:
            i += 1

    # Merge flares closer than 1 cadence apart (likely same event split)
    merged = []
    for fl in flares:
        if merged and (fl["start_index"] - merged[-1]["end_index"]) <= 2:
            prev = merged[-1]
            if fl["peak_flux"] > prev["peak_flux"]:
                merged[-1] = fl
        else:
            merged.append(fl)
    flares = merged

    median_noise = float(global_noise)

    return {
        "flares": flares,
        "n_flares": len(flares),
        "median_noise": round(median_noise, 7),
        "sigma_threshold": sigma_threshold,
        "limitation": (
            "Flare detection uses a simple threshold-plus-rapid-rise "
            "criterion on a normalized light curve. Slow instrumental "
            "trends, cosmic-ray hits, and detector systematics can produce "
            "false positives; genuine flares should be confirmed by their "
            "characteristic fast-rise/slow-decay shape and by checking for "
            "correlated events in nearby pixels. Amplitude is reported in "
            "relative flux units, not physical energy."
        ),
    }


def _empty_result(reason: str) -> dict:
    return {
        "flares": [],
        "n_flares": 0,
        "median_noise": None,
        "sigma_threshold": 6.0,
        "limitation": (
            "Flare detection could not run: " + reason +
            " A longer, better-sampled light curve is required."
        ),
    }
