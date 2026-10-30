"""Single-Transit / Long-Period Detector.

Searches for isolated, non-recurring dip events in a light curve that
could be single-transit planets (long-period, one transit in the data
span) or other single events. Finds local flux minima, then tests
box-car depths around each minimum. Reports candidates with
uncertainties.

Does NOT claim periodicity — a single dip cannot establish an orbit.
Reports a plain-language limitation note.
"""

from __future__ import annotations

import numpy as np


def _local_minima(f, window_pts=5):
    """Find indices of local minima in flux array."""
    minima = []
    for i in range(window_pts, len(f) - window_pts):
        if f[i] == np.min(f[i - window_pts:i + window_pts + 1]):
            minima.append(i)
    return minima


def detect_single_transit(time_arr, flux_arr,
                          min_duration_hours=1.0,
                          max_duration_hours=24.0,
                          cadence_minutes=None,
                          sigma_threshold=6.0,
                          max_candidates=10):
    """Detect single-transit-like dips in a light curve.

    Strategy: find local flux minima, then for each minimum test box-car
    widths around it to measure depth and significance. This avoids the
    O(n * width_range) cost of sliding every possible box.

    Args:
        time_arr: 1D array of time (days).
        flux_arr: 1D array of normalized flux (median ~1.0).
        min_duration_hours: minimum transit duration (hours).
        max_duration_hours: maximum transit duration (hours).
        cadence_minutes: observation cadence (minutes). If None, estimated.
        sigma_threshold: detection threshold in units of local noise.
        max_candidates: maximum candidates to return.

    Returns:
        dict with keys: candidates, n_candidates, best, cadence_minutes,
        sigma_threshold, limitation.
    """
    t = np.asarray(time_arr, dtype=float)
    f = np.asarray(flux_arr, dtype=float)

    if len(t) < 50:
        return _empty_result("light curve too short")

    # Normalize flux
    med = np.nanmedian(f)
    if med > 0:
        f = f / med
    else:
        return _empty_result("non-positive median flux")

    # Estimate cadence
    if cadence_minutes is None:
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        cadence_minutes = float(np.median(dt)) * 24.0 * 60.0 if len(dt) > 0 else 30.0
    cadence_days = cadence_minutes / (24.0 * 60.0)

    # Estimate noise via MAD on quiet (faintest 95%) points
    f_sorted = np.sort(f[np.isfinite(f)])
    n_keep = max(10, int(len(f_sorted) * 0.95))
    quiet = f_sorted[:n_keep]
    mad = np.median(np.abs(quiet - np.median(quiet)))
    noise = 1.4826 * mad if mad > 0 else 0.001
    noise = max(noise, 0.0001)

    # Duration grid in points
    min_pts = max(2, int(min_duration_hours * 60 / cadence_minutes))
    max_pts = max(min_pts + 1, int(max_duration_hours * 60 / cadence_minutes))
    max_pts = min(max_pts, len(t) // 2)

    # Find local minima as candidate dip centers
    window_pts = max(3, min_pts // 2)
    minima = _local_minima(f, window_pts=window_pts)

    candidates = []
    used_ranges = []  # [(start, end)] for de-duplication

    for min_idx in minima:
        # Try box widths around this minimum
        best_for_min = None
        for width in range(min_pts, max_pts + 1):
            half = width // 2
            start = max(0, min_idx - half)
            end = min(len(t), start + width)
            if end - start < min_pts:
                continue

            inside = f[start:end]
            # Outside baseline: points immediately adjacent to the box
            outside_left = f[max(0, start - width):start]
            outside_right = f[end:min(len(t), end + width)]
            outside = np.concatenate([outside_left, outside_right])
            if len(outside) < 3:
                continue

            baseline = np.median(outside)
            baseline_noise = np.std(outside) / np.sqrt(len(outside))
            dip_depth = baseline - np.median(inside)
            if dip_depth <= 0:
                continue
            significance = dip_depth / baseline_noise if baseline_noise > 0 else 0
            if significance < sigma_threshold:
                continue

            center_idx = start + (end - start) // 2
            duration_hours = (end - start) * cadence_minutes / 60.0
            depth_ppm = dip_depth * 1e6

            cand = {
                "center_time": round(float(t[center_idx]), 6),
                "depth_relative": round(dip_depth, 6),
                "depth_ppm": round(depth_ppm, 2),
                "duration_hours": round(duration_hours, 2),
                "significance_sigma": round(significance, 2),
                "start_index": int(start),
                "end_index": int(end),
            }

            if best_for_min is None or significance > best_for_min["significance_sigma"]:
                best_for_min = cand

        if best_for_min is not None:
            # De-duplicate: skip if overlaps significantly with existing
            overlaps = False
            for (us, ue) in used_ranges:
                if (best_for_min["start_index"] < ue and
                        best_for_min["end_index"] > us):
                    overlaps = True
                    break
            if not overlaps:
                used_ranges.append((best_for_min["start_index"],
                                    best_for_min["end_index"]))
                candidates.append(best_for_min)

    # Sort by significance descending
    candidates.sort(key=lambda c: c["significance_sigma"], reverse=True)
    candidates = candidates[:max_candidates]

    best = candidates[0] if candidates else None

    return {
        "candidates": candidates,
        "n_candidates": len(candidates),
        "best": best,
        "cadence_minutes": round(cadence_minutes, 2),
        "sigma_threshold": sigma_threshold,
        "limitation": (
            "Single-transit detection finds isolated dips but cannot "
            "establish periodicity or an orbital period. A single dip "
            "could be a planet transit, a stellar flare, an instrumental "
            "glitch, or a cosmic-ray hit. Significance is a simple "
            "box-car depth-vs-baseline measure, not a full transit model "
            "fit. Any candidate requires follow-up photometry to confirm "
            "recurrence before being called a planet."
        ),
    }


def _empty_result(reason: str) -> dict:
    return {
        "candidates": [],
        "n_candidates": 0,
        "best": None,
        "cadence_minutes": None,
        "sigma_threshold": 6.0,
        "limitation": (
            "Single-transit detection could not run: " + reason +
            " A longer, better-sampled light curve is required."
        ),
    }
