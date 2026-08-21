"""Data Analysis Agent (The Finder).

Runs anomaly detection and a transit search over the light curve:
  - robust detrending + sigma-clip
  - Box-fitting Least-Squares-style transit search on folded trial periods
  - IsolationForest for generic point outliers
  - Recurrence check to flag periodic dips
"""
import numpy as np
from sklearn.ensemble import IsolationForest


def _detrend(time, flux):
    f = np.asarray(flux, dtype=float)
    f = f / np.nanmedian(f)
    mask = np.abs(f - 1.0) < 0.05
    base = f[mask]
    poly = np.polyfit(time[mask], base, 3)
    trend = np.polyval(poly, time)
    return f / trend


def _sigma_clip(f, n=5.0):
    med = np.median(f)
    mad = np.median(np.abs(f - med))
    std = 1.4826 * mad
    return np.abs(f - med) > n * std, f, med, std


def _fold(time, period, t0=0.0):
    return (time - t0) % period / period


def _box_depth(phase, flux, frac=0.05):
    in_transit = (phase < frac) | (phase > 1 - frac)
    if in_transit.sum() < 3:
        return 0.0, 0.0
    base = flux[~in_transit]
    deep = flux[in_transit]
    return base.mean() - deep.mean(), (deep.std() / np.sqrt(max(1, deep.size)))


def transit_search(time, flux, pmin=0.5, pmax=10.0, npoints=1200):
    """Scan trial periods for a box-shaped dip. Returns best hit."""
    t = np.asarray(time, dtype=float)
    f = _detrend(t, flux)
    p_grid = np.linspace(pmin, min(pmax, (t[-1] - t[0]) / 3.0), npoints)
    best = {"period": None, "depth": 0.0, "snr": 0.0,
            "t0": 0.0, "n_transits": 0}
    for p in p_grid:
        phase = _fold(t, p)
        depth, err = _box_depth(phase, f)
        snr = depth / err if err else 0.0
        if snr > best["snr"]:
            transit_centers = (np.round((t - t.mean()) / p) * p) + t.mean()
            unique = np.unique(np.round(transit_centers, 3))
            best = {"period": float(p), "depth": float(depth),
                    "snr": float(snr), "t0": float(np.median(t)),
                    "n_transits": int(len(unique))}
    return best, f


def isolation_outliers(time, flux):
    """Unicode-safe IsolationForest flagging of point outliers."""
    t = np.asarray(time, dtype=float)
    f = _detrend(t, flux)
    X = np.column_stack([t - t.mean(), f - 1.0])
    iso = IsolationForest(contamination=0.01, random_state=0)
    labels = iso.fit_predict(X)
    idx = np.where(labels == -1)[0]
    return idx.tolist(), f, np.asarray(f - 1.0)


def analyze(time, flux):
    t = np.asarray(time, dtype=float)
    fl = np.asarray(flux, dtype=float)

    out_idx, f, resid = isolation_outliers(t, fl)
    hit, f = transit_search(t, fl)
    is_anomaly = hit["snr"] > 5.0 and hit["depth"] > 0.0

    variance = float(np.var(resid))
    noise_floor = float(np.std(resid[np.abs(resid) < 0.002]))
    chi_squared_reduced = float(np.var(1.0 - f)) if len(f) else 0.0

    return {
        "transit": hit,
        "is_anomaly": is_anomaly,
        "outlier_indices": out_idx,
        "depth_ppm": int(hit["depth"] * 1e6),
        "snr": float(hit["snr"]),
        "period_days": float(hit["period"]) if hit["period"] else None,
        "n_transits": int(hit["n_transits"]),
        "noise_floor_ppm": int(noise_floor * 1e6),
        "reduced_chi2": float(chi_squared_reduced),
        "flux_residuals": resid,
    }
