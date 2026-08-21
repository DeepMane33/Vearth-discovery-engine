"""Data Analysis Agent (The Finder).

Runs anomaly detection and a transit search over the light curve:
  - robust detrending + sigma-clip
  - Box-fitting Least-Squares-style transit search on folded trial periods
  - IsolationForest for generic point outliers
  - Recurrence check to flag periodic dips
  - Uses Gaia/SIMBAD metadata for stellar parameters when available
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


def _estimate_stellar_radius(metadata):
    """Estimate stellar radius from Gaia parallax and photometry."""
    gaia = metadata.get("gaia", {})
    if not gaia or "parallax" not in gaia or gaia["parallax"] <= 0:
        return None
    g_mag = gaia.get("g_mag")
    bp_mag = gaia.get("bp_mag")
    rp_mag = gaia.get("rp_mag")
    if g_mag is None or bp_mag is None or rp_mag is None:
        return None
    parallax = gaia["parallax"]  # mas
    distance_pc = 1000.0 / parallax
    bp_rp = bp_mag - rp_mag
    abs_g = g_mag - 5 * np.log10(distance_pc / 10.0)
    if bp_rp < 0.5:
        bc = -0.19
    elif bp_rp < 1.0:
        bc = -0.10
    elif bp_rp < 1.5:
        bc = 0.15
    elif bp_rp < 2.0:
        bc = 0.55
    elif bp_rp < 2.5:
        bc = 1.05
    elif bp_rp < 3.0:
        bc = 1.60
    else:
        bc = 2.20
    abs_bol = abs_g + bc
    L_sun = 10 ** ((4.74 - abs_bol) / 2.5)
    if bp_rp < 0.8:
        Teff = 10000
    elif bp_rp < 1.2:
        Teff = 7500
    elif bp_rp < 1.6:
        Teff = 6000
    elif bp_rp < 2.0:
        Teff = 5000
    elif bp_rp < 2.4:
        Teff = 4000
    else:
        Teff = 3300
    R_sun = np.sqrt(L_sun) * (5778 / Teff) ** 2
    return float(R_sun)


def analyze(time, flux, metadata=None):
    t = np.asarray(time, dtype=float)
    fl = np.asarray(flux, dtype=float)

    out_idx, f, resid = isolation_outliers(t, fl)
    hit, f = transit_search(t, fl)
    is_anomaly = hit["snr"] > 5.0 and hit["depth"] > 0.0

    variance = float(np.var(resid))
    noise_floor = float(np.std(resid[np.abs(resid) < 0.002]))
    chi_squared_reduced = float(np.var(1.0 - f)) if len(f) else 0.0

    result = {
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

    if metadata:
        R_star = _estimate_stellar_radius(metadata)
        if R_star and hit["depth"] > 0:
            depth_frac = hit["depth"]
            Rp_Rs = np.sqrt(depth_frac)
            Rp_Rearth = Rp_Rs * R_star * 109.2
            result["stellar_radius_rsun"] = R_star
            result["planet_radius_rearth"] = float(Rp_Rearth)
            result["planet_radius_rjup"] = float(Rp_Rearth / 11.2)
        result["metadata"] = metadata

    return result
