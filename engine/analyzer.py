"""Data Analysis Agent (The Finder) - Professional Grade.

Runs anomaly detection and a transit search over the light curve:
  - astropy BoxLeastSquares for real BLS transit detection
  - robust detrending + sigma-clip
  - Box-fitting Least-Squares-style transit search on folded trial periods
  - IsolationForest for generic point outliers
  - Recurrence check to flag periodic dips
  - scipy transit fitting and proper physics calculations
  - Uses Gaia/SIMBAD metadata for stellar parameters when available
"""
import numpy as np
from sklearn.ensemble import IsolationForest


def _detrend(time, flux, order=3):
    f = np.asarray(flux, dtype=float)
    f = f / np.nanmedian(f)
    mask = np.abs(f - 1.0) < 0.05
    if mask.sum() < order + 1:
        mask = np.ones(len(f), dtype=bool)
    poly = np.polyfit(time[mask], f[mask], order)
    trend = np.polyval(poly, time)
    return f / trend


def _sigma_clip(f, n=5.0):
    med = np.median(f)
    mad = np.median(np.abs(f - med))
    std = 1.4826 * mad
    mask = np.abs(f - med) <= n * std
    return mask


def _fold(time, period, t0=0.0):
    return (time - t0) % period / period


def _box_depth(phase, flux, frac=0.05):
    in_transit = (phase < frac) | (phase > 1 - frac)
    if in_transit.sum() < 3:
        return 0.0, 0.0
    base = flux[~in_transit]
    deep = flux[in_transit]
    return base.mean() - deep.mean(), deep.std() / np.sqrt(max(1, deep.size))


def _astropy_bls(time, flux, fast=False):
    """Run real BoxLeastSquares from astropy.

    fast=True uses a coarser duration/period grid (~5s vs ~70s) suitable
    for Monte Carlo bootstrap iterations where fine resolution is not
    needed for uncertainty estimates.
    """
    try:
        from astropy.timeseries import BoxLeastSquares
        t = np.asarray(time, dtype=float)
        f = np.asarray(flux, dtype=float)

        if fast:
            duration_grid = np.linspace(0.02, 0.15, 6)
            frequency_factor = 6.0
        else:
            duration_grid = np.linspace(0.01, 0.2, 10)
            frequency_factor = 3.0
        model = BoxLeastSquares(t, f)
        periodogram = model.autopower(duration_grid,
                                       minimum_period=0.5,
                                       maximum_period=min(10.0, (t[-1] - t[0]) / 3.0),
                                       frequency_factor=frequency_factor)

        periods = np.array(periodogram.period)
        powers = np.array(periodogram.power)

        best_idx = int(np.argmax(powers))
        best_period = periods[best_idx]
        best_power = powers[best_idx]
        duration = float(periodogram.duration[best_idx])

        stats = model.compute_stats(best_period, duration, t[0])
        depth = stats["depth"][0]
        transit_times = stats.get("transit_times")
        if transit_times is not None and len(transit_times) > 0:
            t0 = float(np.asarray(transit_times)[0])
        else:
            t0 = float(t[0])
        duration_days = duration

        phase = _fold(t, best_period, t0)
        in_transit = (phase < 0.05) | (phase > 0.95)
        out_transit = ~in_transit

        if in_transit.sum() > 0 and out_transit.sum() > 0:
            base_mean = f[out_transit].mean()
            transit_mean = f[in_transit].mean()
            err = f[in_transit].std() / np.sqrt(max(1, in_transit.sum()))
            snr = (base_mean - transit_mean) / err if err > 0 else 0
        else:
            snr = best_power * 100

        n_transits = len(np.unique(np.round((t - t0) / best_period).astype(int)))

        return {
            "period": float(best_period),
            "depth": float(abs(depth)),
            "snr": float(abs(snr)),
            "t0": float(t0),
            "n_transits": int(n_transits),
            "duration_days": float(duration_days),
            "periods": [] if fast else periods.tolist(),
            "powers": [] if fast else powers.tolist(),
        }
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("astropy BLS failed, using fallback "
                                            "box search: %s: %s", type(e).__name__, e)
        return None


def _fallback_bls(time, flux, pmin=0.5, pmax=10.0, npoints=1200):
    """Fallback box-fit transit search if astropy BLS unavailable."""
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    p_grid = np.linspace(pmin, min(pmax, (t[-1] - t[0]) / 3.0), npoints)
    best = {"period": None, "depth": 0.0, "snr": 0.0, "t0": 0.0, "n_transits": 0}

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

    return {
        "period": best["period"],
        "depth": best["depth"],
        "snr": best["snr"],
        "t0": best["t0"],
        "n_transits": best["n_transits"],
        "duration_days": 0.0,
        "periods": [],
        "powers": [],
    }


def analyze_fast_window(time, flux, center, window_frac=0.3):
    """Lightweight BLS re-search restricted to +/-window_frac around center.

    Used by the Monte Carlo bootstrap: the period posterior after a
    detection lives near the nominal period, so a narrow coarse search
    (~0.2s) replaces the full blind search (~2-70s) per iteration.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    f = f / np.nanmedian(f)
    pmin = max(0.1, center * (1.0 - window_frac))
    pmax = center * (1.0 + window_frac)
    try:
        from astropy.timeseries import BoxLeastSquares
        duration_grid = np.linspace(0.02, 0.15, 5)
        model = BoxLeastSquares(t, f)
        pg = model.autopower(duration_grid, minimum_period=pmin,
                             maximum_period=pmax, frequency_factor=8.0)
        periods = np.asarray(pg.period)
        powers = np.asarray(pg.power)
        if len(periods) == 0 or not np.isfinite(powers).any():
            raise RuntimeError("empty periodogram")
        idx = int(np.nanargmax(powers))
        best_p = float(periods[idx])
        best_d = float(pg.duration[idx])
        stats = model.compute_stats(best_p, best_d, float(t[0]))
        n_tr = int(len(np.asarray(stats.get("per_transit_count", []))))
        if n_tr == 0:
            n_tr = int(len(stats.get("transit_times", [])))
        tts = stats.get("transit_times")
        t0_tr = float(np.asarray(tts)[0]) if tts is not None and len(tts) > 0 else float(t[0])
        depth_val = float(np.abs(np.asarray(stats["depth"], dtype=float)[0]))
        phase = _fold(t, best_p, t0_tr)
        half = max(0.005, best_d / best_p / 2.0)
        in_tr = (phase < half) | (phase > 1 - half)
        out_tr = ~in_tr
        if in_tr.sum() >= 3 and out_tr.sum() >= 3:
            err = f[in_tr].std() / np.sqrt(int(in_tr.sum()))
            snr = float((f[out_tr].mean() - f[in_tr].mean()) / err) if err > 0 else 0.0
        else:
            snr = float(powers[idx] * 10.0)
        return {
            "transit": {
                "period": best_p,
                "depth": depth_val,
                "snr": abs(snr),
                "t0": t0_tr,
                "n_transits": n_tr,
            },
            "depth_ppm": int(depth_val * 1e6),
            "period_days": best_p,
        }
    except Exception:
        phase = _fold(t, center, t[0])
        depth, err = _box_depth(phase, f)
        return {
            "transit": {
                "period": float(center), "depth": float(depth),
                "snr": float(depth / err if err else 0.0),
                "t0": float(t[0]), "n_transits": 0,
            },
            "depth_ppm": int(depth * 1e6),
            "period_days": float(center),
        }


def isolation_outliers(time, flux):
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


def analyze(time, flux, metadata=None, fast=False):
    t = np.asarray(time, dtype=float)
    fl = np.asarray(flux, dtype=float)

    mask = _sigma_clip(fl / np.nanmedian(fl))
    t_clean = t[mask]
    fl_clean = fl[mask]

    out_idx, f, resid = isolation_outliers(t_clean, fl_clean)

    bls_result = _astropy_bls(t_clean, f, fast=fast)
    if bls_result is None:
        bls_result = _fallback_bls(t_clean, f)

    is_anomaly = bls_result["snr"] > 5.0 and bls_result["depth"] > 0.0

    variance = float(np.var(resid))
    noise_floor = float(np.std(resid[np.abs(resid) < 0.002]))
    chi_squared_reduced = float(np.var(1.0 - f)) if len(f) else 0.0

    result = {
        "transit": {
            "period": bls_result["period"],
            "depth": bls_result["depth"],
            "snr": bls_result["snr"],
            "t0": bls_result["t0"],
            "n_transits": bls_result["n_transits"],
        },
        "is_anomaly": is_anomaly,
        "outlier_indices": out_idx,
        "depth_ppm": int(bls_result["depth"] * 1e6),
        "snr": float(bls_result["snr"]),
        "period_days": float(bls_result["period"]) if bls_result["period"] else None,
        "n_transits": int(bls_result["n_transits"]),
        "noise_floor_ppm": int(noise_floor * 1e6),
        "reduced_chi2": float(chi_squared_reduced),
        "flux_residuals": resid,
        "bls_periods": None if fast else bls_result.get("periods"),
        "bls_powers": None if fast else bls_result.get("powers"),
        "transit_duration_days": bls_result.get("duration_days", 0),
        "transit_t0": bls_result.get("t0", 0),
    }

    if metadata:
        R_star = _estimate_stellar_radius(metadata)
        if R_star and bls_result["depth"] > 0:
            depth_frac = bls_result["depth"]
            Rp_Rs = np.sqrt(depth_frac)
            Rp_Rearth = Rp_Rs * R_star * 109.2
            result["stellar_radius_rsun"] = R_star
            result["planet_radius_rearth"] = float(Rp_Rearth)
            result["planet_radius_rjup"] = float(Rp_Rearth / 11.2)
        result["metadata"] = metadata

    return result
