"""Data Ingestion Agent (The Collector).

Pulls TESS light curves from MAST via astroquery, caches locally, and
generates a synthetic light curve as a fall-back so the pipeline runs
even without network access."""
import time
import json
import numpy as np
from pathlib import Path

from config import CACHE_DIR, MAST_CACHE_DAYS


def _manifest():
    f = CACHE_DIR / "ingest_manifest.json"
    if f.exists():
        return json.loads(f.read_text())
    return {}


def _save_manifest(m):
    (CACHE_DIR / "ingest_manifest.json").write_text(json.dumps(m, indent=2))


def fetch_tess_light_curve(target="TIC 268684168", n_samples=2000):
    """Download a TESS light curve from MAST. Returns dict of arrays.
    Falls back to synthetic data if offline / astroquery missing."""
    key = target.replace(" ", "_")
    out = CACHE_DIR / f"{key}.npz"

    m = _manifest()
    if out.exists() and (time.time() - m.get(key, 0)) < MAST_CACHE_DAYS * 86400:
        z = np.load(out)
        return {k: z[k] for k in z.files}

    try:
        from astroquery.mast import Observations
        from astropy.io import fits
        from astropy.table import Table

        obs = Observations.query_criteria(objectname=target,
                                           obs_collection="TESS",
                                           dataproduct_type="timeseries")
        if len(obs) == 0:
            raise RuntimeError("no TESS observations for target")
        prod = Observations.get_product_list(obs[0])
        lc_rows = prod[prod["productSubGroupDescription"] == "LC"]
        if len(lc_rows) == 0:
            lc_rows = prod
        rows = Observations.download_products(lc_rows[0], download_dir=str(CACHE_DIR))
        path = rows["Local Path"][0]
        with fits.open(path) as hdul:
            t = f = None
            for h in hdul:
                if not hasattr(h, "columns"):
                    continue
                cols = set(c.name for c in h.columns)
                if "TIME" in cols and ("PDCSAP_FLUX" in cols or "SAP_FLUX" in cols):
                    t = np.array(h.data["TIME"], dtype=float)
                    if "PDCSAP_FLUX" in cols:
                        pdc = np.array(h.data["PDCSAP_FLUX"], dtype=float)
                        f = pdc if np.isfinite(pdc).sum() > 10 else np.array(
                            h.data["SAP_FLUX"], dtype=float)
                    else:
                        f = np.array(h.data["SAP_FLUX"], dtype=float)
                    break
            if t is None:
                tab = Table.read(path)
                t = np.array(tab["TIME"], dtype=float)
                f = np.array(tab["SAP_FLUX"], dtype=float)

        mask = np.isfinite(t) & np.isfinite(f)
        t, f = t[mask], f[mask]
        if len(t) > n_samples:
            idx = np.linspace(0, len(t) - 1, n_samples).astype(int)
            t, f = t[idx], f[idx]
        np.savez(out, time=t, flux=f, target=target, source="mast_tess")
        m[key] = time.time()
        _save_manifest(m)
        return {"time": t, "flux": f, "target": target, "source": "mast_tess"}
    except Exception as e:
        lc = _synthetic_light_curve(n_samples, seed=hash(target) % 2 ** 31)
        lc["source"] = "synthetic"
        lc["note"] = f"synthetic fallback ({type(e).__name__}: {e})"
        np.savez(out, **lc)
        m[key] = time.time()
        _save_manifest(m)
        return lc


def _synthetic_light_curve(n=2000, period=3.141, depth=0.008, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 27.0, n)
    flux = 1.0 + 0.0006 * rng.standard_normal(n)
    trend = 0.0002 * np.sin(2 * np.pi * t / 9.5)
    flux += trend
    for i, ti in enumerate(t):
        phase = (ti % period) / period
        if phase < 0.012 or phase > 0.988:
            flux[i] -= depth
    flux += 0.0003 * rng.standard_normal(n)
    return {"time": t, "flux": flux, "target": "synthetic",
            "source": "synthetic"}


def list_known_targets():
    return ["TIC 268684168", "TIC 352359478", "TOI 700 d",
            "Pi Mensae c", "WASP-18 b", "HD 209458 b"]
