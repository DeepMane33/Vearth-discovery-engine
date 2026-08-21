"""Data Ingestion Agent (The Collector).

Pulls TESS light curves from MAST via astroquery, caches locally, and
generates a synthetic light curve as a fall-back so the pipeline runs
even without network access. Also queries SIMBAD, Gaia, and Vizier for
comprehensive target metadata."""
import time
import json
import numpy as np
from pathlib import Path

from config import CACHE_DIR, MAST_CACHE_DAYS


def query_simbad(target):
    """Query SIMBAD for basic target info (coordinates, type, magnitudes)."""
    try:
        from astroquery.simbad import Simbad
        Simbad.add_votable_fields('ra', 'dec', 'otype', 'flux(V)', 'flux(J)', 'flux(K)')
        result = Simbad.query_object(target)
        if result is None:
            return {}
        return {
            "ra": str(result['RA'][0]),
            "dec": str(result['DEC'][0]),
            "otype": str(result['OTYPE'][0]),
            "v_mag": float(result['FLUX_V'][0]) if result['FLUX_V'][0] != '--' else None,
            "j_mag": float(result['FLUX_J'][0]) if result['FLUX_J'][0] != '--' else None,
            "k_mag": float(result['FLUX_K'][0]) if result['FLUX_K'][0] != '--' else None,
        }
    except Exception as e:
        return {"error": f"SIMBAD: {type(e).__name__}"}


def query_gaia(target):
    """Query Gaia DR3 for parallax, proper motion, and photometry."""
    try:
        from astroquery.gaia import Gaia
        Gaia.ROW_LIMIT = 5
        job = Gaia.launch_job_async(
            f"""
            SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag,
                   phot_bp_mean_mag, phot_rp_mean_mag, ruwe
            FROM gaiadr3.gaia_source
            WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                             CIRCLE('ICRS', 
                                    (SELECT ra FROM gaiadr3.gaia_source WHERE source_id =
                                     (SELECT source_id FROM gaiadr3.gaia_source 
                                      WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                                                     CIRCLE('ICRS', 
                                                            (SELECT ra FROM gaiadr3.gaia_source 
                                                             WHERE source_name='{target}'
                                                             LIMIT 1), 5/3600)) LIMIT 1),
                                    5/3600))
            ORDER BY parallax DESC
            """
        )
        results = job.get_results()
        if len(results) == 0:
            return {}
        row = results[0]
        return {
            "source_id": int(row['source_id']),
            "ra": float(row['ra']),
            "dec": float(row['dec']),
            "parallax": float(row['parallax']),
            "pmra": float(row['pmra']),
            "pmdec": float(row['pmdec']),
            "g_mag": float(row['phot_g_mean_mag']),
            "bp_mag": float(row['phot_bp_mean_mag']),
            "rp_mag": float(row['phot_rp_mean_mag']),
            "ruwe": float(row['ruwe']),
        }
    except Exception as e:
        return {"error": f"Gaia: {type(e).__name__}"}


def query_vizier(target, radius_arcsec=30):
    """Query Vizier catalog for known exoplanets, binaries, variables near target."""
    try:
        from astroquery.vizier import Vizier
        from astroquery.simbad import Simbad
        Simbad.add_votable_fields('ra', 'dec')
        coord = Simbad.query_object(target)
        if coord is None:
            return {}
        ra, dec = str(coord['RA'][0]), str(coord['DEC'][0])
        Vizier.ROW_LIMIT = 20
        catalogs = Vizier.query_region(
            f"{ra} {dec}",
            radius=f"{radius_arcsec}s",
            catalog=["B/exoplanet", "B/ebcat", "B/vsx", "J/ApJS/211/24"],
        )
        out = []
        for cat in catalogs:
            for row in cat:
                out.append({k: row[k] for k in row.colnames if row[k] != '--'})
        return {"matches": out[:10]}
    except Exception as e:
        return {"error": f"Vizier: {type(e).__name__}"}


def fetch_comprehensive_metadata(target):
    """Fetch metadata from all available astroquery services."""
    return {
        "simbad": query_simbad(target),
        "gaia": query_gaia(target),
        "vizier": query_vizier(target),
    }


def _manifest():
    f = CACHE_DIR / "ingest_manifest.json"
    if f.exists():
        return json.loads(f.read_text())
    return {}


def _save_manifest(m):
    (CACHE_DIR / "ingest_manifest.json").write_text(json.dumps(m, indent=2))


def _manifest():
    f = CACHE_DIR / "ingest_manifest.json"
    if f.exists():
        return json.loads(f.read_text())
    return {}


def _save_manifest(m):
    (CACHE_DIR / "ingest_manifest.json").write_text(json.dumps(m, indent=2))


def fetch_tess_light_curve(target="TIC 268684168", n_samples=2000):
    """Download a TESS light curve from MAST. Returns dict of arrays.
    Falls back to synthetic data if offline / astroquery missing.
    Also attaches comprehensive metadata from SIMBAD, Gaia, Vizier."""
    key = target.replace(" ", "_")
    out = CACHE_DIR / f"{key}.npz"

    m = _manifest()
    if out.exists() and (time.time() - m.get(key, 0)) < MAST_CACHE_DAYS * 86400:
        z = np.load(out)
        result = {k: z[k] for k in z.files}
        result["metadata"] = fetch_comprehensive_metadata(target)
        return result

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
        result = {"time": t, "flux": f, "target": target, "source": "mast_tess"}
        result["metadata"] = fetch_comprehensive_metadata(target)
        return result
    except Exception as e:
        lc = _synthetic_light_curve(n_samples, seed=hash(target) % 2 ** 31)
        lc["source"] = "synthetic"
        lc["note"] = f"synthetic fallback ({type(e).__name__}: {e})"
        np.savez(out, **lc)
        m[key] = time.time()
        _save_manifest(m)
        lc["metadata"] = fetch_comprehensive_metadata(target)
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
