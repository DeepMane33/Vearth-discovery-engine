"""Multi-Source Data Ingestion Layer.

Fetches astronomical data from free, no-key-required APIs:
  - NASA Exoplanet Archive (TAP) for known planets
  - SIMBAD for stellar parameters
  - VizieR for catalog cross-matching
  - arXiv for latest literature
"""
import json
import time
import logging
import requests
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
SIMBAD_TAP = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"
EXOARCHIVE_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
VIZIER_TAP = "https://vizier.cds.unistra.fr/viz-bin/votable"

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_get(key, max_age_hours=72):
    fp = _CACHE_DIR / f"ds_{key}.json"
    if fp.exists():
        age = time.time() - fp.stat().st_mtime
        if age < max_age_hours * 3600:
            try:
                return json.loads(fp.read_text())
            except Exception:
                pass
    return None


def _cache_set(key, data):
    fp = _CACHE_DIR / f"ds_{key}.json"
    try:
        fp.write_text(json.dumps(data, indent=2, default=str))
    except Exception:
        pass


def query_exoplanet_archive(adql_query, timeout=30):
    """Execute an ADQL query against the NASA Exoplanet Archive TAP service.
    Returns list of dicts, or empty list on failure."""
    cache_key = "exoarchive_" + str(abs(hash(adql_query)))[:12]
    cached = _cache_get(cache_key, max_age_hours=24)
    if cached is not None:
        return cached
    try:
        resp = requests.get(EXOARCHIVE_TAP, params={
            "query": adql_query,
            "format": "json",
        }, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "data" in data:
            result = data["data"]
        elif isinstance(data, list):
            result = data
        else:
            result = []
        _cache_set(cache_key, result)
        return result
    except Exception as e:
        log.warning("Exoplanet Archive query failed: %s", e)
        return []


def get_known_planets_near(ra_deg, dec_deg, radius_deg=0.1):
    """Query confirmed exoplanets within radius_deg of (ra, dec)."""
    query = (
        f"SELECT pl_name,hostname,pl_orbper,pl_rade,pl_bmasse,pl_eqt,"
        f"pl_orbsmax,discoverymethod,ra,dec,disc_facility "
        f"FROM ps WHERE ra BETWEEN {ra_deg - radius_deg} AND {ra_deg + radius_deg} "
        f"AND dec BETWEEN {dec_deg - radius_deg} AND {dec_deg + radius_deg}"
    )
    return query_exoplanet_archive(query)


def get_known_planet_by_period(period_days, tolerance=0.01):
    """Query confirmed exoplanets with matching orbital period."""
    query = (
        f"SELECT pl_name,hostname,pl_orbper,pl_rade,pl_bmasse,pl_eqt,"
        f"pl_orbsmax,discoverymethod,disc_facility "
        f"FROM ps WHERE pl_orbper BETWEEN {period_days - tolerance} "
        f"AND {period_days + tolerance}"
    )
    return query_exoplanet_archive(query)


def get_stellar_params(target_name):
    """Fetch stellar parameters from the Exoplanet Archive for a target star."""
    query = (
        f"SELECT pl_hostname,pl_star_teff,pl_starlogg,pl_star_feh,"
        f"pl_star_radius,pl_star_mass "
        f"FROM pscomppars WHERE pl_hostname='{target_name}' LIMIT 1"
    )
    results = query_exoplanet_archive(query)
    if results:
        return results[0]
    return None


def query_simbad(target_name):
    """Query SIMBAD for basic stellar information.
    Returns dict with ra, dec, type, magnitude, or empty dict."""
    cache_key = "simbad_" + target_name.replace(" ", "_")
    cached = _cache_get(cache_key, max_age_hours=168)
    if cached is not None:
        return cached
    try:
        query = (
            f"SELECT main_id, ra, dec, otype, flux(V), flux(B) "
            f"FROM basic AS b "
            f"JOIN mesVai AS v ON b.oid = v.oidref "
            f"WHERE b.main_id = '{target_name}' "
            f"OR b.main_id LIKE '%{target_name}%' "
            f"OR b.alias = '{target_name}' "
            f"TOP 1"
        )
        resp = requests.get(SIMBAD_TAP, params={
            "request": "doQuery",
            "lang": "adql",
            "format": "json",
            "query": query,
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data and data["data"]:
            row = data["data"][0]
            cols = [c["name"] for c in data.get("metadata", [])]
            result = dict(zip(cols, row)) if cols else {}
            _cache_set(cache_key, result)
            return result
        return {}
    except Exception as e:
        log.warning("SIMBAD query failed for %s: %s", target_name, e)
        return {}


def query_simbad_coordinates(target_name):
    """Get RA/Dec from SIMBAD using a simpler name resolver."""
    cache_key = "simbad_coords_" + target_name.replace(" ", "_")
    cached = _cache_get(cache_key, max_age_hours=168)
    if cached is not None:
        return cached
    try:
        url = "https://simbad.u-strasbg.fr/simbad/sim-id"
        resp = requests.get(url, params={
            "Ident": target_name,
            "output.format": "json",
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("data", {})
        coords = {}
        if "main_id" in result:
            coords["main_id"] = result["main_id"]
        if "coordinates" in result:
            c = result["coordinates"]
            coords["ra"] = c.get("ra")
            coords["dec"] = c.get("dec")
            coords["dec_dms"] = c.get("dec_dms")
            coords["ra_dms"] = c.get("ra_dms")
        if "object_type" in result:
            coords["otype"] = result["object_type"].get("long_txt", "")
        if "fluxes" in result:
            for f in result["fluxes"]:
                if f.get("filter") == "V":
                    coords["mag_v"] = f.get("flux")
                if f.get("filter") == "B":
                    coords["mag_b"] = f.get("flux")
        _cache_set(cache_key, coords)
        return coords
    except Exception as e:
        log.warning("SIMBAD coordinate query failed for %s: %s", target_name, e)
        return {}


def search_arxiv(query, max_results=5):
    """Search arXiv for recent astrophysics papers. Returns list of dicts."""
    cache_key = "arxiv_" + str(abs(hash(query)))[:12]
    cached = _cache_get(cache_key, max_age_hours=48)
    if cached is not None:
        return cached
    try:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_API, params=params, timeout=8)
        resp.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)
            authors = [a.find("atom:name", ns).text
                       for a in entry.findall("atom:author", ns)]
            papers.append({
                "title": (title_el.text or "").strip() if title_el is not None else "",
                "abstract": (summary_el.text or "").strip() if summary_el is not None else "",
                "url": id_el.text.strip() if id_el is not None else "",
                "published": published_el.text.strip() if published_el is not None else "",
                "authors": authors[:3],
            })
        _cache_set(cache_key, papers)
        return papers
    except Exception as e:
        log.warning("arXiv search failed: %s", e)
        return []


GAIA_DR3_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
TIC_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def get_gaia_params(target_name):
    """Fetch stellar parameters from Gaia DR3 via TAP.

    Returns parallax, radius, teff, logg, feh, distance, luminosity.
    """
    cache_key = "gaia_" + target_name.replace(" ", "_")
    cached = _cache_get(cache_key, max_age_hours=168)
    if cached is not None:
        return cached

    try:
        coords = query_simbad_coordinates(target_name)
        ra = coords.get("ra")
        dec = coords.get("dec")
        if not ra or not dec:
            return None

        query = (
            f"SELECT TOP 1 "
            f"source_id, ra, dec, parallax, parallax_over_error, "
            f"phot_g_mean_mag, bp_rp, teff_gspphot, logg_gspphot, "
            f"mh_gspphot, radius_gspphot, luminosity_gspphot, "
            f"ruwe, distance_gspphot "
            f"FROM gaiadr3.gaia_source "
            f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.01)) "
            f"ORDER BY parallax_over_error DESC"
        )

        resp = requests.get(GAIA_DR3_TAP, params={
            "request": "doQuery",
            "lang": "adql",
            "format": "json",
            "query": query,
        }, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if "data" in data and data["data"]:
            row = data["data"][0]
            cols = [c["name"] for c in data.get("metadata", [])]
            result = dict(zip(cols, row)) if cols else {}
            result["source"] = "gaia_dr3"
            _cache_set(cache_key, result)
            return result
        return None
    except Exception as e:
        log.warning("Gaia DR3 query failed for %s: %s", target_name, e)
        return None


def get_tic_params(target_name):
    """Fetch TESS Input Catalog (TIC) parameters.

    Returns crowding, contamination ratio, TIC ID, and stellar properties.
    """
    cache_key = "tic_" + target_name.replace(" ", "_")
    cached = _cache_get(cache_key, max_age_hours=168)
    if cached is not None:
        return cached

    try:
        coords = query_simbad_coordinates(target_name)
        ra = coords.get("ra")
        dec = coords.get("dec")
        if not ra or not dec:
            return None

        query = (
            f"SELECT TOP 1 "
            f"ticid, ra, dec, mag, teff, logg, mh, rad, mass, "
            f"rho, lumclass, dist, epoch, congregation, "
            f"e_irname, e_angsep "
            f"FROM tic_v8 "
            f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.005)) "
            f"ORDER BY mag ASC"
        )

        resp = requests.get(TIC_TAP, params={
            "request": "doQuery",
            "lang": "adql",
            "format": "json",
            "query": query,
        }, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if "data" in data and data["data"]:
            row = data["data"][0]
            cols = [c["name"] for c in data.get("metadata", [])]
            result = dict(zip(cols, row)) if cols else {}
            result["source"] = "tic_v8"
            _cache_set(cache_key, result)
            return result
        return None
    except Exception as e:
        log.warning("TIC query failed for %s: %s", target_name, e)
        return None


def check_kepler_k2(target_name):
    """Check if target was observed by Kepler or K2.

    Returns observation info if found, None otherwise.
    """
    cache_key = "kepler_" + target_name.replace(" ", "_")
    cached = _cache_get(cache_key, max_age_hours=168)
    if cached is not None:
        return cached

    try:
        coords = query_simbad_coordinates(target_name)
        ra = coords.get("ra")
        dec = coords.get("dec")
        if not ra or not dec:
            return None

        query = (
            f"SELECT TOP 3 "
            f"ktwo_epic_num, kic_kepler_id, kic_star_name, "
            f"kic_ra, kic_dec, kic_kepmag, "
            f"campaign, quarter, "
            f"kic_teff, kiclogg, kic_feh, kic_radius "
            f"FROM kepler_v10 "
            f"WHERE 1=CONTAINS(POINT('ICRS', kic_ra, kic_dec), CIRCLE('ICRS', {ra}, {dec}, 0.01)) "
            f"ORDER BY kic_kepmag ASC"
        )

        resp = requests.get(TIC_TAP, params={
            "request": "doQuery",
            "lang": "adql",
            "format": "json",
            "query": query,
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if "data" in data and data["data"]:
            results = []
            for row in data["data"][:3]:
                cols = [c["name"] for c in data.get("metadata", [])]
                result = dict(zip(cols, row)) if cols else {}
                results.append(result)
            _cache_set(cache_key, results)
            return results
        _cache_set(cache_key, [])
        return []
    except Exception as e:
        log.warning("Kepler/K2 query failed for %s: %s", target_name, e)
        return []


def multi_mission_summary(target_name):
    """Comprehensive multi-mission availability check.

    Returns a summary of which missions have observed this target.
    """
    summary = {
        "target": target_name,
        "tess": True,
        "kepler_k2": [],
        "gaia": None,
        "tic": None,
    }

    try:
        kepler_data = check_kepler_k2(target_name)
        summary["kepler_k2"] = kepler_data if kepler_data else []
    except Exception:
        summary["kepler_k2"] = []

    try:
        gaia_data = get_gaia_params(target_name)
        summary["gaia"] = gaia_data
    except Exception:
        summary["gaia"] = None

    try:
        tic_data = get_tic_params(target_name)
        summary["tic"] = tic_data
    except Exception:
        summary["tic"] = None

    summary["mission_count"] = (
        1 +
        (1 if summary["kepler_k2"] else 0) +
        (1 if summary["gaia"] else 0) +
        (1 if summary["tic"] else 0)
    )
    summary["available_missions"] = ["TESS"]
    if summary["kepler_k2"]:
        for obs in summary["kepler_k2"]:
            if obs.get("kic_kepler_id"):
                summary["available_missions"].append("Kepler")
            elif obs.get("ktwo_epic_num"):
                summary["available_missions"].append("K2")
    if summary["gaia"]:
        summary["available_missions"].append("Gaia DR3")
    if summary["tic"]:
        summary["available_missions"].append("TIC")

    return summary


def searchLiteratureForTarget(target_name, star_name=None):
    """Combined literature search: arXiv + cross-reference with known planets."""
    results = {"arxiv": [], "known_planets": [], "stellar_params": None}

    search_terms = f"{target_name} exoplanet transit"
    if star_name and star_name != target_name:
        search_terms = f"{star_name} {target_name} exoplanet"
    results["arxiv"] = search_arxiv(search_terms, max_results=5)

    coords = query_simbad_coordinates(star_name or target_name)
    if coords.get("ra") and coords.get("dec"):
        try:
            ra = float(coords["ra"])
            dec = float(coords["dec"])
            results["known_planets"] = get_known_planets_near(ra, dec, radius_deg=0.15)
        except (ValueError, TypeError):
            pass

    results["stellar_params"] = get_stellar_params(star_name or target_name)
    results["simbad"] = coords

    return results
