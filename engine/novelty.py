"""Novelty Checker for both research tracks.

Verifies that a detected signal is not already a known planet, TOI, or
published result before it can be described as a new discovery.

Queries public sources by ID/coordinates/body+dataset — never free text.

Track A (stars/exoplanets):
  - NASA Exoplanet Archive (confirmed planets + TOI catalog by TIC ID)
  - SIMBAD (target identification / coordinates)
  - arXiv (literature mentioning the target)
  - NASA ADS (if API token configured)

Track B (solar-system bodies):
  - NASA ADS (literature mentioning body + dataset, if token)
  - arXiv (literature mentioning body + dataset)

Returns one of: KNOWN, POSSIBLY KNOWN, NO MATCH FOUND.

"No match found" is never worded as "confirmed new discovery."
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from config import NASA_ADS_TOKEN
from engine import datasources
from engine.paper_ingestion import search_ads


def _has_ads_token() -> bool:
    return bool(NASA_ADS_TOKEN and NASA_ADS_TOKEN.strip())


def _clean_title(title: str | None) -> str | None:
    """Return title if non-empty, else None (drop empty titles)."""
    if not title:
        return None
    t = title.strip()
    return t if t else None


def _check_exoplanet_archive(tic_id: str | None,
                             ra: float | None,
                             dec: float | None,
                             period_days: float | None) -> dict:
    """Query NASA Exoplanet Archive for known planets and TOIs.

    Queries by TIC ID (TOI catalog) and by sky position (known planets).
    Never uses free-text target name.
    """
    result: dict = {"matches": [], "source": "nasa_exoplanet_archive"}

    # TOI catalog by TIC ID
    if tic_id:
        try:
            for table in ("toi", "toi_cat", "toi_data"):
                query = (
                    "SELECT tic_id, toi_id, disposition, pl_orbper, pl_rade, "
                    "pl_eqt, ra, dec FROM " + table + " "
                    f"WHERE tic_id = {tic_id}"
                )
                matches = datasources.query_exoplanet_archive(query)
                if matches:
                    result["matches"] = matches[:10]
                    result["table_used"] = table
                    break
        except Exception as e:
            result["toi_error"] = f"{type(e).__name__}: {e}"

    # Known planets near sky coordinates
    if ra is not None and dec is not None:
        try:
            planets = datasources.get_known_planets_near(
                float(ra), float(dec), radius_deg=0.15
            )
            if planets:
                result["nearby_planets"] = planets[:10]
        except Exception as e:
            result["nearby_error"] = f"{type(e).__name__}: {e}"

    # Known planet by period
    if period_days:
        try:
            by_period = datasources.get_known_planet_by_period(
                float(period_days), tolerance=0.01
            )
            if by_period:
                result["period_matches"] = by_period[:5]
        except Exception as e:
            result["period_error"] = f"{type(e).__name__}: {e}"

    return result


def _check_simbad(target: str) -> dict:
    """Query SIMBAD for target identification and coordinates."""
    result: dict = {"info": {}, "source": "simbad"}
    try:
        coords = datasources.query_simbad_coordinates(target)
        if coords:
            result["info"] = coords
            result["ra"] = coords.get("ra")
            result["dec"] = coords.get("dec")
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def _check_arxiv(query_terms: list[str], max_results: int = 5) -> dict:
    """Search arXiv for published papers.

    query_terms: list of terms combined with AND logic. Never free-text
    target name alone — always paired with context (body, dataset, etc.).
    """
    result: dict = {"papers": [], "source": "arxiv"}
    try:
        query = " AND ".join(f'all:{t}' for t in query_terms if t)
        if not query:
            return result
        papers = datasources.search_arxiv(query, max_results=max_results)
        # Drop papers with empty titles
        clean = []
        for p in papers:
            title = _clean_title(p.get("title"))
            if title:
                p["title"] = title
                clean.append(p)
        result["papers"] = clean[:max_results]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def _check_ads(query_terms: list[str], max_results: int = 5) -> dict:
    """Search NASA ADS for published papers (requires API token)."""
    result: dict = {"papers": [], "source": "nasa_ads", "skipped": False}
    if not _has_ads_token():
        result["skipped"] = True
        result["reason"] = "no NASA ADS API token configured"
        return result
    try:
        token = NASA_ADS_TOKEN.strip()
        query = " AND ".join(f'all:{t}' for t in query_terms if t)
        if not query:
            return result
        papers = search_ads(
            query, rows=max_results, token=token
        )
        clean = []
        for p in papers:
            title = _clean_title(p.get("title"))
            if title:
                p["title"] = title
                clean.append(p)
        result["papers"] = clean[:max_results]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def check_novelty_track_a(target: str,
                          period_days: float | None = None,
                          depth_ppm: float | None = None,
                          stellar_info: dict | None = None,
                          analysis: dict | None = None) -> dict:
    """Run novelty checks for Track A (stars / exoplanets).

    Queries by TIC ID and sky coordinates — never free-text target name.
    """
    checked: list[str] = []
    references: list[dict] = []
    details: dict = {}

    # Resolve target to TIC ID + coordinates via SIMBAD / MAST
    tic_id = None
    ra = None
    dec = None
    try:
        from engine.datasources import get_tic_params, query_simbad_coordinates
        tic = get_tic_params(target)
        if tic and tic.get("ticid"):
            tic_id = str(tic["ticid"])
        coords = query_simbad_coordinates(target)
        if coords:
            ra = coords.get("ra")
            dec = coords.get("dec")
            if ra:
                try:
                    ra = float(ra)
                except (TypeError, ValueError):
                    ra = None
            if dec:
                try:
                    dec = float(dec)
                except (TypeError, ValueError):
                    dec = None
    except Exception:
        pass

    # 1. NASA Exoplanet Archive (TOI catalog + known planets)
    archive = _check_exoplanet_archive(tic_id, ra, dec, period_days)
    checked.append("nasa_exoplanet_archive")
    details["nasa_exoplanet_archive"] = archive

    confirmed = archive.get("matches", []) + archive.get("nearby_planets", [])
    toi_matches = archive.get("matches", [])
    period_matches = archive.get("period_matches", [])

    # 2. SIMBAD
    simbad = _check_simbad(target)
    checked.append("simbad")
    details["simbad"] = simbad

    # 3. arXiv (target + exoplanet + transit context)
    arxiv = _check_arxiv([target, "exoplanet", "transit"])
    checked.append("arxiv")
    details["arxiv"] = arxiv

    # 4. NASA ADS (if token)
    ads = _check_ads([target, "exoplanet", "transit"])
    checked.append("nasa_ads")
    details["nasa_ads"] = ads

    # Build references (drop empty titles)
    for p in confirmed:
        references.append({
            "source": "nasa_exoplanet_archive",
            "name": p.get("pl_name", "unknown"),
            "period": p.get("pl_orbper"),
            "disposition": "confirmed",
        })
    for p in toi_matches:
        references.append({
            "source": "toi_catalog",
            "name": p.get("toi_id", p.get("tic_id", "unknown")),
            "period": p.get("pl_orbper"),
            "disposition": p.get("disposition", "unknown"),
        })
    for p in period_matches:
        references.append({
            "source": "nasa_exoplanet_archive",
            "name": p.get("pl_name", "unknown"),
            "period": p.get("pl_orbper"),
            "disposition": "period_match",
        })
    for p in arxiv.get("papers", []):
        references.append({
            "source": "arxiv",
            "name": p["title"],
            "url": p.get("url", ""),
        })
    for p in ads.get("papers", []):
        references.append({
            "source": "nasa_ads",
            "name": p["title"],
            "url": p.get("url", ""),
        })

    # Determine status
    if confirmed or toi_matches:
        exact_match = False
        if period_days:
            for p in confirmed + toi_matches + period_matches:
                p_period = p.get("pl_orbper")
                if p_period:
                    try:
                        pp = float(p_period)
                        if pp > 0 and abs(
                            pp - float(period_days)
                        ) / float(period_days) < 0.05:
                            exact_match = True
                            break
                    except (TypeError, ValueError):
                        pass
        status = "KNOWN" if exact_match else "POSSIBLY KNOWN"
    elif arxiv.get("papers") or ads.get("papers"):
        status = "POSSIBLY KNOWN"
    else:
        status = "NO MATCH FOUND"

    return {
        "track": "A",
        "status": status,
        "checked": checked,
        "references": references,
        "details": details,
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }


def check_novelty_track_b(body: str,
                          dataset: str | None = None,
                          start_date: str | None = None,
                          end_date: str | None = None) -> dict:
    """Run novelty checks for Track B (solar-system bodies).

    Queries by body name + dataset — never free-text alone.
    """
    checked: list[str] = []
    references: list[dict] = []
    details: dict = {}

    # Build query terms from body + dataset (not free text)
    query_terms = [body]
    if dataset:
        query_terms.append(dataset)

    # 1. arXiv (body + dataset)
    arxiv = _check_arxiv(query_terms)
    checked.append("arxiv")
    details["arxiv"] = arxiv

    # 2. NASA ADS (body + dataset, if token)
    ads = _check_ads(query_terms)
    checked.append("nasa_ads")
    details["nasa_ads"] = ads

    # Build references (drop empty titles)
    for p in arxiv.get("papers", []):
        references.append({
            "source": "arxiv",
            "name": p["title"],
            "url": p.get("url", ""),
        })
    for p in ads.get("papers", []):
        references.append({
            "source": "nasa_ads",
            "name": p["title"],
            "url": p.get("url", ""),
        })

    # Determine status
    if arxiv.get("papers") or ads.get("papers"):
        status = "POSSIBLY KNOWN"
    else:
        status = "NO MATCH FOUND"

    return {
        "track": "B",
        "status": status,
        "checked": checked,
        "references": references,
        "details": details,
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }


def check_novelty(target: str,
                  track: str = "A",
                  **kwargs) -> dict:
    """Dispatch to the appropriate track novelty checker.

    track: "A" (stars/exoplanets) or "B" (solar-system bodies).
    """
    if track == "B":
        return check_novelty_track_b(
            body=target,
            dataset=kwargs.get("dataset"),
            start_date=kwargs.get("start_date"),
            end_date=kwargs.get("end_date"),
        )
    return check_novelty_track_a(
        target=target,
        period_days=kwargs.get("period_days"),
        depth_ppm=kwargs.get("depth_ppm"),
        stellar_info=kwargs.get("stellar_info"),
        analysis=kwargs.get("analysis"),
    )
