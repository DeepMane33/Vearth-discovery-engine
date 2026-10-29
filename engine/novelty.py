"""Novelty Checker.

Verifies that a detected transit signal is not already a known planet,
TESS Object of Interest (TOI), or published result before it can be
described as a new discovery.

Checks public sources:
  - NASA Exoplanet Archive (confirmed planets near the target)
  - TOI catalog (TESS Objects of Interest)
  - SIMBAD (target identification)
  - arXiv (published literature mentioning the target)

Returns one of: KNOWN, POSSIBLY KNOWN, NO MATCH FOUND.

"No match found" is never worded as "confirmed new discovery."
"""

from __future__ import annotations

from datetime import datetime, timezone

from engine import datasources


def _resolve_tic(target: str) -> str | None:
    """Resolve a target name to a TIC ID."""
    try:
        tic = datasources.get_tic_params(target)
        if tic and tic.get("ticid"):
            return str(tic["ticid"])
    except Exception:
        pass
    return None


def _check_exoplanet_archive(target: str,
                             period_days: float | None = None) -> dict:
    """Check confirmed planets near the target and by period."""
    result: dict = {"matches": [], "source": "nasa_exoplanet_archive"}
    try:
        coords = datasources.query_simbad_coordinates(target)
        ra = coords.get("ra")
        dec = coords.get("dec")
        if ra and dec:
            planets = datasources.get_known_planets_near(
                float(ra), float(dec), radius_deg=0.15
            )
            result["matches"] = planets[:10]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    if period_days:
        try:
            by_period = datasources.get_known_planet_by_period(
                float(period_days), tolerance=0.01
            )
            result["period_matches"] = by_period[:5]
        except Exception as e:
            result["period_error"] = f"{type(e).__name__}: {e}"

    return result


def _check_toi_catalog(target: str,
                       period_days: float | None = None) -> dict:
    """Check the TOI catalog via the Exoplanet Archive TAP.

    Tries a few known table names because the Exoplanet Archive exposes
    TOI data under different names in different TAP deployments.
    """
    result: dict = {"matches": [], "source": "toi_catalog"}
    tic_id = _resolve_tic(target)
    if not tic_id:
        result["error"] = "could not resolve TIC ID"
        return result

    candidate_tables = ["toi", "toi_cat", "toi_data"]
    last_error = None
    for table in candidate_tables:
        try:
            query = (
                "SELECT tic_id, toi_id, disposition, pl_orbper, pl_rade, "
                "pl_eqt, ra, dec FROM " + table + " "
                f"WHERE tic_id = {tic_id}"
            )
            matches = datasources.query_exoplanet_archive(query)
            result["matches"] = matches[:10]
            result["table_used"] = table
            return result
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            continue

    result["error"] = last_error or "all toi table queries failed"
    return result


def _check_simbad(target: str) -> dict:
    """Check SIMBAD for target identification."""
    result: dict = {"info": {}, "source": "simbad"}
    try:
        info = datasources.query_simbad(target)
        result["info"] = info
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def _check_arxiv(target: str) -> dict:
    """Search arXiv for published papers mentioning the target."""
    result: dict = {"papers": [], "source": "arxiv"}
    try:
        papers = datasources.search_arxiv(
            f"{target} exoplanet transit", max_results=5
        )
        result["papers"] = papers[:5]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def check_novelty(target: str,
                  period_days: float | None = None,
                  depth_ppm: float | None = None,
                  stellar_info: dict | None = None,
                  analysis: dict | None = None) -> dict:
    """Run novelty checks against all public sources.

    Returns dict with keys:
        status: "KNOWN" | "POSSIBLY KNOWN" | "NO MATCH FOUND"
        checked: list of sources checked
        references: list of reference dicts
        details: per-source results
        ran_at: ISO timestamp
    """
    checked: list[str] = []
    references: list[dict] = []
    details: dict = {}

    # 1. NASA Exoplanet Archive
    archive = _check_exoplanet_archive(target, period_days)
    checked.append("nasa_exoplanet_archive")
    details["nasa_exoplanet_archive"] = archive

    confirmed_matches = archive.get("matches", [])
    period_matches = archive.get("period_matches", [])

    # 2. TOI catalog
    toi = _check_toi_catalog(target, period_days)
    checked.append("toi_catalog")
    details["toi_catalog"] = toi

    toi_matches = toi.get("matches", [])

    # 3. SIMBAD
    simbad = _check_simbad(target)
    checked.append("simbad")
    details["simbad"] = simbad

    # 4. arXiv
    arxiv = _check_arxiv(target)
    checked.append("arxiv")
    details["arxiv"] = arxiv

    arxiv_papers = arxiv.get("papers", [])

    # Build references list
    for p in confirmed_matches:
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
    for p in arxiv_papers:
        references.append({
            "source": "arxiv",
            "name": p.get("title", "unknown"),
            "url": p.get("url", ""),
        })

    # Determine status
    if confirmed_matches or toi_matches:
        exact_match = False
        if period_days:
            for p in confirmed_matches + toi_matches:
                p_period = p.get("pl_orbper")
                if p_period:
                    try:
                        pp = float(p_period)
                        if pp > 0 and abs(pp - float(period_days)) / float(
                            period_days
                        ) < 0.05:
                            exact_match = True
                            break
                    except (TypeError, ValueError):
                        pass

        status = "KNOWN" if exact_match else "POSSIBLY KNOWN"
    elif arxiv_papers:
        status = "POSSIBLY KNOWN"
    else:
        status = "NO MATCH FOUND"

    return {
        "status": status,
        "checked": checked,
        "references": references,
        "details": details,
        "ran_at": datetime.now(timezone.utc).isoformat(),
    }
