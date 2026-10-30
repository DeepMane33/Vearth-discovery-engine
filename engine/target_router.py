"""Target Router.

Classifies a user-supplied target string into one of three categories:
  - solar_system: a known planet/moon in our solar system -> Track B
  - stellar: a star or exoplanet host resolvable via SIMBAD/MAST -> Track A
  - unresolvable: cannot be identified -> stop with a helpful error

Solar-system body names are matched case-insensitively against a curated list.
Stellar targets are resolved to TIC ID / coordinates via SIMBAD (with MAST
TIC fallback for TIC/TOI identifiers that SIMBAD does not index).

All results are cached on disk under data/cache/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from config import CACHE_DIR

# Curated solar-system bodies (case-insensitive matching).
SOLAR_SYSTEM_BODIES = {
    "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus",
    "neptune", "pluto", "moon", "luna", "sol", "sun",
    "io", "europa", "ganymede", "callisto", "titan", "enceladus",
    "triton", "charon", "ceres", "eris", "haumea", "makemake",
}

_CACHE_DIR = CACHE_DIR / "router"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(target: str) -> str:
    return hashlib.md5(target.strip().lower().encode()).hexdigest()


def _load_cache(target: str) -> dict | None:
    p = _CACHE_DIR / f"{_cache_key(target)}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(target: str, result: dict) -> None:
    p = _CACHE_DIR / f"{_cache_key(target)}.json"
    try:
        p.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                     encoding="utf-8")
    except Exception:
        pass


def classify_target(target: str) -> dict:
    """Classify a target string.

    Returns dict with keys:
      category: "solar_system" | "stellar" | "unresolvable"
      target: normalized input
      tic_id: str | None (stellar only)
      ra: float | None (stellar only)
      dec: float | None (stellar only)
      error: str | None
    """
    target = target.strip()
    if not target:
        return {
            "category": "unresolvable",
            "target": target,
            "tic_id": None,
            "ra": None,
            "dec": None,
            "error": "empty target identifier",
        }

    cached = _load_cache(target)
    if cached is not None:
        return cached

    result: dict

    # 1. Solar-system body check (exact name match, case-insensitive).
    if target.lower() in SOLAR_SYSTEM_BODIES:
        result = {
            "category": "solar_system",
            "target": target,
            "tic_id": None,
            "ra": None,
            "dec": None,
            "error": None,
        }
        _save_cache(target, result)
        return result

    # 2. Stellar resolution via SIMBAD / MAST.
    stellar = _resolve_stellar(target)
    if stellar["resolved"]:
        result = {
            "category": "stellar",
            "target": target,
            "tic_id": stellar.get("tic_id"),
            "ra": stellar.get("ra"),
            "dec": stellar.get("dec"),
            "error": None,
        }
        _save_cache(target, result)
        return result

    # 3. Unresolvable.
    result = {
        "category": "unresolvable",
        "target": target,
        "tic_id": None,
        "ra": None,
        "dec": None,
        "error": (
            f"Could not resolve '{target}' as a solar-system body or a "
            "stellar/exoplanet target. Check the spelling or try a TIC/TOI "
            "identifier (e.g. 'TIC 268684168' or 'TOI 700')."
        ),
    }
    _save_cache(target, result)
    return result


def _resolve_stellar(target: str) -> dict:
    """Try to resolve a stellar/exoplanet target to TIC ID + coordinates.

    Tries SIMBAD first, then MAST TIC catalog for TIC/TOI identifiers that
    SIMBAD does not index. Returns {"resolved": bool, ...}.
    """
    # Try SIMBAD coordinates.
    try:
        from engine.datasources import query_simbad_coordinates
        coords = query_simbad_coordinates(target)
        if coords and coords.get("ra") and coords.get("dec"):
            tic_id = _resolve_tic_from_coords(
                float(coords["ra"]), float(coords["dec"])
            )
            return {
                "resolved": True,
                "tic_id": tic_id,
                "ra": float(coords["ra"]),
                "dec": float(coords["dec"]),
            }
    except Exception:
        pass

    # Try MAST TIC catalog directly (handles TIC/TOI names SIMBAD lacks).
    try:
        from astroquery.mast import Catalogs
        tic = Catalogs.query_object(target, catalog="TIC", radius=0.002)
        if tic is not None and len(tic) > 0:
            row = tic[0]
            return {
                "resolved": True,
                "tic_id": str(row.get("ID", target)),
                "ra": float(row["ra"]),
                "dec": float(row["dec"]),
            }
    except Exception:
        pass

    return {"resolved": False}


def _resolve_tic_from_coords(ra: float, dec: float) -> str | None:
    """Best-effort TIC ID lookup from coordinates via MAST."""
    try:
        from astroquery.mast import Catalogs
        tic = Catalogs.query_region(ra, dec, catalog="TIC", radius=0.001)
        if tic is not None and len(tic) > 0:
            return str(tic[0].get("ID", ""))
    except Exception:
        pass
    return None


def route_target(target: str) -> dict:
    """Public entry point: classify and return routing instructions.

    Returns the same dict as classify_target plus a 'track' field:
      track: "A" (stellar) | "B" (solar_system) | None (unresolvable)
    """
    result = classify_target(target)
    if result["category"] == "stellar":
        result["track"] = "A"
    elif result["category"] == "solar_system":
        result["track"] = "B"
    else:
        result["track"] = None
    return result
