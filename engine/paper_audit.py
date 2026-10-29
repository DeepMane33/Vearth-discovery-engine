"""Paper Audit Module.

Verifies that every number in a generated research paper is traceable to
the results JSON produced by the VEARTH pipeline. Also verifies that every
citation in the paper corresponds to a paper actually retrieved by the
engine (arXiv/NASA ADS lookup), dropping unverifiable citations.

This enforces the core rule: all numbers come from code, never from the LLM.
"""

from __future__ import annotations

import re
from typing import Any

# Regex for numbers: integers, decimals, scientific notation, negative
_NUMBER_RE = re.compile(
    r"(?<![\w.\-])"
    r"[-+]?"
    r"(?:\d+\.?\d*|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
    r"(?![\w.\-])"
)

# Common non-measurement tokens that look like numbers but are safe to skip
_SKIP_TOKENS = {
    "2026",           # year
    "v3.0",           # version
    "1.0",            # sometimes used as baseline
}


def extract_numbers(text: str) -> list[dict]:
    """Find all numeric tokens in text with their positions.

    Returns list of {value, start, end, raw}.
    """
    numbers = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0)
        # Skip pure version-like tokens and years
        if raw in _SKIP_TOKENS:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        numbers.append({
            "value": value,
            "raw": raw,
            "start": m.start(),
            "end": m.end(),
        })
    return numbers


def collect_json_numbers(obj: Any, prefix: str = "") -> set[float]:
    """Recursively collect all float/int values from a JSON-like object.

    Returns a set of float values found anywhere in the object.
    """
    found: set[float] = set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            found |= collect_json_numbers(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found |= collect_json_numbers(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool):
        # booleans are not numbers
        pass
    elif isinstance(obj, (int, float)):
        found.add(float(obj))

    return found


def _round_match(value: float, target: float, rel_tol: float = 1e-6,
                 abs_tol: float = 1e-9) -> bool:
    """Check if value matches target within tolerance.

    Also checks common rounded forms (1 decimal, 2 decimals, integer).
    """
    if abs(value - target) <= abs_tol:
        return True
    if abs(target) > 0 and abs(value - target) / abs(target) <= rel_tol:
        return True
    # Check rounded versions
    for nd in (0, 1, 2, 3, 4):
        if abs(value - round(target, nd)) <= abs_tol * max(1, abs(round(target, nd))):
            return True
    return False


def audit_paper_numbers(paper_text: str, results_json: dict,
                        max_unverified: int = 0) -> dict:
    """Audit that every number in paper_text appears in results_json.

    Args:
        paper_text: the generated paper Markdown.
        results_json: the full results dict from orchestrator.run().
        max_unverified: max allowed unverified numbers (0 = fail on any).

    Returns:
        dict with keys:
            passed: bool
            total_numbers_found: int
            verified_numbers: int
            unverified_numbers: list of {raw, value, context}
            unverified_count: int
            json_number_count: int
            message: str
    """
    paper_numbers = extract_numbers(paper_text)
    json_numbers = collect_json_numbers(results_json)

    verified = []
    unverified = []

    for num in paper_numbers:
        val = num["value"]
        # Check if this value appears anywhere in the JSON
        matched = any(_round_match(val, jv) for jv in json_numbers)
        if matched:
            verified.append(num)
        else:
            # Get surrounding context for reporting
            start = max(0, num["start"] - 60)
            end = min(len(paper_text), num["end"] + 60)
            context = paper_text[start:end].replace("\n", " ")
            unverified.append({
                "raw": num["raw"],
                "value": val,
                "context": context,
            })

    passed = len(unverified) <= max_unverified

    return {
        "passed": passed,
        "total_numbers_found": len(paper_numbers),
        "verified_numbers": len(verified),
        "unverified_numbers": unverified,
        "unverified_count": len(unverified),
        "json_number_count": len(json_numbers),
        "message": (
            f"{len(verified)}/{len(paper_numbers)} numbers traceable to results JSON. "
            f"{len(unverified)} unverified."
        ),
    }


def extract_citations(text: str) -> list[str]:
    """Extract citation tokens like [ref-2012-bls] or [arxiv-0] from text."""
    return re.findall(r"\[([^\]]+)\]", text)


def verify_citations(paper_text: str, retrieved_papers: list[dict]) -> dict:
    """Verify that every citation in the paper refers to a retrieved paper.

    Args:
        paper_text: the generated paper Markdown.
        retrieved_papers: list of paper dicts actually retrieved by the engine
            (each should have an 'id', 'arxiv_id', 'ads_id', or 'title').

    Returns:
        dict with keys:
            passed: bool
            total_citations: int
            verified_citations: int
            dropped_citations: list of citation strings not found
            verified_citation_ids: list of verified citation strings
    """
    citations = extract_citations(paper_text)

    # Build lookup set of paper identifiers
    paper_ids: set[str] = set()
    for p in retrieved_papers:
        pid = p.get("id") or p.get("arxiv_id") or p.get("ads_id") or ""
        if pid:
            paper_ids.add(pid)
        # Also add arxiv id without prefix
        arxiv = p.get("arxiv_id", "")
        if arxiv:
            paper_ids.add(f"arxiv_{arxiv}")
        ads = p.get("ads_id", "")
        if ads:
            paper_ids.add(f"ads_{ads}")
        # Add title words as fuzzy fallback
        title = p.get("title", "")
        if title:
            paper_ids.add(title.lower().strip())

    verified = []
    dropped = []

    for cite in citations:
        # Normalize citation for comparison
        cite_norm = cite.strip().lower()
        found = cite_norm in paper_ids or any(
            cite_norm in pid for pid in paper_ids
        )
        if found:
            verified.append(cite)
        else:
            dropped.append(cite)

    return {
        "passed": len(dropped) == 0,
        "total_citations": len(citations),
        "verified_citations": len(verified),
        "dropped_citations": dropped,
        "verified_citation_ids": verified,
    }


def strip_unverified_citations(paper_text: str,
                               retrieved_papers: list[dict]) -> str:
    """Remove citation tokens from paper text that can't be verified.

    Returns cleaned paper text.
    """
    citations = extract_citations(paper_text)

    paper_ids: set[str] = set()
    for p in retrieved_papers:
        pid = p.get("id") or p.get("arxiv_id") or p.get("ads_id") or ""
        if pid:
            paper_ids.add(pid)
        arxiv = p.get("arxiv_id", "")
        if arxiv:
            paper_ids.add(f"arxiv_{arxiv}")
        ads = p.get("ads_id", "")
        if ads:
            paper_ids.add(f"ads_{ads}")
        title = p.get("title", "")
        if title:
            paper_ids.add(title.lower().strip())

    cleaned = paper_text
    for cite in citations:
        cite_norm = cite.strip().lower()
        found = cite_norm in paper_ids or any(
            cite_norm in pid for pid in paper_ids
        )
        if not found:
            # Remove [cite] token
            cleaned = cleaned.replace(f"[{cite}]", "")

    return cleaned


def generate_audit_header(results_json: dict) -> str:
    """Generate the mandatory header line for AI-assisted drafts."""
    target = results_json.get("target", "unknown")
    source = results_json.get("ingestion", {}).get("source", "unknown")
    veto = results_json.get("validation", {}).get("veto", True)
    verdict = "vetoed" if veto else "candidate"

    return (
        f"# Draft generated with AI assistance. Requires human verification "
        f"before any submission.\n\n"
        f"**Target:** {target}  |  **Data source:** {source}  |  "
        f"**Pipeline verdict:** {verdict}\n\n"
    )
