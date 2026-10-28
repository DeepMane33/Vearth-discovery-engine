"""Builder: methodology_extractor.py"""

def get_content():
    return '''"""Methodology Extractor - Extracts research patterns from papers using LLM.

For each paper, extracts structured methodology data that teaches
VEARTH how researchers approach different types of questions.
"""
import json
import logging
from pathlib import Path

from engine import llm
from config import DATA_DIR

log = logging.getLogger(__name__)

PAPERS_DIR = DATA_DIR / "papers"
METHODOLOGY_DB = PAPERS_DIR / "methodologies.json"


def _load_db():
    if METHODOLOGY_DB.exists():
        return json.loads(METHODOLOGY_DB.read_text())
    return {"methodologies": []}


def _save_db(db):
    METHODOLOGY_DB.write_text(json.dumps(db, indent=2, default=str))


EXTRACTION_SYSTEM = (
    "You are an expert research analyst. Given a paper title and abstract, "
    "extract structured methodology information. Respond ONLY as JSON with "
    "these keys:\\n"
    "- research_type: (discovery|characterization|comparison|review|methodology|simulation)\\n"
    "- question_type: specific research question category\\n"
    "- methods: list of analysis methods used\\n"
    "- data_sources: list of telescopes/instruments/catalogs used\\n"
    "- key_findings: list of main results (2-3 sentences each)\\n"
    "- novelty_claim: what is new about this work\\n"
    "- statistical_methods: list of statistical techniques\\n"
    "- tools_software: list of tools/packages mentioned\\n"
    "- paper_structure: typical sections for this type of work\\n"
    "- limitations: what the authors say they could not do\\n"
    "- future_work: what future work they suggest\\n"
)

FALLBACK_PATTERNS = {
    "atmospheric": {
        "research_type": "characterization",
        "methods": ["spectral_analysis", "atmospheric_retrieval"],
        "data_sources": ["JWST", "HST"],
        "paper_structure": ["Introduction", "Observations", "Data Reduction",
                          "Analysis", "Results", "Discussion", "Conclusion"],
    },
    "population": {
        "research_type": "comparison",
        "methods": ["statistical_analysis", "catalog_mining"],
        "data_sources": ["NASA Exoplanet Archive", "Kepler", "TESS"],
        "paper_structure": ["Introduction", "Sample Selection", "Data",
                          "Analysis", "Results", "Discussion", "Conclusion"],
    },
    "discovery": {
        "research_type": "discovery",
        "methods": ["transit_detection", "radial_velocity", "direct_imaging"],
        "data_sources": ["TESS", "Kepler", "ground-based"],
        "paper_structure": ["Introduction", "Discovery", "Follow-up",
                          "Validation", "Discussion", "Conclusion"],
    },
    "formation": {
        "research_type": "simulation",
        "methods": ["n_body_simulation", "hydrodynamic_simulation"],
        "data_sources": ["theoretical_models"],
        "paper_structure": ["Introduction", "Methods", "Simulations",
                          "Results", "Discussion", "Conclusion"],
    },
    "habitability": {
        "research_type": "comparison",
        "methods": ["habitability_scoring", "comparative_analysis"],
        "data_sources": ["NASA Exoplanet Archive", "stellar_catalogs"],
        "paper_structure": ["Introduction", "Target Selection", "Methods",
                          "Habitability Analysis", "Results", "Discussion"],
    },
}


def extract_methodology(paper, use_llm=True):
    """Extract methodology from a single paper.

    Args:
        paper: dict with title, abstract, authors, year, etc.
        use_llm: whether to use LLM for extraction

    Returns:
        dict with structured methodology data
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    if not abstract:
        return None

    if use_llm and llm.llm_available():
        prompt = (
            f"Title: {title}\\n\\n"
            f"Abstract: {abstract}\\n\\n"
            f"Extract the research methodology as JSON."
        )
        result = llm.ask_llm(EXTRACTION_SYSTEM, prompt,
                            temperature=0.1, max_tokens=800)
        parsed = llm.extract_json(result["text"])

        if parsed and result["source"] == "llm":
            parsed["paper_id"] = paper.get("ads_id") or paper.get("arxiv_id", "")
            parsed["title"] = title
            parsed["year"] = paper.get("year")
            parsed["citations"] = paper.get("citations", 0)
            parsed["extraction_source"] = "llm"
            return parsed

    # Fallback: keyword-based extraction
    abstract_lower = abstract.lower()
    best_match = None
    best_score = 0

    for pattern_name, pattern in FALLBACK_PATTERNS.items():
        score = abstract_lower.count(pattern_name)
        if score > best_score:
            best_score = score
            best_match = pattern_name

    if best_match and best_score > 0:
        result = dict(FALLBACK_PATTERNS[best_match])
        result["research_type"] = best_match
        result["paper_id"] = paper.get("ads_id") or paper.get("arxiv_id", "")
        result["title"] = title
        result["year"] = paper.get("year")
        result["citations"] = paper.get("citations", 0)
        result["key_findings"] = [abstract[:200]]
        result["novelty_claim"] = ""
        result["extraction_source"] = "keyword_fallback"
        return result

    return {
        "research_type": "unknown",
        "paper_id": paper.get("ads_id") or paper.get("arxiv_id", ""),
        "title": title,
        "year": paper.get("year"),
        "citations": paper.get("citations", 0),
        "methods": [],
        "data_sources": [],
        "key_findings": [abstract[:200]],
        "novelty_claim": "",
        "extraction_source": "minimal_fallback",
    }


def extract_all_methodologies(papers, use_llm=True):
    """Extract methodologies for a batch of papers.

    Args:
        papers: list of paper dicts
        use_llm: whether to use LLM

    Returns:
        list of methodology dicts
    """
    db = _load_db()
    existing_ids = {m["paper_id"] for m in db["methodologies"]}

    new_count = 0
    for i, paper in enumerate(papers):
        pid = paper.get("ads_id") or paper.get("arxiv_id", "")
        if not pid or pid in existing_ids:
            continue

        meth = extract_methodology(paper, use_llm=use_llm)
        if meth:
            db["methodologies"].append(meth)
            existing_ids.add(pid)
            new_count += 1

            if new_count % 10 == 0:
                _save_db(db)
                log.info("Extracted %d methodologies so far...", new_count)

    _save_db(db)
    log.info("Extracted %d new methodologies (total: %d)",
             new_count, len(db["methodologies"]))
    return db["methodologies"]


def search_methodologies(query_type, k=5):
    """Find methodologies that match a research question type.

    Args:
        query_type: research question type string
        k: max results

    Returns:
        list of matching methodology dicts
    """
    db = _load_db()
    scored = []
    for meth in db["methodologies"]:
        score = 0
        if meth.get("research_type") == query_type:
            score += 3
        if query_type in str(meth.get("methods", [])):
            score += 2
        if query_type in str(meth.get("data_sources", [])):
            score += 1
        if query_type in meth.get("title", "").lower():
            score += 1
        if score > 0:
            scored.append((score, meth))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]


def get_methodology_stats():
    """Return stats about extracted methodologies."""
    db = _load_db()
    meths = db["methodologies"]

    types = {}
    methods_count = {}
    for m in meths:
        rt = m.get("research_type", "unknown")
        types[rt] = types.get(rt, 0) + 1
        for method in m.get("methods", []):
            methods_count[method] = methods_count.get(method, 0) + 1

    return {
        "total": len(meths),
        "research_types": types,
        "top_methods": dict(sorted(methods_count.items(),
                                   key=lambda x: x[1], reverse=True)[:20]),
    }
'''
