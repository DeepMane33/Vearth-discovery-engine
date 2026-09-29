"""Methodology Extractor - Extracts structured research methodologies from papers."""
import json
import logging
from pathlib import Path
from collections import Counter
from engine import llm
from config import DATA_DIR

PAPERS_DIR = DATA_DIR / "papers"
METHODOLOGY_DB = PAPERS_DIR / "methodologies.json"

def _load_db():
    if METHODOLOGY_DB.exists():
        with open(METHODOLOGY_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"methodologies": []}

def _save_db(db):
    METHODOLOGY_DB.parent.mkdir(parents=True, exist_ok=True)
    with open(METHODOLOGY_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

EXTRACTION_SYSTEM = """Extract structured methodology from this astronomy paper. Return ONLY valid JSON:
{
  "research_type": "atmospheric|population|discovery|formation|habitat|other",
  "question_type": "characterization|comparison|discovery|simulation|review",
  "methods": ["list of methods used"],
  "data_sources": ["telescopes or datasets"],
  "key_findings": ["main conclusions"],
  "novelty_claim": "what is new",
  "statistical_methods": ["stats techniques"],
  "tools_software": ["tools used"],
  "paper_structure": "methodology section structure",
  "limitations": ["stated limitations"],
  "future_work": ["future directions"]
}"""

FALLBACK_PATTERNS = {
    "atmospheric": {"question_type": "characterization", "methods": ["spectral_analysis", "atmospheric_modeling", "transmission_spectroscopy"]},
    "population": {"question_type": "comparison", "methods": ["statistical_analysis", "bayesian_inference", "population_synthesis"]},
    "discovery": {"question_type": "discovery", "methods": ["transit_detection", "radial_velocity", "direct_imaging"]},
    "formation": {"question_type": "simulation", "methods": ["n_body_simulation", "hydrodynamic_modeling", "disk_evolution"]},
    "habitat": {"question_type": "comparison", "methods": ["habitability_scoring", "climate_modeling", "biosignature_detection"]},
}

def extract_methodology(paper, use_llm=True):
    paper_id = paper.get("id", paper.get("arxiv_id", "unknown"))
    title = paper.get("title", "")
    year = paper.get("year", 0)
    citations = paper.get("citations", 0)
    abstract = paper.get("abstract", "")
    methodology = {"paper_id": paper_id, "title": title, "year": year, "citations": citations, "research_type": "other", "methods": [], "data_sources": [], "key_findings": [], "novelty_claim": "", "extraction_source": "llm"}
    if use_llm:
        try:
            result = llm.ask_llm(EXTRACTION_SYSTEM, f"Title: {title}\n\nAbstract: {abstract}", temperature=0.1, max_tokens=800)
            response = result["text"]
            extracted = json.loads(response)
            methodology.update(extracted)
            methodology["extraction_source"] = "llm"
            return methodology
        except Exception as e:
            logging.warning(f"LLM extraction failed for {paper_id}: {e}")
    text = (title + " " + abstract).lower()
    for rtype, patterns in FALLBACK_PATTERNS.items():
        if rtype in text or patterns["methods"][0].replace("_", " ") in text:
            methodology["research_type"] = rtype
            methodology["question_type"] = patterns["question_type"]
            methodology["methods"] = patterns["methods"]
            methodology["extraction_source"] = "fallback"
            break
    return methodology

def extract_all_methodologies(papers, use_llm=True):
    db = _load_db()
    existing_ids = {m["paper_id"] for m in db["methodologies"]}
    new_methodologies = []
    for i, paper in enumerate(papers):
        pid = paper.get("id", paper.get("arxiv_id", f"paper_{i}"))
        if pid in existing_ids:
            continue
        meth = extract_methodology(paper, use_llm=use_llm)
        new_methodologies.append(meth)
        db["methodologies"].append(meth)
        existing_ids.add(pid)
        if (i + 1) % 10 == 0:
            _save_db(db)
    _save_db(db)
    return new_methodologies

def search_methodologies(query_type, k=5):
    db = _load_db()
    scored = []
    for m in db["methodologies"]:
        score = 0
        if m.get("research_type") == query_type:
            score += 3
        if query_type in str(m.get("methods", [])):
            score += 2
        if query_type in m.get("question_type", ""):
            score += 1
        if score > 0:
            scored.append((score, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:k]]

def get_methodology_stats():
    db = _load_db()
    methodologies = db["methodologies"]
    if not methodologies:
        return {"total": 0, "by_type": {}, "top_methods": []}
    type_counts = Counter(m.get("research_type", "other") for m in methodologies)
    method_counts = Counter(m for meth in methodologies for m in meth.get("methods", []))
    return {"total": len(methodologies), "by_type": dict(type_counts), "top_methods": method_counts.most_common(10)}
