"""Paper Ingestion Pipeline."""
import json
import math
import time
import hashlib
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import requests
import numpy as np

from config import DATA_DIR, NASA_ADS_TOKEN

logger = logging.getLogger(__name__)

PAPERS_DIR = DATA_DIR / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_INDEX = PAPERS_DIR / "paper_index.json"

NASA_ADS_API = "https://api.adsabs.harvard.edu/v1"
ARXIV_API = "http://export.arxiv.org/api/query"


def _load_index() -> dict:
    if PAPER_INDEX.exists():
        return json.loads(PAPER_INDEX.read_text(encoding="utf-8"))
    return {"papers": {}, "stats": {}}


def _save_index(idx: dict) -> None:
    PAPER_INDEX.write_text(json.dumps(idx, indent=2, default=str), encoding="utf-8")


def _embed_text(text: str) -> np.ndarray:
    vec = np.zeros(2048, dtype=np.float32)
    for i in range(2048):
        h = hashlib.md5(f"{text}_{i}".encode()).hexdigest()
        vec[i] = int(h[:8], 16) / 0xFFFFFFFF
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _paper_id(paper: dict) -> str:
    if paper.get("ads_id"):
        return f"ads_{paper['ads_id']}"
    if paper.get("arxiv_id"):
        return f"arxiv_{paper['arxiv_id']}"
    h = hashlib.md5(paper.get("title", "").encode()).hexdigest()[:12]
    return f"custom_{h}"


def search_ads(
    query: str,
    rows: int = 50,
    sort: str = "citation_count desc",
    year_range: Optional[str] = None,
    token: str = "",
) -> List[dict]:
    if not token:
        return search_arxiv(query, max_results=rows)
    url = f"{NASA_ADS_API}/search/query"
    params = {"q": query, "rows": rows, "sort": sort, "fl": "title,abstract,author,year,bibcode,doctype,citation_count,pub,keyword"}
    if year_range:
        params["q"] += f" year:{year_range}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
    except Exception as e:
        logger.warning("NASA ADS query failed: %s", e)
        return search_arxiv(query, max_results=rows)
    papers = []
    for doc in docs:
        papers.append({
            "source": "nasa_ads",
            "ads_id": doc.get("bibcode", ""),
            "title": doc.get("title", [""])[0] if doc.get("title") else "",
            "abstract": doc.get("abstract", ""),
            "authors": doc.get("author", []),
            "year": doc.get("year"),
            "journal": doc.get("pub", ""),
            "citations": doc.get("citation_count", 0),
            "doctype": doc.get("doctype", ""),
            "keywords": doc.get("keyword", []),
            "read_count": 0,
        })
    return papers


def search_arxiv(query: str, max_results: int = 50) -> List[dict]:
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results, "sortBy": "relevance"}
    try:
        r = requests.get(ARXIV_API, params=params, timeout=30)
        r.raise_for_status()
    except Exception as e:
        logger.warning("arXiv query failed: %s", e)
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    papers = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id_raw = entry.find("atom:id", ns)
        arxiv_id = arxiv_id_raw.text.split("/abs/")[-1].split("v")[0] if arxiv_id_raw is not None else ""
        title_el = entry.find("atom:title", ns)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
        abstract_el = entry.find("atom:summary", ns)
        abstract = abstract_el.text.strip().replace("\n", " ") if abstract_el is not None else ""
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns) if a.find("atom:name", ns) is not None]
        pub_el = entry.find("atom:published", ns)
        year = int(pub_el.text[:4]) if pub_el is not None and pub_el.text else None
        cat_el = entry.find("atom:primary_category", ns)
        journal = cat_el.attrib.get("term", "") if cat_el is not None else ""
        categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns)]
        papers.append({
            "source": "arxiv",
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "journal": journal,
            "citations": 0,
            "categories": categories,
        })
    return papers


TOPIC_QUERIES = {
    "exoplanet_atmospheres": [
        "exoplanet atmosphere characterization JWST transmission spectroscopy",
        "exoplanet atmospheric composition thermal emission clouds",
        "hot Jupiter atmosphere metallicity carbon-to-oxygen ratio",
        "exoplanet atmosphere photochemistry haze JWST",
    ],
    "exoplanet_populations": [
        "exoplanet population statistics radius distribution",
        "planetary radius gap super-Earth sub-Neptune transition",
        "exoplanet occurrence rate Kepler survey",
        "exoplanet mass-radius relation bulk density",
    ],
    "habitability": [
        "habitable zone exoplanet liquid water stellar flux",
        "biosignature gas detection exoplanet atmosphere",
        "tidal locking habitability M dwarf planets",
        "Earth analog exoplanet climate model habitability",
    ],
    "stellar_astrophysics": [
        "stellar abundance pattern Milky Way chemical evolution",
        "M dwarf activity flare habitability impact",
        "stellar metallicity planet formation correlation",
        "galactic chemical evolution alpha elements iron ratio",
    ],
    "planetary_formation": [
        "protoplanetary disk planet formation migration",
        "core accretion gas giant formation timescale",
        "planetary migration resonant chain multi-planet",
        "dust coagulation planetesimal formation disk",
    ],
    "solar_system": [
        "comparative planetology terrestrial planet evolution",
        "water delivery solar system giant impact",
        "Mars atmosphere loss solar wind interaction",
        "Venus greenhouse runaway climate comparison Earth",
    ],
}


def download_topic(
    topic_name: str,
    queries: Optional[List[str]] = None,
    papers_per_query: int = 30,
    token: str = "",
) -> List[dict]:
    queries = queries or TOPIC_QUERIES.get(topic_name, [])
    all_papers = []
    seen_ids = set()
    for query in queries:
        papers = search_ads(query, rows=papers_per_query, token=token)
        for p in papers:
            pid = _paper_id(p)
            if pid not in seen_ids:
                seen_ids.add(pid)
                p["topic"] = topic_name
                all_papers.append(p)
        time.sleep(3)
    return all_papers


def download_all_topics(papers_per_query: int = 30, token: str = "") -> Dict[str, int]:
    idx = _load_index()
    counts = {}
    for topic_name, queries in TOPIC_QUERIES.items():
        existing = [p for p in idx["papers"].values() if p.get("topic") == topic_name]
        if len(existing) >= papers_per_query * len(queries) * 0.5:
            counts[topic_name] = len(existing)
            continue
        papers = download_topic(topic_name, queries, papers_per_query, token)
        for p in papers:
            pid = _paper_id(p)
            if pid not in idx["papers"]:
                p["embedding"] = _embed_text(p.get("title", "") + " " + p.get("abstract", "")).tolist()
                idx["papers"][pid] = p
        _save_index(idx)
        counts[topic_name] = len([p for p in idx["papers"].values() if p.get("topic") == topic_name])
    idx["stats"]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save_index(idx)
    return counts


def search_papers(query: str, k: int = 10) -> List[dict]:
    idx = _load_index()
    if not idx["papers"]:
        return []
    q_emb = _embed_text(query)
    results = []
    for pid, paper in idx["papers"].items():
        emb = paper.get("embedding")
        if emb is None:
            continue
        score = _cosine(q_emb, np.array(emb, dtype=np.float32))
        results.append({
            "id": pid,
            "score": round(score, 4),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "authors": paper.get("authors", []),
            "year": paper.get("year"),
            "citations": paper.get("citations", 0),
            "journal": paper.get("journal", ""),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


def get_paper_stats() -> dict:
    idx = _load_index()
    papers = idx.get("papers", {})
    by_topic: Dict[str, int] = {}
    by_year: Dict[int, int] = {}
    by_source: Dict[str, int] = {}
    total_citations = 0
    for p in papers.values():
        t = p.get("topic", "unknown")
        by_topic[t] = by_topic.get(t, 0) + 1
        y = p.get("year")
        if y:
            by_year[y] = by_year.get(y, 0) + 1
        s = p.get("source", "unknown")
        by_source[s] = by_source.get(s, 0) + 1
        total_citations += p.get("citations", 0)
    return {
        "total_papers": len(papers),
        "by_topic": by_topic,
        "by_year": by_year,
        "by_source": by_source,
        "total_citations": total_citations,
        "last_updated": idx.get("stats", {}).get("last_updated", "never"),
    }
