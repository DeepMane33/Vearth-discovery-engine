"""Builder: paper_ingestion.py"""

def get_content():
    return '''"""Paper Ingestion Pipeline - Downloads and indexes research papers.

Sources:
  - NASA ADS (astrophysics data system) - 14M papers, free API
  - arXiv astrophysics - full text papers since 1991

For each paper, extracts metadata and builds a searchable knowledge base.
"""
import json
import math
import time
import hashlib
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
import numpy as np

from config import DATA_DIR

log = logging.getLogger(__name__)

PAPERS_DIR = DATA_DIR / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

PAPER_INDEX = PAPERS_DIR / "paper_index.json"

NASA_ADS_API = "https://api.adsabs.harvard.edu/v1"
ARXIV_API = "http://export.arxiv.org/api/query"


def _load_index():
    if PAPER_INDEX.exists():
        return json.loads(PAPER_INDEX.read_text())
    return {"papers": {}, "stats": {"total": 0, "last_updated": None}}


def _save_index(idx):
    PAPER_INDEX.write_text(json.dumps(idx, indent=2, default=str))


def _embed_text(text):
    tokens = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    vec = [0.0] * 2048
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 2048
        vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def _paper_id(paper):
    if paper.get("ads_id"):
        return "ads_" + paper["ads_id"]
    if paper.get("arxiv_id"):
        return "arxiv_" + paper["arxiv_id"]
    return "hash_" + hashlib.md5(paper.get("title", "").encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# NASA ADS
# ---------------------------------------------------------------------------

def search_ads(query, rows=50, sort="citation_count desc",
               year_range=None, token=""):
    if not token:
        log.warning("No NASA ADS token. Using arXiv fallback.")
        return search_arxiv(query, max_results=rows)

    if year_range:
        query = f"{query} year:{year_range}"

    headers = {"Authorization": f"Bearer {token}"}
    url = f"{NASA_ADS_API}/search/query"
    params = {
        "q": query,
        "rows": min(rows, 200),
        "fl": (
            "bibcode,title,abstract,author,pub,year,citation_count,"
            "doctype,keyword,read_count"
        ),
        "sort": sort,
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        papers = []
        for doc in data.get("response", {}).get("docs", []):
            paper = {
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
                "read_count": doc.get("read_count", 0),
            }
            papers.append(paper)
        log.info("ADS: %d papers for: %s", len(papers), query)
        return papers
    except Exception as e:
        log.warning("ADS failed: %s. arXiv fallback.", e)
        return search_arxiv(query, max_results=rows)


def get_ads_paper(bibcode, token=""):
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{NASA_ADS_API}/search/query"
    params = {
        "q": f"bibcode:{bibcode}",
        "fl": "bibcode,title,abstract,author,pub,year,citation_count,doctype,keyword,read_count",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])
        if docs:
            doc = docs[0]
            return {
                "source": "nasa_ads",
                "ads_id": doc.get("bibcode", ""),
                "title": (doc.get("title", [""])[0] if doc.get("title") else ""),
                "abstract": doc.get("abstract", ""),
                "authors": doc.get("author", []),
                "year": doc.get("year"),
                "journal": doc.get("pub", ""),
                "citations": doc.get("citation_count", 0),
                "doctype": doc.get("doctype", ""),
                "keywords": doc.get("keyword", []),
                "read_count": doc.get("read_count", 0),
            }
    except Exception as e:
        log.warning("ADS fetch failed for %s: %s", bibcode, e)
    return None


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

def search_arxiv(query, max_results=50):
    params = {
        "search_query": f"astro-ph:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    try:
        resp = requests.get(ARXIV_API, params=params, timeout=15)
        resp.raise_for_status()
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
            categories = [c.get("term", "")
                         for c in entry.findall("atom:category", ns)]
            paper = {
                "source": "arxiv",
                "arxiv_id": (id_el.text.strip().split("/abs/")[-1]
                            if id_el is not None else ""),
                "title": (title_el.text or "").strip() if title_el is not None else "",
                "abstract": (summary_el.text or "").strip() if summary_el is not None else "",
                "authors": authors,
                "year": int(published_el.text[:4]) if published_el is not None else None,
                "journal": "arXiv",
                "citations": 0,
                "categories": categories,
            }
            papers.append(paper)
        log.info("arXiv: %d papers for: %s", len(papers), query)
        return papers
    except Exception as e:
        log.warning("arXiv failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Topic-based bulk download
# ---------------------------------------------------------------------------

TOPIC_QUERIES = {
    "exoplanet_atmospheres": [
        "exoplanet atmosphere transmission spectroscopy",
        "exoplanet atmospheric composition JWST",
        "exoplanet atmospheric retrieval",
        "exoplanet biosignature atmosphere",
    ],
    "exoplanet_populations": [
        "exoplanet population statistics radius gap",
        "exoplanet mass radius relation",
        "exoplanet occurrence rate",
        "exoplanet orbital architecture",
    ],
    "habitability": [
        "habitable zone exoplanet Earth twin",
        "exoplanet habitability assessment",
        "biosignature detection exoplanet",
        "planetary habitability criteria",
    ],
    "stellar_astrophysics": [
        "stellar abundance chemical composition",
        "stellar activity cycle magnetic field",
        "M dwarf planet habitability",
        "stellar evolution age dating",
    ],
    "planetary_formation": [
        "planet formation protoplanetary disk",
        "planetary migration resonance",
        "planet formation core accretion",
        "disk instability giant planet formation",
    ],
    "solar_system": [
        "comparative planetology solar system exoplanet",
        "water delivery solar system",
        "planetary magnetic field dynamo",
        "atmospheric escape planetary evolution",
    ],
}


def download_topic(topic_name, queries=None, papers_per_query=30, token=""):
    if queries is None:
        queries = TOPIC_QUERIES.get(topic_name, [topic_name])
    all_papers = []
    seen_ids = set()
    for query in queries:
        log.info("Downloading topic=%s query=\'%s\'", topic_name, query)
        papers = search_ads(query, rows=papers_per_query, token=token)
        for p in papers:
            pid = _paper_id(p)
            if pid not in seen_ids and p.get("abstract"):
                seen_ids.add(pid)
                all_papers.append(p)
        time.sleep(0.5)
    log.info("Topic %s: %d unique papers", topic_name, len(all_papers))
    return all_papers


def download_all_topics(papers_per_query=30, token=""):
    index = _load_index()
    results = {}
    for topic_name, queries in TOPIC_QUERIES.items():
        existing = [pid for pid, p in index["papers"].items()
                    if p.get("topic") == topic_name]
        if len(existing) >= papers_per_query * len(queries) * 0.5:
            log.info("Skipping topic %s (already have %d papers)",
                     topic_name, len(existing))
            continue
        papers = download_topic(topic_name, queries, papers_per_query, token)
        for p in papers:
            pid = _paper_id(p)
            p["topic"] = topic_name
            title = p.get("title", "")
            abstract = p.get("abstract", "")
            p["embedding"] = _embed_text(title + " " + abstract)
            index["papers"][pid] = p
        results[topic_name] = len(papers)
        _save_index(index)
        log.info("Saved %d papers for topic %s", len(papers), topic_name)
    index["stats"]["total"] = len(index["papers"])
    index["stats"]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_index(index)
    return results


# ---------------------------------------------------------------------------
# Search the local paper index
# ---------------------------------------------------------------------------

def search_papers(query, k=10):
    index = _load_index()
    if not index["papers"]:
        return []
    qvec = _embed_text(query)
    scored = []
    for pid, paper in index["papers"].items():
        emb = paper.get("embedding")
        if emb:
            score = _cosine(qvec, emb)
            scored.append((score, pid, paper))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, pid, paper in scored[:k]:
        results.append({
            "id": pid,
            "score": round(score, 4),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", "")[:500],
            "authors": paper.get("authors", [])[:3],
            "year": paper.get("year"),
            "citations": paper.get("citations", 0),
            "journal": paper.get("journal", ""),
        })
    return results


def get_paper_stats():
    index = _load_index()
    papers = index["papers"]
    topics = {}
    years = {}
    sources = {}
    total_citations = 0
    for pid, p in papers.items():
        t = p.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1
        y = p.get("year")
        if y:
            years[y] = years.get(y, 0) + 1
        s = p.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
        total_citations += p.get("citations", 0)
    return {
        "total_papers": len(papers),
        "topics": topics,
        "years": dict(sorted(years.items())),
        "sources": sources,
        "total_citations": total_citations,
        "last_updated": index["stats"].get("last_updated"),
    }
'''
