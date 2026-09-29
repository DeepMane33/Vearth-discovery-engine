"""
VEARTH Research Engine CLI

Usage:
    python cli.py test                - Quick system test
    python cli.py ingest              - Download papers for all topics
    python cli.py ingest atmospheres  - Download for one topic
    python cli.py stats               - Show paper database stats
    python cli.py search "query"      - Search paper database
    python cli.py research "question" - Run research pipeline
"""

import sys
import os
import json
import logging

sys.stdout.reconfigure(errors="replace")
sys.stderr.reconfigure(errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.paper_ingestion import search_arxiv, _embed_text, download_all_topics, download_topic, get_paper_stats, search_papers, _load_index, _save_index, _paper_id, _embed_text
from engine.research_knowledge import get_knowledge
from engine.gap_detector import get_detector
from engine.methodology_extractor import extract_methodology
from engine.research_orchestrator import get_orchestrator
from config import NASA_ADS_TOKEN

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("cli")

TOPIC_ALIASES = {
    "atm": "exoplanet_atmospheres",
    "atmospheres": "exoplanet_atmospheres",
    "hab": "habitability",
    "bio": "biosignatures",
    "orb": "orbital_dynamics",
    "star": "stellar_activity",
    "comp": "computational_methods",
}


def cmd_test():
    """Run quick system checks."""
    tests = [
        ("arXiv search", lambda: search_arxiv("exoplanet", max_results=1)),
        ("Embedding", lambda: _embed_text("test")),
        ("Knowledge store", lambda: get_knowledge()),
        ("Gap detector", lambda: get_detector()),
        ("Methodology extractor", lambda: extract_methodology({"title": "Test", "abstract": "Test abstract.", "year": 2024}, use_llm=False)),
        ("Research orchestrator", lambda: get_orchestrator()),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK]  {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")


def cmd_ingest(args):
    """Download papers for specified topic(s) and save to index."""
    if args:
        topic = TOPIC_ALIASES.get(args[0], args[0])
        log.info(f"Downloading papers for: {topic}")
        papers = download_topic(topic, token=NASA_ADS_TOKEN)
        # Save to index
        idx = _load_index()
        count = 0
        for p in papers:
            pid = _paper_id(p)
            if pid not in idx["papers"]:
                p["embedding"] = _embed_text(p.get("title", "") + " " + p.get("abstract", "")).tolist()
                p["topic"] = topic
                idx["papers"][pid] = p
                count += 1
        _save_index(idx)
        log.info(f"Saved {count} new papers to index (total: {len(idx['papers'])})")
    else:
        log.info("Downloading papers for all topics...")
        results = download_all_topics(token=NASA_ADS_TOKEN)
        for topic, count in results.items():
            log.info(f"  {topic}: {count} papers")


def cmd_stats():
    """Show paper database statistics."""
    stats = get_paper_stats()
    print(json.dumps(stats, indent=2))


def cmd_search(args):
    """Search paper database."""
    if not args:
        print('Usage: cli.py search "query"')
        return
    results = search_papers(args[0], k=5)
    for i, paper in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Title: {paper.get('title', 'N/A')}")
        print(f"Score: {paper.get('score', 0):.4f}")
        print(f"Abstract: {paper.get('abstract', 'N/A')[:200]}...")


def cmd_research(args):
    """Run full research pipeline."""
    if not args:
        print('Usage: cli.py research "question"')
        return
    question = args[0]
    orchestrator = get_orchestrator()
    result = orchestrator.run_research(question)
    print("\n=== Research Plan ===")
    print(json.dumps(result.get("plan", {}), indent=2))
    print("\n=== Prior Work ===")
    papers = result.get("existing_papers", [])
    if papers:
        for i, p in enumerate(papers[:5], 1):
            print(f"  {i}. {p.get('title', 'N/A')[:80]} (score: {p.get('score', 0):.3f})")
    else:
        print("  No matching papers found.")
    print("\n=== Knowledge Gaps ===")
    gaps = result.get("gaps", [])
    for g in gaps[:5]:
        print(f"  - {g}")
    print("\n=== Novelty Assessment ===")
    print(json.dumps(result.get("novelty", {}), indent=2))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd, args = sys.argv[1], sys.argv[2:]
    commands = {
        "test": cmd_test,
        "ingest": cmd_ingest,
        "stats": cmd_stats,
        "search": cmd_search,
        "research": cmd_research,
    }
    if cmd in commands:
        fn = commands[cmd]
        fn(args) if cmd in ("ingest", "search", "research") else fn()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
