# Research Gap Detector
import json, logging
from pathlib import Path

FALLBACK_KEYWORDS = [
    "remain unclear", "not yet known", "uncertain", "poorly constrained",
    "future work", "further study needed", "open question", "debate",
    "conflicting results", "not yet measured",
]
_log = logging.getLogger(__name__)

class GapDetector:
    def __init__(self):
        self.gaps: list[dict] = []
        p = Path("data/papers/research_gaps.json")
        if p.exists():
            try: self.gaps = json.loads(p.read_text(encoding="utf-8"))
            except Exception: _log.warning("Could not load existing research_gaps.json")

    def _make_gap(self, question, paper, hits, topic):
        pri = "high" if len(hits) >= 3 else "medium" if len(hits) >= 2 else "low"
        return {"question": question, "evidence": f"'{paper.get('title','')}' mentions: {', '.join(hits)}", "priority": pri, "topic": topic, "detection_method": "keyword"}

    def detect_gaps_from_papers(self, papers, topic):
        found = []
        for p in papers:
            hits = [kw for kw in FALLBACK_KEYWORDS if kw in (p.get("abstract") or "").lower()]
            if hits: found.append(self._make_gap(f"Gaps in: {topic}", p, hits, topic))
        self.gaps.extend(found)
        return found

    def detect_gaps_for_question(self, question, papers):
        found = []
        for p in papers:
            hits = [kw for kw in FALLBACK_KEYWORDS if kw in (p.get("abstract") or "").lower()]
            if hits: found.append(self._make_gap(question, p, hits, question[:60]))
        self.gaps.extend(found)
        return found

    def save_gaps(self):
        Path("data/papers").mkdir(parents=True, exist_ok=True)
        Path("data/papers/research_gaps.json").write_text(json.dumps(self.gaps, indent=2), encoding="utf-8")

    def get_all_gaps(self): return list(self.gaps)
    def get_gaps_by_priority(self, p): return [g for g in self.gaps if g.get("priority") == p]

    def get_stats(self):
        bp, bt = {}, {}
        for g in self.gaps:
            bp[g["priority"]] = bp.get(g["priority"], 0) + 1
            bt[g["topic"]] = bt.get(g["topic"], 0) + 1
        return {"total": len(self.gaps), "by_priority": bp, "by_topic": bt}

_detector = None
def get_detector():
    global _detector
    if _detector is None: _detector = GapDetector()
    return _detector
