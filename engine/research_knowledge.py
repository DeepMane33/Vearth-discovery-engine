"""Research Knowledge Store – seed corpus, embeddings, similarity search."""

import hashlib, json, logging, math
from pathlib import Path

logger = logging.getLogger(__name__)

SEED_CORPUS = [
    {"id": "bls", "text": "BLS transit detection algorithms identify periodic dimming events in stellar light curves.", "topic": "detection"},
    {"id": "contamination", "text": "Background contamination can mimic planetary transits through blended eclipsing binaries.", "topic": "false_positives"},
    {"id": "vespa", "text": "VESPA calculates false positive probabilities using stellar and transit parameters.", "topic": "validation"},
    {"id": "hz", "text": "Kopparapu 2013 defines habitable zone boundaries based on runaway and maximum greenhouse limits.", "topic": "habitability"},
    {"id": "mass_radius", "text": "Chen & Kipping 2017 derive a mass-radius relation: M ∝ R^α with α ≈ 2.06 for rocky planets.", "topic": "planet_properties"},
    {"id": "atmospheric", "text": "JWST transit spectroscopy uses TSM and ESM to prioritize atmospheric characterization targets.", "topic": "atmospheres"},
    {"id": "tidal", "text": "Tidal locking timescales depend on orbital distance, stellar mass, and planet interior structure.", "topic": "dynamics"},
    {"id": "formation_ca", "text": "Core accretion builds planets from dust to embryos; disk instability fragments massive disks directly.", "topic": "formation"},
    {"id": "stellar_abundances", "text": "APOGEE and GALAH provide chemical abundances for stellar age-chemical tagging of exoplanet hosts.", "topic": "stellar"},
    {"id": "biosignatures", "text": "O2, O3, and CH4 together are key biosignature gases indicative of biological activity.", "topic": "habitability"},
    {"id": "radius_valley", "text": "The radius valley at 1.5-2.0 Re is explained by photoevaporation stripping rocky planet envelopes.", "topic": "populations"},
    {"id": "ttv", "text": "Transit Timing Variations reveal additional planets and constrain masses through gravitational interactions.", "topic": "detection"},
]


def _embed(text: str) -> list[float]:
    h = hashlib.md5(text.encode()).digest()
    vec = [(b - 128) / 128.0 for b in h] * 16
    vec = vec[:2048]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot


class ResearchKnowledge:
    def __init__(self):
        self.seed = SEED_CORPUS
        self.seed_emb = [(s, _embed(s["text"])) for s in self.seed]
        self._paper_store = None
        self._method_store = None
        try:
            from . import paper_ingestion
            self._paper_store = paper_ingestion
        except ImportError:
            pass
        try:
            from . import methodology_extractor
            self._method_store = methodology_extractor
        except ImportError:
            pass

    def query(self, text: str, k: int = 10) -> list[dict]:
        qe = _embed(text)
        hits = []
        for s, se in self.seed_emb:
            hits.append({"score": _cosine(qe, se), "source": "seed", **s})
        if self._paper_store and hasattr(self._paper_store, "search"):
            for r in self._paper_store.search(text, k=k):
                hits.append({"score": r.get("score", 0), "source": "paper", **r})
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k]

    def find_prior_work(self, question: str, k: int = 5) -> list[dict]:
        return self.query(question, k)

    def find_methodology(self, research_type: str, k: int = 5) -> list[dict]:
        if self._method_store and hasattr(self._method_store, "search"):
            return self._method_store.search(research_type, k=k)
        qe = _embed(research_type)
        hits = []
        for s, se in self.seed_emb:
            if s["topic"] == research_type:
                hits.append({"score": _cosine(qe, se), "source": "seed", **s})
        return hits[:k]

    def get_stats(self) -> dict:
        return {"seed_count": len(self.seed), "has_papers": self._paper_store is not None, "has_methodology": self._method_store is not None}


_instance: ResearchKnowledge | None = None

def get_knowledge() -> ResearchKnowledge:
    global _instance
    if _instance is None:
        _instance = ResearchKnowledge()
    return _instance
