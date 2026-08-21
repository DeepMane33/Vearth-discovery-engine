"""Vector Database Layer (The Memory).

A lightweight in-memory + JSON-persisted vector store with hashed
term-frequency embeddings and cosine similarity. Lets the Critic agent
perform RAG against a corpus of short paper summaries. Swap the embed()
function for a real embedding model if desired."""
import math
import json
import hashlib
from config import DATA_DIR, VECTOR_META

DIM = 2048
_corpus = []
_vectors = []
_meta_file = VECTOR_META


def _hash_tokens(text):
    toks = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    vec = [0.0] * DIM
    for tok in toks:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM
        vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed(text):
    return _hash_tokens(text)


def _cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def rebuild_from_seed_corpus():
    _corpus.clear()
    _vectors.clear()
    for entry in _seed_corpus():
        _corpus.append(entry)
        _vectors.append(embed(entry["text"]))
    _persist()


def add_document(doc_id, text, metadata=None):
    _corpus.append({"id": doc_id, "text": text,
                    "metadata": metadata or {}})
    _vectors.append(embed(text))
    _persist()


def query(text, k=5):
    if not _corpus:
        rebuild_from_seed_corpus()
    qv = embed(text)
    scored = sorted(range(len(_corpus)),
                    key=lambda i: _cosine(qv, _vectors[i]), reverse=True)
    out = []
    for i in scored[:k]:
        entry = _corpus[i]
        out.append({**entry, "score": _cosine(qv, _vectors[i])})
    return out


def _persist():
    data = [{"id": c["id"], "text": c["text"], "metadata": c.get("metadata", {})}
            for c in _corpus]
    _meta_file.write_text(json.dumps(data, indent=2))


def _load():
    if _meta_file.exists():
        data = json.loads(_meta_file.read_text())
        _corpus.clear()
        _vectors.clear()
        for d in data:
            _corpus.append(d)
            _vectors.append(embed(d["text"]))


def _seed_corpus():
    """A tiny literature baseline the Critic checks anomalies against."""
    return [
        {"id": "ref-2012-bls",
         "text": ("Kovacs et al. 2002 Box-fitting Least Squares (BLS) detects "
                  "periodic transitlike dips by folding the light curve on trial "
                  "periods and fitting a box shape. A strong single BLS peak with "
                  "shallow depth and short duration is the classic exoplanet "
                  "signature. V-shaped deep transits are often grazing eclipsing "
                   "binaries, not planets. Secondary eclipses indicate a binary "
                   "star.")},
        {"id": "ref-2015-cbv",
         "text": ("TESS / Kepler produce cotrending basis vectors (CBVs) that "
                  "correct systematic detector trends from the spacecraft roll "
                  "and thermal breathing. Uncorrected CBVs can mimic shallow "
                  "dips at the orbital roll period (~13.7 days). If a candidate "
                  "dip repeats at a known instrumental cadence it is likely an "
                  "artifact, not astrophysical.")},
        {"id": "ref-2018-contam",
         "text": ("Background contamination from a nearby unresolved star can "
                  "dilute an eclipsing binary signal into a shallow planetlike "
                  "transit (blending). Centroid shift analysis is required: a "
                  "real planet dip has the flux centroid on the target star, "
                  "while a blended binary shifts the centroid off-target "
                  "during ingress and egress.")},
        {"id": "ref-2019-ttics",
         "text": ("A genuine transiting exoplanet produces a transit with depth "
                  "delta proportional to (Rp/Rs)^2, duration that scales with "
                  "orbital velocity sqrt(GMs/a), and a period that obeys "
                  "Kepler's third law P^2 = a^3 / Ms (in solar units). The "
                  "light curve should be achromatic across TESS bandpasses and "
                  "U-shaped for small planets.")},
        {"id": "ref-2021-noise",
         "text": ("Statistical artifacts from single noisy points are ruled "
                  "out by requiring multiple transit events at the same phase. "
                  "A single dip with no recurrence is almost always a cosmic "
                  "ray hit, momentum dump, or outlier and should be vetoed "
                  "unless folded data strengthens the signal.")},
        {"id": "ref-2023-falsepos",
         "text": ("False positive vetting requires rejecting eclipsing "
                  "binaries through odd-even transit depth mismatch, secondary "
                  "eclipse search, and centroid offset. Odd-even depth "
                  "differences at >3 sigma strongly indicate a nearby "
                  "eclipsing binary at twice the period rather than a planet.")},
        {"id": "ref-2024-stellarvar",
         "text": ("Stellar variability such as starspots and flares produces "
                  "quasi-periodic modulation on the stellar rotation period. "
                  "Starspot crossings during transit can add asymmetry but do "
                  "not produce a clean U-shaped box dip that repeats phase-"
                  "locked across many orbits. Aperiodic flares are sharp and "
                   "positive and do not repeat.")},
    ]


_load()
if not _corpus:
    rebuild_from_seed_corpus()
