# Discovery Engine

An autonomous, multi-agent space-research pipeline. It pulls real TESS
satellite data, searches the light curve for transit-like anomalies, then a
critic agent self-questions the finding from every angle against a literature
memory (RAG) and either **veto**s or **advance**s it. A writer agent then
drafts an IMRaD research report citing the exact data logs.

Five agents, one loop:

1. **Collector** — `astroquery.mast` downloads the TESS light curve (cached). Falls back to a synthetic curve offline.
2. **Memory** — a vector store of prior literature for retrieval (RAG).
3. **Finder** — detrend + box-least-squares transit search + IsolationForest outliers (scikit-learn).
4. **Critic** — drafts a hypothesis, attacks it with skeptical self-questions, cross-checks each against retrieved literature, then veto / advance.
5. **Writer** — writes the IMRaD paper, citing data logs + literature IDs.

## Quick start

```powershell
cd "C:\Users\Anil Mane\Documents\Default Project\discovery-engine"
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>, pick a target (e.g. `TIC 268684168`), and run.

## LLM (the smart reasoning layer)

The Critic and Writer sound much better with a real LLM. Copy `.env.example`
to `.env` and set a provider + key:

- **Cerebras** (OpenAI-compatible): `LLM_PROVIDER=cerebras`, `LLM_MODEL=<model>`, `CEREBRAS_API_KEY=...`
- **OpenAI**: `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`, `OPENAI_API_KEY=...`
- **Anthropic**: `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-3-5-sonnet-latest`, `ANTHROPIC_API_KEY=...`

**Without any key / credits**, the whole pipeline still runs end-to-end using a
transparent, rule-based fallback for the Critic and Writer (clearly labelled
`fallback` in the UI). Real scientific prose requires a live LLM.

## Notes / honesty

- Single-band TESS photometry cannot confirm a planet; this is a research
  *prototyping* engine, not a discovery authority.
- Findings are conservative (default: veto unless S/N >= 7 and >= 2 recurrences).
- Saved JSON reports land in `data/reports/`.

## Layout

```
app.py                 Flask server
config.py              env + paths
engine/
  ingestion.py         Collector
  vectorstore.py        Memory (RAG)
  analyzer.py          Finder (ML)
  critic.py            Critic (self-questioning)
  synthesizer.py        Writer (paper)
  orchestrator.py      runs all 5 agents + plot
  llm.py               provider client + fallback
templates/ static/     minimal UI (no neon)
```
