# VEARTH Discovery Engine v3.0

An autonomous, multi-agent space-research pipeline. It pulls real TESS
satellite data, searches the light curve for transit-like anomalies, then a
critic agent self-questions the finding from every angle against a literature
memory (RAG) and either **veto**s or **advance**s it. A writer agent then
drafts an IMRaD research report citing the exact data logs.

## v3.0 Enhancements

- **15-test verification battery**: phase curve, chromaticity, harmonic alias, transit shape metrics, window function, stellar density consistency
- **Multi-planet detection**: Sequentially detects up to 5 planets per target
- **Monte Carlo uncertainty**: 500-iteration bootstrap for parameter posteriors
- **Atmospheric characterization**: TSM/ESM metrics for JWST observability
- **Orbital dynamics**: Hill stability, tidal locking, eccentricity constraints
- **Multi-mission data**: Gaia DR3, TIC, Kepler/K2 cross-matching
- **Mass-radius relations**: Chen & Kipping (2017) mass estimates, bulk density
- **Enhanced AI dialog**: Physics-informed self-verification with atmospheric metrics

Eight stages, one loop:

1. **Collector** — `astroquery.mast` downloads the TESS light curve (cached). Also queries Gaia DR3, TIC, and Kepler/K2. Falls back to a synthetic curve offline.
2. **Cross-match** — checks NASA Exoplanet Archive for known planets near the target.
3. **Finder** — detrend + box-least-squares transit search + IsolationForest outliers. Sequentially detects up to 5 planets.
4. **Validator** — 15-test verification battery (SNR, recurrence, odd-even, secondary eclipse, V-shape, bootstrap FAP, stellar variability, physical plausibility, catalog cross-match, phase curve, chromaticity, harmonic alias, transit shape, window function, stellar density consistency).
5. **Critic** — LLM drafts a hypothesis, attacks it with skeptical self-questions, cross-checks each against retrieved literature, then veto / advance.
6. **Verifier** — 4-round AI self-verification with physics evidence (atmospheric metrics, orbital stability, mass-radius).
7. **Writer** — writes the IMRaD paper with multi-planet comparison, Monte Carlo uncertainties, and JWST characterization metrics.
8. **Save** — stores results as JSON with publication-quality figures.

## Quick start

```powershell
cd discovery-engine
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
  ingestion.py         Collector (TESS + multi-mission)
  vectorstore.py        Memory (RAG)
  analyzer.py          Finder (ML) + multi-planet detection
  critic.py            Critic (self-questioning) + 15-test validation
  validation_tests.py  15 verification tests
  physics.py           Planet physics + atmospheric metrics
  datasources.py       SIMBAD, Exoplanet Archive, Gaia, TIC, arXiv
  verifier.py          AI self-verification (4-round dialog)
  synthesizer.py       Writer (IMRaD paper) + multi-planet tables
  orchestrator.py      runs all 8 stages + Monte Carlo + plots
  plots.py             Publication-quality figures
  llm.py               provider client + fallback
templates/ static/     minimal UI (no neon)
```
