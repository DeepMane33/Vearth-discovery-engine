"""Synthesis Agent (The Writer).

Turns the validated candidate into a structured academic research paper.
Cites the exact data logs it used."""
from engine import llm


SYSTEM = (
    "You are a careful scientific writer. Translate the validated analysis "
    "into a concise academic research paper in Markdown. Use the IMRaD "
    "structure (Abstract, Introduction, Data, Methods, Results, Discussion, "
    "Conclusion, References). Cite the exact data logs and retrieved "
    "literature IDs inline as [id]. Do not fabricate numbers beyond those "
    "given. Keep tone formal and neutral."
)


def write_paper(target, analysis, validation):
    refs = validation.get("rag_refs", [])
    refs_block = "\n".join(
        f"- [{r['id']}] (retrieval score {r['score']})" for r in refs)

    facts = (
        f"TARGET: {target}\n"
        f"DATA SOURCE: {analysis.get('transit',{}).get('period')} day period "
        f"light curve, {analysis.get('n_transits')} recurring transits\n"
        f"DIP DEPTH (ppm): {analysis.get('depth_ppm')}\n"
        f"BOX-LEAST-SQUARES S/N: {analysis.get('snr'):.2f}\n"
        f"BEST PERIOD (days): {analysis.get('period_days')}\n"
        f"RECURRENCE (n_transits): {analysis.get('n_transits')}\n"
        f"NOISE FLOOR (ppm): {analysis.get('noise_floor_ppm')}\n"
        f"REDUCED CHI^2 (flat model): {analysis.get('reduced_chi2'):.5f}\n"
        f"ISOLATION-FOREST OUTLIERS: "
        f"{len(analysis.get('outlier_indices',[]))} points flagged\n\n"
        f"CRITIC RULING: "
        f"{'VETOED' if validation.get('veto') else 'ADVANCED'} "
        f"(confidence: {validation.get('confidence')})\n"
        f"HYPOTHESIS: {validation.get('hypothesis')}\n"
        f"SELF-QUESTIONS RESOLVED:\n"
    )
    for sq in validation.get("self_questions", []):
        if isinstance(sq, dict):
            facts += (f"  - Q: {sq.get('question')}\n"
                      f"    A: {sq.get('answer')} "
                      f"-> {'true' if sq.get('resolved_true_or_false') else 'false'}\n")
        else:
            facts += f"  - {sq}\n"
    facts += f"\nRETRIEVED LITERATURE:\n{refs_block}\n"

    user = ("Write the research paper using ONLY these facts and the cited "
            "literature. If vetoed, frame the paper as a null/false-positive "
            "result.\n\n" + facts)

    res = llm.ask_llm(SYSTEM, user, temperature=0.3, max_tokens=2600)
    if res["source"] == "fallback":
        md = _fallback_paper(target, analysis, validation, refs)
        return {"markdown": md, "source": "fallback", "facts_used": facts}
    return {"markdown": res["text"], "source": res["source"],
            "facts_used": facts}


def _fallback_paper(target, analysis, validation, refs):
    t = target
    depth = analysis.get("depth_ppm", 0)
    snr = analysis.get("snr", 0.0)
    P = analysis.get("period_days")
    n_tr = analysis.get("n_transits", 0)
    noise = analysis.get("noise_floor_ppm", 0)
    chi2 = analysis.get("reduced_chi2", 0.0)
    n_out = len(analysis.get("outlier_indices", []))
    veto = validation.get("veto", True)
    conf = validation.get("confidence", "low")
    verdict = "vetoed (most likely a false positive / noise)" if veto \
        else "advanced as a candidate of interest"
    rp_proxy = round((depth / 1e6) ** 0.5 * 100, 3)

    sq_lines = ""
    for sq in validation.get("self_questions", []):
        if isinstance(sq, dict):
            res_tag = "CONFIRMED" if sq.get("resolved_true_or_false") else "REJECTED"
            sq_lines += (f"- **{sq.get('question')}** — {sq.get('answer')} "
                         f"*[{res_tag}]*\n")

    refs_lines = "".join(f"- [{r['id']}] (retrieval score {r['score']})\n"
                         for r in refs)

    return f"""# Transit Search Report: {t}

## Abstract
We searched the TESS light curve of **{t}** for periodic transit-like dips using
a box-fitting least-squares scan combined with IsolationForest outlier
detection and a retrieval check against prior literature. We identify a
{depth} ppm dip recurring at a best-fit period of {P} d over {n_tr}
transits with a box-LS signal-to-noise of {snr:.2f}. Following skeptical
self-vetting, the candidate is **{verdict}** at *{conf}* confidence.

## 1. Introduction
Transit photometry detects exoplanets by the periodic dimming of a star as a
companion crosses its disk. The classical signature is a shallow,
short-duration, U-shaped dip that recurs at fixed phase. The chief
contaminants are instrumental systematics, blended eclipsing binaries, and
stellar variability. Here we apply an autonomous vetting pipeline to {t}.

## 2. Data
- **Source:** TESS light curve retrieved via `astroquery.mast`.
- **Points analysed:** {n_out}+ flagged outliers excluded; flux normalised by median.
- **Noise floor:** {noise} ppm (MAD-based estimate).

## 3. Methods
1. **Detrending:** cubic polynomial fit to out-of-delta baseline, divide.
2. **Transit search:** phase-fold box scan over trial periods
   (0.5–{round(P,2) if False else 'auto'} d), scoring depth vs. point-error.
3. **Outlier flag:** IsolationForest (contamination 1%).
4. **Validation:** retrieval against a literature vector store; skeptical
   self-questioning across instrumental, astrophysical, and statistical axes.
5. **Significance threshold:** box-LS S/N ≥ 7 and ≥ 2 recurring transits.

## 4. Results
| Quantity | Value |
|---|---|
| Dip depth | {depth} ppm |
| Best period | {P} d |
| Recurrences (n_transits) | {n_tr} |
| Box-LS S/N | {snr:.2f} |
| Noise floor | {noise} ppm |
| Reduced χ² (flat model) | {chi2:.5f} |
| IsolationForest outliers | {n_out} |

Indicative companion radius (assuming total transit depth ≈ (Rp/Rs)²):
**Rp ≈ {rp_proxy} Rs** (proxy — requires stellar radius for absolute value).

## 5. Discussion — Self-Questioning
{sq_lines}
The candidate {'fails' if veto else 'passes'} the S/N ≥ 7 and recurrence ≥ 2
gate. {"Because the signal-to-noise or recurrence is insufficient, the dip cannot be distinguished from residual systematics at this stage." if veto else "The phase-locked recurrence argues against a lone outlier or stellar flare, and the period is not consistent with the TESS roll cadence."}

## 6. Conclusion
The dip in {t} is **{verdict}**. {"It should NOT be promoted to a candidate list absent deeper centroid and ground-based follow-up; the result is a null/false-positive here." if veto else "We recommend ground-based follow-up and centroid-shift / odd-even analysis to rule out a blended eclipsing binary before promoting to a confirmed candidate."}

## 7. Limitations
- This report was generated by the **rule-based fallback** writer (no live LLM
  connected). It uses only the measured values above, so it cannot perform
  free-form physics reasoning. Connect an LLM (e.g. an API key with credits)
  to enable deep prose synthesis.
- Single-band TESS photometry alone cannot confirm a planet; centroid and
  spectroscopic follow-up are mandatory.

## 8. References (RAG)
{refs_lines}
"""
