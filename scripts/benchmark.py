"""VEARTH Validation Benchmark.

Runs the engine's analysis + validation battery on a list of TESS targets and
compares the engine's verdict against ground truth from the NASA Exoplanet
Archive. Outputs recall, false-positive rate, and per-test FP catch rates.

No thresholds are tuned. Ground truth comes from the archive, not from memory.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --targets TIC 268684168 TIC 352359478
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import ingestion, analyzer, validation_tests, datasources

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "benchmark",
)

# Starter target list. Ground truth is looked up live from the NASA Exoplanet
# Archive for each target, so this list does not need to be perfectly curated.
# Add or remove targets as needed.
DEFAULT_TARGETS = [
    "TIC 268684168",
    "TIC 352359478",
    "TOI 700",
    "Pi Mensae",
    "WASP-18",
    "HD 209458",
    "TIC 417780848",   # TOI 700 host
    "TIC 141147538",   # known quiet-ish star (placeholder - verify)
    "TIC 306271810",   # placeholder - verify
]

ENGINE_VETO_THRESHOLD = 5  # engine vetoes if fewer than this many tests pass


def _engine_verdict(test_results: dict, snr: float, n_transits: int) -> str:
    """Mirror the engine's rule-based veto logic (critic._rule_veto)."""
    passed = test_results.get("passed_count", 0)
    total = test_results.get("total_tests", 1)
    depth = None
    # depth not needed for the verdict string; pass-through is fine
    if passed < ENGINE_VETO_THRESHOLD:
        return "vetoed"
    if snr < 7.0:
        return "vetoed"
    if n_transits < 3:
        return "vetoed"
    return "advanced"


def _ground_truth_known_planet(target: str) -> dict:
    """Query the NASA Exoplanet Archive for known planets near the target.

    Returns {"known_planet": bool, "planets": [...], "source": "archive"|"error"}
    """
    result = {"known_planet": False, "planets": [], "source": "archive"}
    try:
        coords = datasources.query_simbad_coordinates(target)
        ra = coords.get("ra")
        dec = coords.get("dec")
        if ra is None or dec is None:
            result["source"] = "no_coords"
            return result
        ra_f = float(ra)
        dec_f = float(dec)
        planets = datasources.get_known_planets_near(ra_f, dec_f, radius_deg=0.15)
        if planets:
            result["known_planet"] = True
            result["planets"] = planets[:5]
        return result
    except Exception as e:
        result["source"] = f"error:{type(e).__name__}"
        return result


def run_target(target: str) -> dict:
    """Run the engine on a single target and compare to archive ground truth."""
    record = {
        "target": target,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": "unknown",
        "snr": None,
        "depth_ppm": None,
        "period_days": None,
        "n_transits": None,
        "tests_passed": None,
        "tests_total": None,
        "engine_verdict": "error",
        "ground_truth": {"known_planet": False, "planets": [], "source": "not_run"},
        "match": None,          # True=TP, False=FP, None=unknown/truth-unavailable
        "failed_tests": [],
        "passed_tests": [],
        "error": None,
    }

    # --- Data collection ---
    try:
        stellar_info = ingestion.fetch_stellar_info(target)
        lc = ingestion.fetch_tess_light_curve(target)
        record["data_source"] = lc.get("source", "unknown")
    except Exception as e:
        record["error"] = f"data_fetch:{type(e).__name__}:{e}"
        # still attempt ground truth
        record["ground_truth"] = _ground_truth_known_planet(target)
        return record

    # --- Analysis ---
    try:
        analysis = analyzer.analyze(lc["time"], lc["flux"], metadata=stellar_info)
        transit = analysis.get("transit", {})
        record["snr"] = analysis.get("snr")
        record["depth_ppm"] = analysis.get("depth_ppm")
        record["period_days"] = analysis.get("period_days")
        record["n_transits"] = transit.get("n_transits")
    except Exception as e:
        record["error"] = (record["error"] or "") + f"; analyze:{type(e).__name__}:{e}"
        record["ground_truth"] = _ground_truth_known_planet(target)
        return record

    # --- Validation battery ---
    try:
        test_results = validation_tests.run_all_tests(
            lc["time"],
            lc["flux"],
            analysis,
            known_planets=record["ground_truth"].get("planets") or None,
            stellar_info=stellar_info,
        )
        record["tests_passed"] = test_results.get("passed_count")
        record["tests_total"] = test_results.get("total_tests")

        for t in test_results.get("tests", []):
            if t.get("passed"):
                record["passed_tests"].append(t["test_name"])
            else:
                record["failed_tests"].append(t["test_name"])

        record["engine_verdict"] = _engine_verdict(
            test_results, record["snr"] or 0.0, record["n_transits"] or 0
        )
    except Exception as e:
        record["error"] = (record["error"] or "") + f"; validate:{type(e).__name__}:{e}"

    # --- Ground truth from archive ---
    record["ground_truth"] = _ground_truth_known_planet(target)

    # --- Match engine vs truth ---
    gt = record["ground_truth"]
    if gt.get("source", "").startswith("error") or gt.get("source") == "no_coords":
        record["match"] = None  # truth unavailable
    else:
        engine_positive = record["engine_verdict"] == "advanced"
        truth_positive = bool(gt.get("known_planet"))
        record["match"] = engine_positive == truth_positive

    return record


def summarize(records: list[dict]) -> dict:
    """Compute recall, FPR, and per-test FP catch stats."""
    tp = fp = tn = fn = 0
    truth_unavailable = 0
    fp_test_failures: dict[str, int] = {}

    for r in records:
        match = r.get("match")
        if match is None:
            truth_unavailable += 1
            continue

        engine_positive = r["engine_verdict"] == "advanced"
        truth_positive = bool(r["ground_truth"].get("known_planet"))

        if engine_positive and truth_positive:
            tp += 1
        elif engine_positive and not truth_positive:
            fp += 1
            for test_name in r.get("failed_tests", []):
                fp_test_failures[test_name] = fp_test_failures.get(test_name, 0) + 1
        elif not engine_positive and not truth_positive:
            tn += 1
        else:  # not engine_positive and truth_positive
            fn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None

    return {
        "total_targets": len(records),
        "truth_available": tp + fp + tn + fn,
        "truth_unavailable": truth_unavailable,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "recall": round(recall, 4) if recall is not None else None,
        "false_positive_rate": round(fpr, 4) if fpr is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
        "fp_caught_by_test": fp_test_failures,
    }


def write_outputs(records: list[dict], summary: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")
    md_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, default=str)

    lines = []
    lines.append("# VEARTH Validation Benchmark Report")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total targets: {summary['total_targets']}")
    lines.append(f"- Truth available: {summary['truth_available']}")
    lines.append(f"- Truth unavailable: {summary['truth_unavailable']}")
    lines.append(f"- True positives: {summary['true_positives']}")
    lines.append(f"- False positives: {summary['false_positives']}")
    lines.append(f"- True negatives: {summary['true_negatives']}")
    lines.append(f"- False negatives: {summary['false_negatives']}")
    lines.append(f"- Recall: {summary['recall']}")
    lines.append(f"- False-positive rate: {summary['false_positive_rate']}")
    lines.append(f"- Precision: {summary['precision']}")
    lines.append("")
    lines.append("## False positives caught by validation test")
    if summary["fp_caught_by_test"]:
        for test_name, count in sorted(
            summary["fp_caught_by_test"].items(), key=lambda x: -x[1]
        ):
            lines.append(f"- {test_name}: {count}")
    else:
        lines.append("- None recorded")
    lines.append("")
    lines.append("## Per-target results")
    lines.append(
        "| Target | Data source | Engine verdict | Ground truth | Match | Tests passed | Failed tests |"
    )
    for r in records:
        gt = r["ground_truth"]
        truth_str = "known planet" if gt.get("known_planet") else "no known planet"
        if gt.get("source", "").startswith("error") or gt.get("source") == "no_coords":
            truth_str = f"unavailable ({gt.get('source')})"
        match_str = {
            True: "TP" if r["engine_verdict"] == "advanced" else "TN",
            False: "FP" if r["engine_verdict"] == "advanced" else "FN",
            None: "?",
        }.get(r["match"], "?")
        failed = "; ".join(r.get("failed_tests", [])) or "—"
        lines.append(
            f"| {r['target']} | {r['data_source']} | {r['engine_verdict']} | "
            f"{truth_str} | {match_str} | {r['tests_passed']}/{r['tests_total']} | {failed} |"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"JSON results: {json_path}")
    print(f"Markdown report: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VEARTH validation benchmark")
    parser.add_argument(
        "targets",
        nargs="*",
        help="Target identifiers (TIC/TOI/star names). Defaults to built-in list.",
    )
    args = parser.parse_args()

    targets = args.targets if args.targets else DEFAULT_TARGETS
    print(f"Running benchmark on {len(targets)} target(s): {targets}")

    records = []
    for target in targets:
        print(f"  -> {target} ...", end=" ", flush=True)
        try:
            record = run_target(target)
            records.append(record)
            print(f"{record['engine_verdict']} (tests={record['tests_passed']}/{record['tests_total']})")
        except Exception as e:
            print(f"ERROR: {e}")
            records.append({
                "target": target,
                "error": str(e),
                "engine_verdict": "error",
                "match": None,
            })

    summary = summarize(records)
    print("\nBenchmark summary:")
    print(json.dumps(summary, indent=2, default=str))

    write_outputs(records, summary)


if __name__ == "__main__":
    main()
