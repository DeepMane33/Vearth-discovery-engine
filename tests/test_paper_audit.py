"""Tests for the paper_audit module."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import paper_audit


def test_extract_numbers():
    nums = paper_audit.extract_numbers("Depth is 300 ppm, period is 3.5 days, S/N = 8.2")
    values = [n["value"] for n in nums]
    assert 300.0 in values
    assert 3.5 in values
    assert 8.2 in values


def test_extract_numbers_skips_years_and_versions():
    nums = paper_audit.extract_numbers("Year 2026, version v3.0, depth 150 ppm")
    values = [n["value"] for n in nums]
    assert 150.0 in values
    assert 2026.0 not in values


def test_collect_json_numbers():
    obj = {
        "depth_ppm": 300,
        "snr": 8.5,
        "nested": {"period_days": 3.5, "n_transits": 4},
        "list": [1.0, 2.0, 3.0],
        "text": "not a number",
        "flag": True,
    }
    found = paper_audit.collect_json_numbers(obj)
    assert 300.0 in found
    assert 8.5 in found
    assert 3.5 in found
    assert 1.0 in found
    assert 2.0 in found
    assert 3.0 in found


def test_audit_paper_all_verified():
    paper = "The transit depth is 300 ppm with S/N of 8.5 at period 3.5 days."
    results_json = {
        "analysis": {"depth_ppm": 300, "snr": 8.5, "period_days": 3.5},
    }
    result = paper_audit.audit_paper_numbers(paper, results_json)
    assert result["passed"] is True
    assert result["unverified_count"] == 0
    assert result["verified_numbers"] >= 3


def test_audit_paper_unverified():
    paper = "The transit depth is 999 ppm at period 3.5 days."
    results_json = {
        "analysis": {"depth_ppm": 300, "snr": 8.5, "period_days": 3.5},
    }
    result = paper_audit.audit_paper_numbers(paper, results_json)
    assert result["passed"] is False
    assert result["unverified_count"] >= 1
    assert any(u["raw"] == "999" for u in result["unverified_numbers"])


def test_extract_citations():
    cites = paper_audit.extract_citations("See [ref-2012-bls] and [arxiv-2103.12345].")
    assert "ref-2012-bls" in cites
    assert "arxiv-2103.12345" in cites


def test_verify_citations_all_verified():
    retrieved = [
        {"id": "ref-2012-bls", "title": "BLS paper"},
        {"arxiv_id": "2103.12345", "title": "Some paper"},
    ]
    paper = "See [ref-2012-bls] and [arxiv_2103.12345]."
    result = paper_audit.verify_citations(paper, retrieved)
    assert result["passed"] is True
    assert result["dropped_citations"] == []


def test_verify_citations_drops_unverified():
    retrieved = [{"id": "ref-2012-bls", "title": "BLS paper"}]
    paper = "See [ref-2012-bls] and [ref-fake-paper]."
    result = paper_audit.verify_citations(paper, retrieved)
    assert result["passed"] is False
    assert "ref-fake-paper" in result["dropped_citations"]


def test_strip_unverified_citations():
    retrieved = [{"id": "ref-2012-bls", "title": "BLS paper"}]
    paper = "See [ref-2012-bls] and [ref-fake-paper] here."
    cleaned = paper_audit.strip_unverified_citations(paper, retrieved)
    assert "[ref-2012-bls]" in cleaned
    assert "[ref-fake-paper]" not in cleaned


def test_generate_audit_header():
    header = paper_audit.generate_audit_header({
        "target": "TIC 12345",
        "ingestion": {"source": "mast_tess"},
        "validation": {"veto": False},
    })
    assert "Draft generated with AI assistance" in header
    assert "Requires human verification" in header
    assert "TIC 12345" in header
