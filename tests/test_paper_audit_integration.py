"""Tests for paper audit integration with the synthesizer."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import paper_audit
from engine import synthesizer


def test_audit_catches_fabricated_number():
    """audit_paper_numbers flags numbers not in results JSON."""
    paper = "The planet radius is 99999 Re at period 3.5 days."
    results_json = {
        "analysis": {"period_days": 3.5, "depth_ppm": 300, "snr": 10.0},
        "physics": {"planet_radius_rearth": 2.5},
    }
    result = paper_audit.audit_paper_numbers(paper, results_json)
    assert result["passed"] is False
    assert any(u["raw"] == "99999" for u in result["unverified_numbers"])


def test_audit_passes_when_all_numbers_traceable():
    """audit_paper_numbers passes when all numbers are in results JSON."""
    paper = "The planet radius is 2.5 Re at period 3.5 days with S/N of 10.0."
    results_json = {
        "analysis": {"period_days": 3.5, "depth_ppm": 300, "snr": 10.0},
        "physics": {"planet_radius_rearth": 2.5},
    }
    result = paper_audit.audit_paper_numbers(paper, results_json)
    assert result["passed"] is True
    assert result["unverified_count"] == 0


def test_header_contains_ai_warning():
    """generate_audit_header includes the mandatory AI warning."""
    header = paper_audit.generate_audit_header({
        "target": "TIC 12345",
        "ingestion": {"source": "mast_tess"},
        "validation": {"veto": False},
    })
    assert "Draft generated with AI assistance" in header
    assert "Requires human verification" in header
    assert "before any submission" in header


def test_strip_removes_unverified_citations():
    """strip_unverified_citations removes citations not in retrieved list."""
    retrieved = [{"id": "ref-2012-bls", "title": "BLS paper"}]
    paper = "See [ref-2012-bls] and [ref-fake-paper] here."
    cleaned = paper_audit.strip_unverified_citations(paper, retrieved)
    assert "[ref-2012-bls]" in cleaned
    assert "[ref-fake-paper]" not in cleaned


def test_write_paper_returns_audit_result():
    """write_paper returns paper_audit in its result dict."""
    from engine import synthesizer
    result = synthesizer.write_paper(
        "TIC 12345",
        {"transit": {}, "depth_ppm": 300, "snr": 10.0,
         "period_days": 3.5, "n_transits": 4},
        {"verification_tests": {"passed_count": 10, "total_tests": 15},
         "veto": False, "confidence": "low",
         "reason": "passes verification"},
        novelty={"status": "NO MATCH FOUND",
                 "checked": ["nasa_exoplanet_archive", "simbad",
                             "arxiv", "nasa_ads"],
                 "references": [],
                 "details": {}},
        retrieved_papers=[],
    )
    assert "paper_audit" in result
    audit = result["paper_audit"]
    assert "passed" in audit
    assert "total_numbers_found" in audit
    assert "verified_numbers" in audit
    assert "unverified_count" in audit
    assert "json_number_count" in audit
