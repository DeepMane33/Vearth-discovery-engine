"""Tests for the novelty checker module."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import novelty
from engine import datasources
import engine.paper_ingestion


def _mock_all_datasources(monkeypatch):
    """Replace all datasource functions with no-op fakes to avoid network."""
    monkeypatch.setattr(datasources, "get_known_planets_near",
                        lambda ra, dec, radius_deg=0.15: [])
    monkeypatch.setattr(datasources, "get_known_planet_by_period",
                        lambda period_days, tolerance=0.01: [])
    monkeypatch.setattr(datasources, "query_exoplanet_archive",
                        lambda query: [])
    monkeypatch.setattr(datasources, "search_arxiv",
                        lambda query, max_results=5: [])
    monkeypatch.setattr(datasources, "get_tic_params",
                        lambda target: None)
    monkeypatch.setattr(datasources, "query_simbad",
                        lambda target: {})
    monkeypatch.setattr(datasources, "query_simbad_coordinates",
                        lambda target: {})
    monkeypatch.setattr(engine.paper_ingestion, "search_ads",
                        lambda query, rows=50, sort="citation_count desc",
                        year_range=None, token="": [])


def test_check_novelty_returns_expected_keys(monkeypatch):
    _mock_all_datasources(monkeypatch)
    result = novelty.check_novelty(
        "TIC 12345",
        period_days=3.5,
        depth_ppm=300,
    )
    assert "status" in result
    assert "checked" in result
    assert "references" in result
    assert "details" in result
    assert "ran_at" in result
    assert result["status"] in ("KNOWN", "POSSIBLY KNOWN", "NO MATCH FOUND")


def test_check_novelty_all_sources_checked(monkeypatch):
    _mock_all_datasources(monkeypatch)
    result = novelty.check_novelty("TIC 999999", period_days=999.0)
    assert result["status"] == "NO MATCH FOUND"
    assert "nasa_exoplanet_archive" in result["checked"]
    assert "simbad" in result["checked"]
    assert "arxiv" in result["checked"]
    assert "nasa_ads" in result["checked"]


def test_write_paper_blocks_without_novelty():
    """write_paper must raise ValueError when novelty is not provided."""
    from engine import synthesizer
    try:
        synthesizer.write_paper(
            "TIC 12345",
            {"transit": {}, "depth_ppm": 300, "snr": 10.0,
             "period_days": 3.5, "n_transits": 4},
            {"verification_tests": {"passed_count": 10, "total_tests": 15}},
            novelty=None,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "novelty check has not run" in str(e).lower()


def test_novelty_known_with_period_match(monkeypatch):
    """When archive returns a planet with matching period, status is KNOWN."""
    import engine.paper_ingestion as pi

    def fake_known_planets_near(ra, dec, radius_deg=0.15):
        return [{"pl_name": "KEEP-1 b", "pl_orbper": 3.5}]

    def fake_query_exoplanet_archive(query):
        return []

    def fake_search_arxiv(query, max_results=5):
        return []

    def fake_get_tic_params(target):
        return {"ticid": "12345"}

    def fake_query_simbad_coordinates(target):
        return {"ra": "10.0", "dec": "20.0"}

    def fake_search_ads(query, rows=50, sort="citation_count desc",
                        year_range=None, token=""):
        return []

    monkeypatch.setattr(datasources, "get_known_planets_near",
                        fake_known_planets_near)
    monkeypatch.setattr(datasources, "get_known_planet_by_period",
                        lambda period_days, tolerance=0.01: [])
    monkeypatch.setattr(datasources, "query_exoplanet_archive",
                        fake_query_exoplanet_archive)
    monkeypatch.setattr(datasources, "search_arxiv", fake_search_arxiv)
    monkeypatch.setattr(datasources, "get_tic_params", fake_get_tic_params)
    monkeypatch.setattr(datasources, "query_simbad_coordinates",
                        fake_query_simbad_coordinates)
    monkeypatch.setattr(pi, "search_ads", fake_search_ads)

    result = novelty.check_novelty("TIC 12345", period_days=3.5)
    assert result["status"] == "KNOWN"
    assert len(result["references"]) >= 1


def test_novelty_track_b_no_match(monkeypatch):
    """Track B with no literature matches returns NO MATCH FOUND."""
    import engine.paper_ingestion as pi

    def fake_search_arxiv(query, max_results=5):
        return []

    def fake_search_ads(query, rows=50, sort="citation_count desc",
                        year_range=None, token=""):
        return []

    monkeypatch.setattr(datasources, "search_arxiv", fake_search_arxiv)
    monkeypatch.setattr(pi, "search_ads", fake_search_ads)

    result = novelty.check_novelty("Mars", track="B",
                                   dataset="MSL RAD")
    assert result["track"] == "B"
    assert result["status"] == "NO MATCH FOUND"
    assert len(result["references"]) == 0


def test_novelty_track_b_possibly_known(monkeypatch):
    """Track B with arXiv match returns POSSIBLY KNOWN."""
    import engine.paper_ingestion as pi

    def fake_search_arxiv(query, max_results=5):
        return [{"title": "Mars radiation environment",
                  "url": "https://arxiv.org/abs/1234.5678"}]

    def fake_search_ads(query, rows=50, sort="citation_count desc",
                        year_range=None, token=""):
        return []

    monkeypatch.setattr(datasources, "search_arxiv", fake_search_arxiv)
    monkeypatch.setattr(pi, "search_ads", fake_search_ads)

    result = novelty.check_novelty("Mars", track="B",
                                   dataset="RAD")
    assert result["track"] == "B"
    assert result["status"] == "POSSIBLY KNOWN"
    assert len(result["references"]) >= 1


def test_novelty_drops_empty_titles(monkeypatch):
    """References with empty titles are dropped."""
    import engine.paper_ingestion as pi

    def fake_search_arxiv(query, max_results=5):
        return [{"title": "", "url": "https://arxiv.org/abs/1234"},
                {"title": "   ", "url": "https://arxiv.org/abs/5678"},
                {"title": "Real Paper", "url": "https://arxiv.org/abs/9999"}]

    def fake_search_ads(query, rows=50, sort="citation_count desc",
                        year_range=None, token=""):
        return []

    monkeypatch.setattr(datasources, "search_arxiv", fake_search_arxiv)
    monkeypatch.setattr(pi, "search_ads", fake_search_ads)

    result = novelty.check_novelty("Mars", track="B", dataset="RAD")
    titles = [r["name"] for r in result["references"]]
    assert "Real Paper" in titles
    assert "" not in titles
    assert "   " not in titles


def test_novelty_ads_skipped_without_token(monkeypatch):
    """ADS is skipped (not errored) when no token configured."""
    import engine.paper_ingestion as pi

    def fake_search_arxiv(query, max_results=5):
        return []

    def fake_search_ads(query, rows=50, sort="citation_count desc",
                        year_range=None, token=""):
        return []

    monkeypatch.setattr(datasources, "search_arxiv", fake_search_arxiv)
    monkeypatch.setattr(pi, "search_ads", fake_search_ads)
    monkeypatch.setattr("engine.novelty.NASA_ADS_TOKEN", "")

    result = novelty.check_novelty("TIC 12345", period_days=3.5)
    ads_detail = result["details"].get("nasa_ads", {})
    assert ads_detail.get("skipped") is True
