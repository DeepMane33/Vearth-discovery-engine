"""Tests for the target router module."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.target_router import route_target, classify_target, SOLAR_SYSTEM_BODIES


def test_solar_system_bodies_route_to_track_b():
    """Known solar-system bodies must route to Track B."""
    for body in ["mars", "venus", "jupiter", "saturn", "moon", "luna",
                 "mercury", "uranus", "neptune", "pluto", "sol", "sun"]:
        result = route_target(body)
        assert result["category"] == "solar_system", f"{body}: {result}"
        assert result["track"] == "B", f"{body}: {result}"
        assert result["error"] is None


def test_solar_system_case_insensitive():
    """Solar-system matching must be case-insensitive."""
    assert route_target("MARS")["category"] == "solar_system"
    assert route_target("Mars")["category"] == "solar_system"
    assert route_target("mArS")["category"] == "solar_system"


def test_unresolvable_target():
    """A made-up target must be unresolvable with a helpful error."""
    result = route_target("xyzzy_not_a_real_target_12345")
    assert result["category"] == "unresolvable"
    assert result["track"] is None
    assert result["error"] is not None
    assert "xyzzy_not_a_real_target_12345" in result["error"]


def test_empty_target():
    """Empty target must be unresolvable."""
    result = route_target("")
    assert result["category"] == "unresolvable"
    assert result["error"] == "empty target identifier"


def test_solar_system_bodies_set_nonempty():
    """The curated solar-system body list must not be empty."""
    assert len(SOLAR_SYSTEM_BODIES) > 0
    assert "mars" in SOLAR_SYSTEM_BODIES
    assert "jupiter" in SOLAR_SYSTEM_BODIES
    assert "moon" in SOLAR_SYSTEM_BODIES


def test_stellar_target_not_resolvable_without_network(monkeypatch):
    """Without network, a stellar target should be unresolvable (not silently
    treated as solar-system). We monkeypatch the network functions to fail."""
    import engine.target_router as tr

    def fail_simbad(target):
        raise ConnectionError("SIMBAD down")

    def fail_mast(*args, **kwargs):
        raise ConnectionError("MAST down")

    monkeypatch.setattr(tr, "_resolve_stellar", lambda target: {"resolved": False})

    result = route_target("TIC 268684168")
    # Without network, TIC can't be resolved -> unresolvable with helpful error
    assert result["category"] == "unresolvable"
    assert result["track"] is None
    assert "TIC 268684168" in result["error"]
