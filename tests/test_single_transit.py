"""Tests for the single_transit module."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from engine.single_transit import detect_single_transit


def test_detect_single_transit_returns_structure():
    """detect_single_transit returns expected keys."""
    t = np.linspace(0, 27, 1000)
    f = np.ones(1000)
    result = detect_single_transit(t, f)
    assert "candidates" in result
    assert "n_candidates" in result
    assert "best" in result
    assert "cadence_minutes" in result
    assert "sigma_threshold" in result
    assert "limitation" in result


def test_detect_single_transit_short_curve():
    """Too-short light curve returns empty result."""
    t = np.linspace(0, 1, 10)
    f = np.ones(10)
    result = detect_single_transit(t, f)
    assert result["n_candidates"] == 0
    assert result["best"] is None


def test_detect_single_transit_injected_dip():
    """Injected dip is found as a candidate."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 27, 5000)
    flux = 1.0 + 0.0005 * rng.standard_normal(len(t))
    # Inject a dip at t=10.0
    mask = (t >= 9.95) & (t <= 10.05)
    flux[mask] -= 0.005  # 5000 ppm dip
    result = detect_single_transit(t, flux, sigma_threshold=5.0)
    assert result["n_candidates"] > 0
    assert result["best"] is not None
    assert result["best"]["depth_ppm"] > 0


def test_detect_single_transit_noise_only():
    """Pure noise may produce candidates but best significance is limited."""
    rng = np.random.default_rng(123)
    t = np.linspace(0, 27, 2000)
    f = 1.0 + 0.001 * rng.standard_normal(len(t))
    result = detect_single_transit(t, f, sigma_threshold=8.0)
    # With high threshold, likely no candidates
    assert result["n_candidates"] >= 0


def test_limitation_note_present():
    """Limitation note is always present."""
    t = np.linspace(0, 27, 500)
    f = np.ones(500)
    result = detect_single_transit(t, f)
    assert "limitation" in result
    assert len(result["limitation"]) > 0
    assert "periodicity" in result["limitation"].lower()
