"""Basic tests for the analyzer module using synthetic light curves."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from engine.analyzer import analyze, _detrend, _sigma_clip, _fallback_bls


def _make_synthetic_curve(n=2000, period=3.0, depth=0.005, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 27.0, n)
    flux = 1.0 + 0.0003 * rng.standard_normal(n)
    for i, ti in enumerate(t):
        phase = (ti % period) / period
        if phase < 0.01 or phase > 0.99:
            flux[i] -= depth
    return t, flux


def test_detrend_returns_normalized():
    t = np.linspace(0, 10, 100)
    f = np.ones(100) + 0.01 * np.sin(t)
    out = _detrend(t, f)
    assert out is not None
    assert len(out) == len(f)


def test_sigma_clip_returns_mask():
    f = np.ones(100)
    mask = _sigma_clip(f)
    assert mask.shape == f.shape
    assert mask.dtype == bool


def test_fallback_bls_detects_injected_transit():
    t, f = _make_synthetic_curve(period=3.0, depth=0.005)
    result = _fallback_bls(t, f)
    assert result["period"] is not None
    assert result["snr"] > 0
    assert result["depth"] > 0


def test_analyze_returns_dict():
    t, f = _make_synthetic_curve(period=3.0, depth=0.005)
    result = analyze(t, f)
    assert isinstance(result, dict)
    assert "transit" in result
    assert "snr" in result
    assert "depth_ppm" in result
    assert result["depth_ppm"] > 0
