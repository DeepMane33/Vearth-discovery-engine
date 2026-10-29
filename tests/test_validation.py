"""Basic tests for the validation_tests module."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from engine.validation_tests import (
    test_snr_threshold as snr_threshold_test,
    test_recurrence as recurrence_test,
    test_physical_plausibility as plausibility_test,
    run_all_tests,
)


def test_snr_threshold_pass_and_fail():
    p = snr_threshold_test(10.0)
    assert p["passed"] is True
    p = snr_threshold_test(2.0)
    assert p["passed"] is False


def test_recurrence_pass_and_fail():
    p = recurrence_test(5)
    assert p["passed"] is True
    p = recurrence_test(1)
    assert p["passed"] is False


def test_physical_plausibility_rejects_extreme_depth():
    p = plausibility_test(depth_ppm=200000, period_days=3.0, snr=10.0)
    assert p["passed"] is False


def test_run_all_tests_returns_structure():
    t = np.linspace(0, 27, 2000)
    f = np.ones(2000)
    analysis = {"transit": {"period": 3.0, "t0": 0.0}, "snr": 10.0,
                "depth_ppm": 500, "n_transits": 5}
    result = run_all_tests(t, f, analysis)
    assert "tests" in result
    assert "passed_count" in result
    assert "total_tests" in result
    assert result["total_tests"] == 15
    assert isinstance(result["all_passed"], bool)
