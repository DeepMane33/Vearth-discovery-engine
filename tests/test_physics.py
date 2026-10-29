"""Basic tests for the physics module."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.physics import (
    planet_radius_from_depth,
    semi_major_axis,
    equilibrium_temperature,
    classify_planet,
    mass_from_radius,
    bulk_density,
    compute_all_physics,
)


def test_planet_radius_from_depth():
    rp = planet_radius_from_depth(0.01, 1.0)  # 1% depth, 1 Rsun -> ~11.2 Re (Jupiter)
    assert rp > 0
    assert abs(rp - 11.2) < 0.5


def test_semi_major_axis_positive():
    a = semi_major_axis(365.0, 1.0)
    assert a > 0
    assert abs(a - 1.0) < 0.1


def test_equilibrium_temperature_positive():
    teq = equilibrium_temperature(5778, 1.0, 1.0)
    assert teq > 0


def test_classify_planet_bounds():
    assert classify_planet(1.0) == "rocky (Earth-size)"
    assert classify_planet(1.5) == "super-Earth"
    assert classify_planet(3.0) == "sub-Neptune"
    assert classify_planet(11.5) == "sub-Jupiter"
    assert classify_planet(20.0) == "Jupiter-size or larger"


def test_mass_from_radius_positive():
    m = mass_from_radius(1.0)
    assert m > 0


def test_bulk_density_positive():
    rho = bulk_density(1.0, 1.0)
    assert rho > 0


def test_compute_all_physics_returns_dict():
    analysis = {"depth_ppm": 100, "period_days": 3.0}
    stellar = {"pl_star_teff": 5778, "pl_starlogg": 4.44, "pl_star_feh": 0.0,
               "pl_star_radius": 1.0, "pl_star_mass": 1.0}
    result = compute_all_physics(analysis, stellar)
    assert isinstance(result, dict)
    assert "planet_radius_rearth" in result
    assert "equilibrium_temp_k" in result
    assert "in_habitable_zone" in result
