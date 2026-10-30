"""Physics Calculations Module.

Implements the real physics behind exoplanet transit detection:
  - Planet radius from transit depth
  - Orbital semi-major axis from Kepler's third law
  - Equilibrium temperature
  - Habitable zone boundaries
  - Transit duration
  - Impact parameter
  - Stellar density from transit
"""
import numpy as np

# Physical constants (SI)
G = 6.67430e-11
SIGMA_SB = 5.670374419e-8
AU = 1.496e11
RSUN = 6.957e8
MSUN = 1.989e30
RJUP = 7.149e7
REARTH = 6.371e6
MEARTH = 5.972e24
TEARTH = 288.0
LUMINOSITY_SUN = 3.828e26


def planet_radius_from_depth(depth, stellar_radius_rsun):
    """Rp = sqrt(depth) * Rsun.
    Args:
        depth: transit depth as a fraction (not ppm)
        stellar_radius_rsun: stellar radius in solar radii
    Returns:
        Planet radius in Earth radii
    """
    rprs = np.sqrt(max(0, depth))
    rp_rsun = rprs * stellar_radius_rsun
    return rp_rsun * RSUN / REARTH


def semi_major_axis(period_days, stellar_mass_msun):
    """Kepler's third law: a = (P^2 * G * Ms / (4 pi^2))^(1/3)
    Args:
        period_days: orbital period in days
        stellar_mass_msun: stellar mass in solar masses
    Returns:
        Semi-major axis in AU
    """
    P = period_days * 86400.0
    Ms = stellar_mass_msun * MSUN
    a_m = (P**2 * G * Ms / (4.0 * np.pi**2))**(1.0/3.0)
    return a_m / AU


def equilibrium_temperature(stellar_teff, stellar_radius_rsun, semi_major_axis_au):
    """Teq = Teff * sqrt(Rstar / (2 * a))
    Args:
        stellar_teff: stellar effective temperature in Kelvin
        stellar_radius_rsun: stellar radius in solar radii
        semi_major_axis_au: semi-major axis in AU
    Returns:
        Equilibrium temperature in Kelvin (assuming albedo=0)
    """
    rs = stellar_radius_rsun * RSUN
    a = semi_major_axis_au * AU
    return stellar_teff * np.sqrt(rs / (2.0 * a))


def habitable_zone_inner(stellar_luminosity_lsun):
    """Inner edge of conservative habitable zone (Kopparapu et al. 2013).
    Returns distance in AU."""
    L = stellar_luminosity_lsun
    return np.sqrt(L / 1.107)


def habitable_zone_outer(stellar_luminosity_lsun):
    """Outer edge of conservative habitable zone (Kopparapu et al. 2013).
    Returns distance in AU."""
    L = stellar_luminosity_lsun
    return np.sqrt(L / 0.356)


def is_in_habitable_zone(semi_major_axis_au, stellar_luminosity_lsun):
    """Check if a planet's orbit falls within the habitable zone."""
    inner = habitable_zone_inner(stellar_luminosity_lsun)
    outer = habitable_zone_outer(stellar_luminosity_lsun)
    return inner <= semi_major_axis_au <= outer


def transit_duration(period_days, stellar_radius_rsun, semi_major_axis_au,
                     impact_param=0.0):
    """Transit duration T_dur = (P * Rsun) / (pi * a * sin(i))
    For grazing transits, T_dur_shorter = T_dur * sqrt(1 - b^2)
    Args:
        period_days: orbital period in days
        stellar_radius_rsun: stellar radius in solar radii
        semi_major_axis_au: semi-major axis in AU
        impact_param: impact parameter b (0=central, 1=grazing)
    Returns:
        Transit duration in hours
    """
    rs = stellar_radius_rsun * RSUN
    a = semi_major_axis_au * AU
    P = period_days * 86400.0
    T_dur_s = (P * rs) / (np.pi * a) * np.sqrt(1.0 - impact_param**2)
    return T_dur_s / 3600.0


def impact_parameter_from_geometry(stellar_radius_rsun, semi_major_axis_au,
                                    inclination_deg):
    """b = (a * cos(i)) / Rsun
    Args:
        stellar_radius_rsun: stellar radius in solar radii
        semi_major_axis_au: semi-major axis in AU
        inclination_deg: orbital inclination in degrees
    Returns:
        Dimensionless impact parameter
    """
    rs = stellar_radius_rsun * RSUN
    a = semi_major_axis_au * AU
    inc_rad = np.radians(inclination_deg)
    return (a * np.cos(inc_rad)) / rs


def stellar_density_from_transit(period_days, stellar_radius_rsun,
                                  semi_major_axis_au):
    """Estimate stellar density from transit geometry.
    rho_star = (3 * pi) / (G * P^2) * (a / Rs)^3
    Args:
        period_days: orbital period in days
        stellar_radius_rsun: stellar radius in solar radii
        semi_major_axis_au: semi-major axis in AU
    Returns:
        Stellar density in kg/m^3
    """
    P = period_days * 86400.0
    rs = stellar_radius_rsun * RSUN
    a = semi_major_axis_au * AU
    return (3.0 * np.pi) / (G * P**2) * (a / rs)**3


def transit_depth_from_radius(rp_rearth, rs_rsun):
    """Calculate expected transit depth from planet and star radii.
    depth = (Rp/Rs)^2
    Args:
        rp_rearth: planet radius in Earth radii
        rs_rsun: stellar radius in solar radii
    Returns:
        Transit depth as a fraction
    """
    rp_m = rp_rearth * REARTH
    rs_m = rs_rsun * RSUN
    return (rp_m / rs_m)**2


def classify_planet(rp_rearth):
    """Classify planet type by radius (rough categories)."""
    if rp_rearth < 1.25:
        return "rocky (Earth-size)"
    elif rp_rearth < 2.0:
        return "super-Earth"
    elif rp_rearth < 4.0:
        return "sub-Neptune"
    elif rp_rearth < 10.0:
        return "Neptune-size"
    elif rp_rearth < 15.0:
        return "sub-Jupiter"
    else:
        return "Jupiter-size or larger"


def mass_from_radius(rp_rearth):
    """Estimate planet mass from radius using Chen & Kipping (2017) relation.

    For rocky planets (Rp < 1.22 Re): M = 0.9 * (Rp/Re)^3.95  (Earth-like)
    For sub-Neptunes (1.22 < Rp < 4 Re): M = 1.7 * (Rp/Re)^1.3   (volatile-rich)
    For gas giants (Rp > 4 Re): M = (Rp/Rj)^2.06  (giant branch)
    Returns mass in Earth masses.
    """
    if rp_rearth <= 0:
        return 0.0
    if rp_rearth < 1.22:
        return 0.9 * rp_rearth ** 3.95
    elif rp_rearth < 4.0:
        return 1.7 * rp_rearth ** 1.3
    else:
        rp_rj = rp_rearth * REARTH / RJUP
        return rp_rj ** 2.06 * 317.8


def bulk_density(rp_rearth, mass_earth=None):
    """Calculate bulk density from radius and estimated mass.

    rho = M / V = (M * MEARTH) / (4/3 * pi * (Rp * REARTH)^3)
    Returns density in g/cm3.
    """
    if rp_rearth <= 0:
        return 0.0
    if mass_earth is None:
        mass_earth = mass_from_radius(rp_rearth)
    mass_kg = mass_earth * MEARTH if mass_earth else MEARTH
    radius_m = rp_rearth * REARTH
    volume = (4.0 / 3.0) * np.pi * radius_m ** 3
    if volume <= 0:
        return 0.0
    density_kgm3 = mass_kg / volume
    return density_kgm3 / 1000.0


def eccentricity_from_transit(period_days, stellar_radius_rsun, stellar_mass_msun,
                               transit_duration_hours):
    """Estimate minimum eccentricity from transit duration vs. expected circular duration.

    Watson & Welsh (2011): T_dur_circ = P/pi * Rs/(a*sin(i)) for circular orbit.
    If observed T_dur differs from circular T_dur, eccentricity is non-zero.
    Returns (e_min, e_max) bounds.
    """
    a_au = semi_major_axis(period_days, stellar_mass_msun)
    t_dur_circ = transit_duration(period_days, stellar_radius_rsun, a_au, impact_param=0.0)

    if t_dur_circ <= 0 or transit_duration_hours <= 0:
        return (0.0, 0.3)

    ratio = transit_duration_hours / t_dur_circ
    if ratio > 1.0:
        ratio = min(ratio, 1.5)

    e_max = min(0.9, 1.0 - ratio ** 2) if ratio < 1.0 else 0.0
    e_min = 0.0

    if ratio < 0.8:
        e_min = max(0.0, 1.0 - ratio ** 2 - 0.1)

    return (round(e_min, 3), round(min(e_max, 0.9), 3))


def atmospheric_tsm(rp_rearth, stellar_radius_rsun, stellar_teff, mass_earth=None,
                     stellar_distance_pc=None):
    """Transmission Spectroscopy Metric (Kempton+ 2018).

    TSM = (Rp * T_eq * scale_height) / (Mp * Rp_star^2)
    Higher TSM = easier to characterize atmosphere with JWST.

    Returns TSM value (dimensionless) and JWST tier.
    """
    if rp_rearth <= 0 or stellar_teff <= 0:
        return {"tsm": 0, "tier": "not characterizable", "scale_height_m": 0}

    if mass_earth is None:
        mass_earth = mass_from_radius(rp_rearth)

    rp_m = rp_rearth * REARTH
    mp_kg = mass_earth * MEARTH if mass_earth else MEARTH
    rs_m = stellar_radius_rsun * RSUN

    a_au_est = 0.05
    teq = stellar_teff * np.sqrt(rs_m / (2.0 * a_au_est * AU))
    teq = min(teq, 3000)

    mu_h2 = 2.3 * 1.66e-27
    k_b = 1.381e-23
    g_planet = 6.674e-11 * mp_kg / (rp_m ** 2) if mp_kg > 0 and rp_m > 0 else 9.8
    scale_height = k_b * teq / (mu_h2 * g_planet) if g_planet > 0 else 800

    rs_rsun = stellar_radius_rsun
    tsm = (rp_rearth * (teq / 1200.0) * (scale_height / 800.0)) / (
        (mass_earth if mass_earth else 1.0) * rs_rsun ** 2
    ) * 1.27

    if tsm > 100:
        tier = "Tier 1 (excellent)"
    elif tsm > 30:
        tier = "Tier 2 (good)"
    elif tsm > 10:
        tier = "Tier 3 (moderate)"
    else:
        tier = "Tier 4 (difficult)"

    return {
        "tsm": round(tsm, 2),
        "tier": tier,
        "scale_height_m": round(scale_height, 1),
        "teq_for_tsm": round(teq, 1),
    }


def atmospheric_esm(rp_rearth, stellar_teff, stellar_radius_rsun):
    """Emission Spectroscopy Metric (Kempton+ 2018).

    ESM = (Tp / 1200) * (Rp/Rs)^2
    Higher ESM = easier thermal emission detection.
    """
    if rp_rearth <= 0 or stellar_teff <= 0 or stellar_radius_rsun <= 0:
        return {"esm": 0, "tier": "not characterizable"}

    rp_rj = rp_rearth * REARTH / RJUP
    rs_rj = stellar_radius_rsun * RSUN / RJUP
    depth = (rp_rj / rs_rj) ** 2

    a_au_est = 0.05
    rs_m = stellar_radius_rsun * RSUN
    tp = stellar_teff * np.sqrt(rs_m / (2.0 * a_au_est * AU)) * 0.85
    tp = min(tp, 3000)

    esm = (tp / 1200.0) * depth

    if esm > 10:
        tier = "Tier 1 (excellent)"
    elif esm > 3:
        tier = "Tier 2 (good)"
    elif esm > 1:
        tier = "Tier 3 (moderate)"
    else:
        tier = "Tier 4 (difficult)"

    return {
        "esm": round(esm, 3),
        "tier": tier,
        "tp_estimated": round(tp, 1),
        "thermal_depth_ppm": round(depth * 1e6, 1),
    }


def hill_stability(mass_planet_earth, mass_star_msun, semi_major_axis_au,
                    mass_planet2_earth=None):
    """Hill stability criterion: can the orbit remain stable?

    For single planet: stable if a >> R_Hill.
    For two planets: mutual Hill radius criterion.

    Returns dict with stability assessment.
    """
    mp1_kg = mass_planet_earth * MEARTH if mass_planet_earth else 0
    ms_kg = mass_star_msun * MSUN if mass_star_msun else MSUN
    a_m = semi_major_axis_au * AU

    r_hill = a_m * (mp1_kg / (3.0 * ms_kg)) ** (1.0 / 3.0) if mp1_kg > 0 and ms_kg > 0 else 0
    r_hill_rearth = r_hill / REARTH

    stable_single = True
    if mp1_kg > 0 and ms_kg > 0:
        hill_ratio = a_m / r_hill if r_hill > 0 else float("inf")
        stable_single = hill_ratio > 10

    mutual_stable = True
    min_separation_au = None
    if mass_planet2_earth is not None and mass_planet2_earth > 0:
        mp2_kg = mass_planet2_earth * MEARTH
        a2_m = a_m * 1.5
        r_hill_mutual = ((a_m + a2_m) / 2.0) * ((mp1_kg + mp2_kg) / (3.0 * ms_kg)) ** (1.0 / 3.0)
        separation = abs(a2_m - a_m)
        n_mutual_hill_radii = separation / r_hill_mutual if r_hill_mutual > 0 else float("inf")
        mutual_stable = n_mutual_hill_radii > 2 * np.sqrt(3)
        min_separation_au = separation / AU

    return {
        "hill_radius_rearth": round(r_hill_rearth, 2),
        "hill_radius_au": round(r_hill / AU, 6),
        "stable_single": stable_single,
        "mutual_stable": mutual_stable,
        "min_separation_au": round(min_separation_au, 4) if min_separation_au else None,
        "classification": "stable" if stable_single and mutual_stable else "potentially unstable",
    }


def transit_probability(stellar_radius_rsun, stellar_mass_msun, period_days,
                         eccentricity=0.0):
    """Geometric + tidal transit probability.

    Geometric: P(transit) = Rs / a
    Tidal enhancement (Barnes 2007): enhanced by (1-e^2) / (1-e*sin(omega))
    Using omega=0 for maximum tidal boost.
    """
    a_au = semi_major_axis(period_days, stellar_mass_msun)
    rs_m = stellar_radius_rsun * RSUN
    a_m = a_au * AU

    p_geom = rs_m / a_m if a_m > 0 else 0

    p_tidal = p_geom * (1 + 15 * (1 - eccentricity) ** 2) if eccentricity < 1 else p_geom

    return {
        "geometric_probability": round(min(p_geom, 1.0), 4),
        "tidal_probability": round(min(p_tidal, 1.0), 4),
        "in_percent": round(min(p_geom * 100, 100), 2),
    }


def tidal_locking_timescale(rp_rearth, mass_earth, period_days, stellar_mass_msun,
                             stellar_radius_rsun, semi_major_axis_au):
    """Estimate time to tidal locking.

    For close-in planets, tidal locking happens quickly.
    T_lock ~ (a^6 * Mp * Q) / (Ms^2 * Rs^5)
    Uses Q=100 for rocky, Q=100 for gas giants.
    """
    a_m = semi_major_axis_au * AU
    mp_kg = mass_earth * MEARTH if mass_earth else MEARTH
    ms_kg = stellar_mass_msun * MSUN if stellar_mass_msun else MSUN
    rs_m = stellar_radius_rsun * RSUN

    if ms_kg <= 0 or rs_m <= 0 or a_m <= 0:
        return {"locked": False, "timescale_gyr": None, "q_factor": None}

    Q_planet = 100
    t_lock_s = (a_m ** 6 * mp_kg * Q_planet) / (ms_kg ** 2 * rs_m ** 5)
    t_lock_gyr = t_lock_s / (3.156e16)

    is_locked = t_lock_gyr < 5.0

    return {
        "locked": is_locked,
        "timescale_gyr": round(t_lock_gyr, 3) if t_lock_gyr < 1e6 else ">1e6 Gyr",
        "q_factor": Q_planet,
        "assessment": "likely tidally locked" if is_locked else "not yet tidally locked",
    }


def compute_all_physics(analysis, stellar_params=None):
    """Given analysis results and stellar parameters, compute all physics.
    Returns a dict of computed values."""
    depth_frac = analysis.get("depth_ppm", 0) / 1e6
    period = analysis.get("period_days")

    result = {
        "depth_fraction": depth_frac,
        "rprs": np.sqrt(depth_frac) if depth_frac > 0 else 0,
    }

    if stellar_params:
        teff = stellar_params.get("pl_star_teff") or 5778
        logg = stellar_params.get("pl_starlogg") or 4.44
        feh = stellar_params.get("pl_star_feh") or 0.0
        rs = stellar_params.get("pl_star_radius") or 1.0
        ms = stellar_params.get("pl_star_mass") or 1.0

        result["stellar_teff"] = teff
        result["stellarlogg"] = logg
        result["stellar_feh"] = feh
        result["stellar_radius_rsun"] = rs
        result["stellar_mass_msun"] = ms
        # Flag whether stellar params came from a catalog or were assumed
        has_real = (
            stellar_params.get("pl_star_teff") is not None
            and stellar_params.get("pl_star_radius") is not None
        )
        result["stellar_params_source"] = (
            "measured" if has_real else "assumed_solar"
        )
    else:
        teff = 5778
        rs = 1.0
        ms = 1.0
        result["stellar_teff"] = teff
        result["stellar_radius_rsun"] = rs
        result["stellar_mass_msun"] = ms
        result["stellar_params_source"] = "assumed_solar"

    if period and period > 0:
        rp_earth = planet_radius_from_depth(depth_frac, rs)
        a_au = semi_major_axis(period, ms)
        teq = equilibrium_temperature(teff, rs, a_au)
        luminosity = ms ** 4 if ms else 1.0
        hz_inner = habitable_zone_inner(luminosity)
        hz_outer = habitable_zone_outer(luminosity)
        in_hz = is_in_habitable_zone(a_au, luminosity)
        t_dur = transit_duration(period, rs, a_au)
        rho = stellar_density_from_transit(period, rs, a_au)
        classification = classify_planet(rp_earth)

        est_mass = mass_from_radius(rp_earth)
        est_density = bulk_density(rp_earth, est_mass)
        e_bounds = eccentricity_from_transit(period, rs, ms, t_dur)
        tsm_data = atmospheric_tsm(rp_earth, rs, teff, est_mass)
        esm_data = atmospheric_esm(rp_earth, teff, rs)
        hill_data = hill_stability(est_mass, ms, a_au)
        prob_data = transit_probability(rs, ms, period, eccentricity=(e_bounds[0] + e_bounds[1]) / 2)
        tidal_data = tidal_locking_timescale(rp_earth, est_mass, period, ms, rs, a_au)

        result.update({
            "planet_radius_rearth": round(rp_earth, 3),
            "planet_radius_rjup": round(rp_earth * REARTH / RJUP, 4),
            "semi_major_axis_au": round(a_au, 4),
            "equilibrium_temp_k": round(teq, 1),
            "habitable_zone_inner_au": round(hz_inner, 3),
            "habitable_zone_outer_au": round(hz_outer, 3),
            "in_habitable_zone": in_hz,
            "transit_duration_hours": round(t_dur, 2),
            "stellar_density_kgm3": round(rho, 2),
            "planet_classification": classification,
            "estimated_mass_earth": round(est_mass, 3),
            "estimated_mass_jup": round(est_mass * MEARTH / (317.8 * MEARTH), 4),
            "bulk_density_gcm3": round(est_density, 3),
            "eccentricity_min": e_bounds[0],
            "eccentricity_max": e_bounds[1],
            "tsm": tsm_data["tsm"],
            "tsm_tier": tsm_data["tier"],
            "tsm_scale_height_m": tsm_data["scale_height_m"],
            "esm": esm_data["esm"],
            "esm_tier": esm_data["tier"],
            "esm_tp_estimated": esm_data["tp_estimated"],
            "hill_stability": hill_data,
            "transit_probability": prob_data,
            "tidal_locking": tidal_data,
        })
    else:
        result["planet_radius_rearth"] = 0
        result["estimated_mass_earth"] = 0
        result["bulk_density_gcm3"] = 0
        result["planet_classification"] = "unknown (no period)"
        result["tsm"] = 0
        result["tsm_tier"] = "unknown"
        result["esm"] = 0
        result["esm_tier"] = "unknown"

    return result
