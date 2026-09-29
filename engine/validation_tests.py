"""Validation Tests Module - Fixed.

All flux-dependent tests normalize flux to median=1.0 before computing depths.
"""
import numpy as np


def _fold(time, period, t0=0.0):
    return (time - t0) % period / period


def _normalize_flux(flux):
    f = np.asarray(flux, dtype=float)
    med = np.nanmedian(f)
    if med > 0:
        return f / med
    return f


def test_snr_threshold(snr, threshold=7.0):
    passed = snr >= threshold
    score = min(1.0, snr / (threshold * 2))
    status = "PASS" if passed else "FAIL"
    rel = ">=" if passed else "<"
    sig = "is" if passed else "is not"
    return {
        "test_name": "S/N Threshold",
        "passed": passed,
        "score": round(score, 3),
        "details": "S/N = %.2f %s threshold %.1f. %s: signal %s statistically significant." % (snr, rel, threshold, status, sig),
    }


def test_recurrence(n_transits, min_transits=3):
    passed = n_transits >= min_transits
    score = min(1.0, n_transits / (min_transits * 2))
    status = "PASS" if passed else "FAIL"
    rel = ">=" if passed else "<"
    conf = "confirmed" if passed else "unconfirmed"
    return {
        "test_name": "Recurrence Count",
        "passed": passed,
        "score": round(score, 3),
        "details": "%d recurring transits %s minimum %d. %s: periodicity is %s." % (n_transits, rel, min_transits, status, conf),
    }


def test_odd_even_depth(time, flux, period, t0=0.0):
    f = _normalize_flux(flux)
    phase = _fold(time, period, t0)
    in_transit = (phase < 0.05) | (phase > 0.95)
    transit_times = time[in_transit]
    transit_flux_vals = f[in_transit]

    if len(transit_times) < 4:
        return {
            "test_name": "Odd-Even Depth",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient transits for odd-even test (< 4).",
        }

    epochs = np.round((transit_times - t0) / period).astype(int)
    odd_mask = epochs % 2 != 0
    even_mask = epochs % 2 == 0

    if odd_mask.sum() < 2 or even_mask.sum() < 2:
        return {
            "test_name": "Odd-Even Depth",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient odd or even transits for comparison.",
        }

    odd_depth = 1.0 - np.median(transit_flux_vals[odd_mask])
    even_depth = 1.0 - np.median(transit_flux_vals[even_mask])

    odd_ppm = odd_depth * 1e6
    even_ppm = even_depth * 1e6

    mean_depth = (odd_depth + even_depth) / 2.0
    if mean_depth > 0:
        diff_ratio = abs(odd_depth - even_depth) / mean_depth
    else:
        diff_ratio = 0.0

    passed = diff_ratio < 0.3
    score = max(0, 1.0 - diff_ratio * 3)
    status = "PASS" if passed else "FAIL"
    desc = "consistent" if passed else "mismatched (EB indicator)"
    return {
        "test_name": "Odd-Even Depth",
        "passed": passed,
        "score": round(score, 3),
        "details": "Odd depth: %.0f ppm, Even depth: %.0f ppm. Difference: %.1f%%. %s: depths are %s." % (odd_ppm, even_ppm, diff_ratio * 100, status, desc),
    }


def test_secondary_eclipse(time, flux, period, t0=0.0, primary_depth=0.0):
    f = _normalize_flux(flux)
    phase = _fold(time, period, t0)
    sec_in = (phase > 0.45) & (phase < 0.55)
    sec_out = ((phase > 0.35) & (phase < 0.45)) | ((phase > 0.55) & (phase < 0.65))

    if sec_in.sum() < 3 or sec_out.sum() < 3:
        return {
            "test_name": "Secondary Eclipse",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient data at secondary eclipse phase to test.",
        }

    sec_depth = (np.median(f[sec_out]) - np.median(f[sec_in]))
    sec_ppm = sec_depth * 1e6

    if primary_depth > 0:
        sec_ratio = abs(sec_depth) / primary_depth
    else:
        sec_ratio = 0.0

    passed = sec_ratio < 0.1
    score = max(0, 1.0 - sec_ratio * 5)
    status = "PASS" if passed else "FAIL"
    desc = "absent" if passed else "detected (EB indicator)"
    return {
        "test_name": "Secondary Eclipse",
        "passed": passed,
        "score": round(score, 3),
        "details": "Secondary eclipse depth: %.0f ppm (%.1f%% of primary). %s: secondary eclipse %s." % (sec_ppm, sec_ratio * 100, status, desc),
    }


def test_vshape(time, flux, period, t0=0.0):
    f = _normalize_flux(flux)
    phase = _fold(time, period, t0)
    in_transit = (phase < 0.05) | (phase > 0.95)
    transit_phase = phase[in_transit]
    transit_flux_vals = f[in_transit]

    if len(transit_phase) < 10:
        return {
            "test_name": "V-Shape Test",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient in-transit points for shape analysis.",
        }

    sorted_idx = np.argsort(transit_phase)
    sorted_flux = transit_flux_vals[sorted_idx]

    n = len(sorted_flux)
    quarter = n // 4
    if quarter < 2:
        return {
            "test_name": "V-Shape Test",
            "passed": True,
            "score": 0.5,
            "details": "Too few points per quarter for shape analysis.",
        }

    edges = np.mean(sorted_flux[:quarter]) - np.mean(sorted_flux[quarter:3*quarter])
    bottom = np.mean(sorted_flux[quarter:3*quarter])
    baseline = np.mean(sorted_flux[3*quarter:])

    depth_range = baseline - bottom
    if depth_range > 0:
        u_ratio = edges / depth_range
    else:
        u_ratio = 0.5

    passed = u_ratio < 0.6
    score = max(0, 1.0 - u_ratio * 2)
    status = "PASS" if passed else "FAIL"
    shape = "U-shaped" if passed else "V-shaped"
    desc = "consistent with planet" if passed else "suggests grazing EB"
    return {
        "test_name": "V-Shape Test",
        "passed": passed,
        "score": round(score, 3),
        "details": "Edge-to-bottom ratio: %.3f (%s). %s: transit shape %s." % (u_ratio, shape, status, desc),
    }


def test_bootstrap_fap(time, flux, period, t0=0.0, snr=0.0, n_bootstrap=100):
    f = _normalize_flux(flux)
    rng = np.random.default_rng(42)
    fap_count = 0

    phase = _fold(time, period, t0)
    in_transit = (phase < 0.05) | (phase > 0.95)
    out_transit = ~in_transit

    if in_transit.sum() < 3 or out_transit.sum() < 10:
        return {
            "test_name": "Bootstrap FAP",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient data for bootstrap analysis.",
        }

    base_depth = np.median(f[out_transit]) - np.median(f[in_transit])

    for _ in range(n_bootstrap):
        shifted_flux = np.roll(f, rng.integers(len(f)))
        phase_s = _fold(time, period, t0)
        in_s = (phase_s < 0.05) | (phase_s > 0.95)
        out_s = ~in_s
        if in_s.sum() < 3:
            continue
        boot_depth = np.median(shifted_flux[out_s]) - np.median(shifted_flux[in_s])
        if boot_depth >= base_depth:
            fap_count += 1

    fap = fap_count / n_bootstrap
    passed = fap < 0.01
    score = max(0, 1.0 - fap * 10)
    status = "PASS" if passed else "FAIL"
    desc = "is likely real" if passed else "could be noise artifact"
    return {
        "test_name": "Bootstrap FAP",
        "passed": passed,
        "score": round(score, 3),
        "details": "False Alarm Probability: %.1f%% (%d/%d samples). %s: signal %s." % (fap * 100, fap_count, n_bootstrap, status, desc),
    }


def test_stellar_variability(time, flux, period, t0=0.0):
    f = _normalize_flux(flux)
    phase = _fold(time, period, t0)
    in_transit = (phase < 0.05) | (phase > 0.95)

    if in_transit.sum() < 6:
        return {
            "test_name": "Stellar Variability",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient transits to test stellar variability.",
        }

    transit_epochs = np.round((time[in_transit] - t0) / period).astype(int)
    unique_epochs = np.unique(transit_epochs)

    if len(unique_epochs) < 3:
        return {
            "test_name": "Stellar Variability",
            "passed": True,
            "score": 0.5,
            "details": "Too few unique transit epochs for variability test.",
        }

    out_phase = (phase > 0.15) & (phase < 0.85)
    out_flux = f[out_phase]

    epoch_depths = []
    for ep in unique_epochs:
        ep_mask = transit_epochs == ep
        ep_flux = f[in_transit][ep_mask]
        if len(ep_flux) > 0 and len(out_flux) > 0:
            depth = np.median(out_flux) - np.median(ep_flux)
            epoch_depths.append(depth)

    if len(epoch_depths) < 3:
        return {
            "test_name": "Stellar Variability",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient epoch data for variability analysis.",
        }

    depths_arr = np.array(epoch_depths)
    mean_d = np.mean(depths_arr)
    std_d = np.std(depths_arr)

    if mean_d > 0:
        cv = std_d / mean_d
    else:
        cv = 1.0

    passed = cv < 0.3
    score = max(0, 1.0 - cv * 3)
    status = "PASS" if passed else "FAIL"
    desc = "stable" if passed else "variable (stellar activity indicator)"
    return {
        "test_name": "Stellar Variability",
        "passed": passed,
        "score": round(score, 3),
        "details": "Depth CV across %d epochs: %.1f%%. Mean depth: %.0f ppm. %s: depths are %s." % (len(epoch_depths), cv * 100, mean_d * 1e6, status, desc),
    }


def test_physical_plausibility(depth_ppm, period_days, snr):
    issues = []
    if depth_ppm < 10:
        issues.append("depth %d ppm is extremely shallow" % depth_ppm)
    if depth_ppm > 100000:
        issues.append("depth %d ppm is deeper than any known planet transit" % depth_ppm)
    if period_days and period_days < 0.01:
        issues.append("period %.4f d is unphysically short" % period_days)
    if period_days and period_days > 1000:
        issues.append("period %.1f d is extremely long" % period_days)

    passed = len(issues) == 0
    score = max(0, 1.0 - len(issues) * 0.3)
    status = "PASS" if passed else "FAIL"
    iss = "; ".join(issues) if issues else "all parameters in realistic range"
    return {
        "test_name": "Physical Plausibility",
        "passed": passed,
        "score": round(score, 3),
        "details": "Depth: %d ppm, Period: %s d. %s: %s." % (depth_ppm, period_days, status, iss),
    }


def test_phase_curve(time, flux, period, t0=0.0, primary_depth=0.0):
    """Phase curve: secondary eclipse + reflection/thermal modulation.

    A real planet shows reflected-light and thermal phase modulation;
    an eclipsing binary shows deeper secondary eclipse and ellipsoidal variations.
    """
    f = _normalize_flux(flux)
    phase = _fold(time, period, t0)

    if len(f) < 50:
        return {
            "test_name": "Phase Curve Analysis",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient data for phase curve analysis.",
        }

    n_bins = 30
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_flux = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_flux[i] = np.median(f[mask])
            bin_count[i] = mask.sum()

    valid = bin_count > 2
    if valid.sum() < 10:
        return {
            "test_name": "Phase Curve Analysis",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient bin coverage for phase curve.",
        }

    bc = bin_centers[valid]
    bf = bin_flux[valid]

    secondary_phase = (bc > 0.4) & (bc < 0.6)
    out_phase = (bc < 0.3) | (bc > 0.7)

    if secondary_phase.sum() > 0 and out_phase.sum() > 0:
        secondary_flux = np.median(bf[secondary_phase])
        baseline_flux = np.median(bf[out_phase])
        secondary_depth_frac = baseline_flux - secondary_flux
        secondary_depth_ppm = secondary_depth_frac * 1e6
    else:
        secondary_depth_frac = 0.0
        secondary_depth_ppm = 0.0

    if primary_depth > 0:
        sec_ratio = abs(secondary_depth_frac) / primary_depth
    else:
        sec_ratio = 0.0

    amplitude = (np.max(bf) - np.min(bf)) * 1e6 if len(bf) > 0 else 0.0

    phase_slope = 0.0
    if len(bc) > 5:
        try:
            coeffs = np.polyfit(bc, bf, 2)
            phase_slope = abs(coeffs[0]) * 1e6
        except Exception:
            pass

    is_eb_secondary = sec_ratio > 0.15
    is_ellipsoidal = amplitude > primary_depth * 1e6 * 0.05 if primary_depth > 0 else False
    passed = not is_eb_secondary and not is_ellipsoidal

    score = 1.0
    if is_eb_secondary:
        score -= 0.5
    if is_ellipsoidal:
        score -= 0.3
    score = max(0, score)

    status = "PASS" if passed else "FAIL"
    details = (
        "Secondary eclipse depth: %.0f ppm (%.1f%% of primary). "
        "Phase curve amplitude: %.0f ppm. "
        "Phase modulation slope: %.3f ppm. "
        "%s: phase curve %s."
        % (secondary_depth_ppm, sec_ratio * 100, amplitude, phase_slope,
           status, "consistent with planet" if passed else "suggests binary")
    )

    return {
        "test_name": "Phase Curve Analysis",
        "passed": passed,
        "score": round(score, 3),
        "details": details,
        "secondary_depth_ppm": round(secondary_depth_ppm, 1),
        "secondary_ratio": round(sec_ratio, 4),
        "phase_amplitude_ppm": round(amplitude, 1),
    }


def test_chromaticity(time_flux_pairs, period, t0=0.0):
    """Chromaticity: transit depth should be achromatic for a real planet.

    If multi-band data is available, compare depths across bands.
    Chromatic depth variation indicates a blended eclipsing binary.
    """
    if not time_flux_pairs or len(time_flux_pairs) < 2:
        return {
            "test_name": "Chromaticity",
            "passed": True,
            "score": 0.5,
            "details": "No multi-band data available. Skipped.",
        }

    depths = []
    band_labels = []

    for band_data in time_flux_pairs:
        t_band = np.asarray(band_data["time"], dtype=float)
        f_band = np.asarray(band_data["flux"], dtype=float)
        f_band = _normalize_flux(f_band)
        phase = _fold(t_band, period, t0)

        in_transit = (phase < 0.05) | (phase > 0.95)
        out_transit = ~in_transit

        if in_transit.sum() > 2 and out_transit.sum() > 5:
            depth = np.median(f_band[out_transit]) - np.median(f_band[in_transit])
            depths.append(depth * 1e6)
            band_labels.append(band_data.get("label", f"band_{len(depths)}"))

    if len(depths) < 2:
        return {
            "test_name": "Chromaticity",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient multi-band depths computed.",
        }

    mean_depth = np.mean(depths)
    std_depth = np.std(depths)

    if mean_depth > 0:
        chromatic_fraction = std_depth / mean_depth
    else:
        chromatic_fraction = 0.0

    passed = chromatic_fraction < 0.15
    score = max(0, 1.0 - chromatic_fraction * 5)

    status = "PASS" if passed else "FAIL"
    depth_str = ", ".join(
        "%s: %.0f ppm" % (l, d) for l, d in zip(band_labels, depths)
    )
    details = (
        "Depth variation across %d bands: %.1f%% (chromatic fraction %.3f). "
        "%s: transit depth %s. [%s]"
        % (len(depths), chromatic_fraction * 100, chromatic_fraction,
           status, "is achromatic (planet-like)" if passed else "is chromatic (EB indicator)",
           depth_str)
    )

    return {
        "test_name": "Chromaticity",
        "passed": passed,
        "score": round(score, 3),
        "details": details,
        "depths_per_band": {l: round(d, 1) for l, d in zip(band_labels, depths)},
        "chromatic_fraction": round(chromatic_fraction, 4),
    }


def test_harmonic_alias(time, flux, period, t0=0.0, snr=0.0):
    """Harmonic alias detection: reject signals that are harmonics of a stronger period.

    Check 2P, 3P, P/2, P/3 to see if a longer/shorter period explains the signal better.
    """
    t = np.asarray(time, dtype=float)
    f = _normalize_flux(flux)

    if len(t) < 100 or not period or period <= 0:
        return {
            "test_name": "Harmonic Alias",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient data or no period to check harmonics.",
        }

    candidates = []
    for factor, label in [(0.5, "P/2"), (2.0, "2P"), (3.0, "3P"),
                           (1.0 / 3.0, "P/3"), (1.5, "1.5P")]:
        test_p = period * factor
        if test_p < 0.1 or test_p > (t[-1] - t[0]) / 3.0:
            continue

        phase = _fold(t, test_p, t0)
        in_t = (phase < 0.05) | (phase > 0.95)
        out_t = ~in_t

        if in_t.sum() < 3 or out_t.sum() < 10:
            continue

        base = np.median(f[out_t])
        transit = np.median(f[in_t])
        depth = base - transit
        err = np.std(f[in_t]) / np.sqrt(max(1, in_t.sum()))
        test_snr = depth / err if err > 0 else 0

        n_tr = len(np.unique(np.round((t - t0) / test_p).astype(int)))

        candidates.append({
            "period": test_p,
            "label": label,
            "snr": test_snr,
            "depth": depth,
            "n_transits": n_tr,
        })

    if not candidates:
        return {
            "test_name": "Harmonic Alias",
            "passed": True,
            "score": 0.5,
            "details": "No valid harmonic candidates to test.",
        }

    best_harmonic = max(candidates, key=lambda c: c["snr"])

    if best_harmonic["snr"] > snr * 1.5:
        passed = False
        score = 0.0
        details = (
            "FAIL: Harmonic %s (P=%.4f d) has S/N=%.1f vs detected S/N=%.1f. "
            "The %s period explains the signal better."
            % (best_harmonic["label"], best_harmonic["period"], best_harmonic["snr"],
               snr, best_harmonic["label"])
        )
    else:
        passed = True
        score = 1.0
        max_ratio = best_harmonic["snr"] / snr if snr > 0 else 0
        score = max(0.5, 1.0 - max_ratio * 0.3)
        details = (
            "PASS: No harmonic alias has significantly higher S/N. "
            "Best harmonic %s: P=%.4f d, S/N=%.1f (detected: P=%.4f d, S/N=%.1f)."
            % (best_harmonic["label"], best_harmonic["period"], best_harmonic["snr"],
               period, snr)
        )

    return {
        "test_name": "Harmonic Alias",
        "passed": passed,
        "score": round(score, 3),
        "details": details,
        "best_harmonic": {
            "label": best_harmonic["label"],
            "period": round(best_harmonic["period"], 4),
            "snr": round(best_harmonic["snr"], 1),
        },
    }


def test_transit_shape_metrics(time, flux, period, t0=0.0, depth_ppm=0.0):
    """Quantitative transit shape: ingress/egress slope, a/R*, contacts.

    Detailed shape analysis that goes beyond V/U distinction.
    """
    f = _normalize_flux(flux)
    phase = _fold(time, period, t0)

    in_transit = (phase < 0.05) | (phase > 0.95)
    transit_phase = phase[in_transit]
    transit_flux = f[in_transit]

    n_transit_pts = len(transit_phase)
    if n_transit_pts < 20:
        return {
            "test_name": "Transit Shape Metrics",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient in-transit points for detailed shape analysis.",
        }

    sort_idx = np.argsort(transit_phase)
    sorted_phase = transit_phase[sort_idx]
    sorted_flux = transit_flux[sort_idx]

    n = len(sorted_flux)
    q1 = n // 4
    q2 = n // 2
    q3 = 3 * n // 4

    ingress_slope = 0.0
    if q1 > 1:
        ingress_flux = sorted_flux[:q1]
        ingress_slope = abs(ingress_flux[-1] - ingress_flux[0]) / max(0.001, 1.0 / n)

    flat_bottom = sorted_flux[q1:q3] if q3 > q1 else np.array([sorted_flux[q2]])
    bottom_mean = np.mean(flat_bottom)
    bottom_std = np.std(flat_bottom)
    depth_frac = depth_ppm / 1e6 if depth_ppm > 0 else 0.0

    if depth_frac > 0:
        flatness = (np.max(flat_bottom) - np.min(flat_bottom)) / depth_frac
    else:
        flatness = 0.0

    contacts = {}
    threshold_low = 1.0 - depth_frac * 0.1 if depth_frac > 0 else 0.999
    threshold_mid = 1.0 - depth_frac * 0.5 if depth_frac > 0 else 0.995

    below_low = sorted_phase[sorted_flux < threshold_low]
    if len(below_low) > 0:
        contacts["t1"] = float(below_low[0])
        contacts["t4"] = float(below_low[-1])
        full_duration = (contacts["t4"] - contacts["t1"]) * period * 24

        below_mid = sorted_phase[sorted_flux < threshold_mid]
        if len(below_mid) > 0:
            contacts["t2"] = float(below_mid[0])
            contacts["t3"] = float(below_mid[-1])
            ingress_egress_duration = ((contacts["t2"] - contacts["t1"]) +
                                       (contacts["t3"] - contacts["t4"])) * period * 24
            contacts["ingress_hours"] = round(ingress_egress_duration / 2, 3)
        else:
            contacts["ingress_hours"] = 0
        contacts["full_duration_hours"] = round(full_duration, 3)
    else:
        contacts["full_duration_hours"] = 0
        contacts["ingress_hours"] = 0

    ars_estimate = 0.0
    if depth_frac > 0 and period:
        rprs = np.sqrt(depth_frac)
        if rprs > 0 and rprs < 1:
            ars_estimate = 2.0 * np.pi * (period * 24) / (depth_frac ** 0.5 * 13.4)

    ingress_fraction = 0.0
    if contacts.get("full_duration_hours", 0) > 0:
        ingress_fraction = contacts.get("ingress_hours", 0) / contacts["full_duration_hours"]
    contacts["ingress_fraction"] = round(ingress_fraction, 4)

    passed = True
    issues = []

    if flatness > 0.8 and depth_frac > 0.001:
        issues.append("V-shaped transit detected (flatness=%.3f)" % flatness)
        passed = False

    if n_transit_pts > 30 and bottom_std > 0.003:
        issues.append("Irregular transit bottom (std=%.4f)" % bottom_std)

    score = 1.0
    if not passed:
        score = 0.3
    elif flatness > 0.5:
        score = 0.7

    status = "PASS" if passed else "FAIL"
    details = (
        "%s: a/R* est=%.1f, full duration=%.2fh, ingress=%.2fh, "
        "ingress fraction=%.3f, flatness=%.3f. %s"
        % (status, ars_estimate, contacts.get("full_duration_hours", 0),
           contacts.get("ingress_hours", 0), ingress_fraction, flatness,
           "; ".join(issues) if issues else "all shape metrics nominal")
    )

    return {
        "test_name": "Transit Shape Metrics",
        "passed": passed,
        "score": round(score, 3),
        "details": details,
        "contacts": contacts,
        "ar_estimate": round(ars_estimate, 2),
        "flatness": round(flatness, 4),
        "ingress_fraction": round(ingress_fraction, 4),
    }


def test_window_function(period, t_start=None, t_end=None, known_cadences=None):
    """Window function: detect period matching TESS/Kepler cadences.

    Rejects periods that match instrumental artifacts like the TESS 27.4d roll,
    13.7d momentum dump cycle, or seasonal aliases.
    """
    if not period or period <= 0:
        return {
            "test_name": "Window Function",
            "passed": True,
            "score": 0.5,
            "details": "No period to test against window function.",
        }

    tess_roll = 27.4
    tess_momentum = 13.7
    tess_orbit = 13.7
    kepler_quarter = 89.75

    artifact_periods = [
        (tess_roll, "TESS roll (~27.4d)"),
        (tess_momentum, "TESS momentum dump (~13.7d)"),
        (tess_orbit, "TESS orbit (~13.7d)"),
        (kepler_quarter, "Kepler quarter (~89.75d)"),
        (1.0, "1-day alias"),
        (tess_roll / 2, "TESS half-roll (~13.7d)"),
        (tess_roll / 3, "TESS third-roll (~9.1d)"),
        (tess_roll * 2, "TESS double-roll (~54.8d)"),
    ]

    aliases_found = []
    for art_p, art_name in artifact_periods:
        ratio = period / art_p
        near_integer = abs(ratio - round(ratio)) < 0.02
        near_half = abs(ratio - (round(ratio * 2) / 2)) < 0.02

        if near_integer or near_half:
            aliases_found.append({
                "artifact": art_name,
                "ratio": round(ratio, 3),
                "type": "integer" if near_integer else "half-integer",
            })

    if t_start is not None and t_end is not None and t_end > t_start:
        baseline = t_end - t_start
        if baseline > 0:
            n_expected = baseline / period
            if n_expected > 0:
                cadence_alias = abs(n_expected - round(n_expected)) < 0.1
            else:
                cadence_alias = False
        else:
            cadence_alias = False
    else:
        cadence_alias = False

    passed = len(aliases_found) == 0 and not cadence_alias
    score = max(0, 1.0 - len(aliases_found) * 0.25 - (0.25 if cadence_alias else 0))

    status = "PASS" if passed else "FAIL"
    if aliases_found:
        alias_str = "; ".join(
            "%s (ratio=%.2f)" % (a["artifact"], a["ratio"]) for a in aliases_found
        )
        details = (
            "%s: Period %.4f d matches known instrumental cadence(s): %s. "
            "Signal may be an artifact."
            % (status, period, alias_str)
        )
    elif cadence_alias:
        details = (
            "%s: Period %.4f d is close to an integer multiple of observing baseline. "
            "May be a window function alias."
            % (status, period)
        )
    else:
        details = (
            "PASS: Period %.4f d does not match known TESS/Kepler cadences. "
            "Not a window function artifact." % period
        )

    return {
        "test_name": "Window Function",
        "passed": passed,
        "score": round(score, 3),
        "details": details,
        "aliases_found": aliases_found,
        "cadence_alias": cadence_alias,
    }


def test_stellar_density_consistency(period, depth_ppm, stellar_teff=None,
                                      stellar_logg=None, stellar_radius_rsun=None):
    """Stellar density consistency: compare transit-derived rho_star with spectroscopic.

    The transit shape encodes stellar density. If logg is known from spectroscopy,
    the two should agree. Mismatches indicate eclipsing binary false positive.
    """
    depth_frac = depth_ppm / 1e6

    if not period or period <= 0 or depth_frac <= 0:
        return {
            "test_name": "Stellar Density Check",
            "passed": True,
            "score": 0.5,
            "details": "Insufficient data for density comparison.",
        }

    transit_rho = (3.0 * np.pi) / ((period * 86400.0) ** 2 * 6.674e-11)
    transit_rho_cgs = transit_rho / 1000.0

    spectro_rho = None
    spectro_rho_cgs = None
    if stellar_logg is not None and stellar_teff is not None:
        g_cgs = 10 ** stellar_logg
        logg_rho_ratio = g_cgs * 1e-2 / (6.674e-8)
        teff_ratio = stellar_teff / 5778.0
        spectro_rho_cgs = logg_rho_ratio * (teff_ratio ** 3) * 0.001
        spectro_rho = spectro_rho_cgs * 1000.0

    if stellar_radius_rsun is not None:
        rs_m = stellar_radius_rsun * 6.957e8
        mass_est = 1.0
        if stellar_logg is not None:
            g_cgs = 10 ** stellar_logg
            mass_est = g_cgs * (rs_m * 6.684e-12) ** 2 / 1.327e20
        transit_density_mks = transit_rho
        mass_kg = mass_est * 1.989e30
        volume_m3 = (4.0 / 3.0) * np.pi * rs_m ** 3
        spectro_density_mks = mass_kg / volume_m3
        spectro_rho = spectro_density_mks
        spectro_rho_cgs = spectro_density_mks / 1000.0

    if spectro_rho_cgs is None:
        return {
            "test_name": "Stellar Density Check",
            "passed": True,
            "score": 0.5,
            "details": (
                "No spectroscopic density available (need logg or radius). "
                "Transit-derived rho_star=%.3f g/cm3. Cannot compare."
                % transit_rho_cgs
            ),
        }

    if spectro_rho_cgs > 0:
        density_ratio = transit_rho_cgs / spectro_rho_cgs
    else:
        density_ratio = 1.0

    density_diff_pct = abs(density_ratio - 1.0) * 100
    passed = density_diff_pct < 50
    score = max(0, 1.0 - density_diff_pct / 50.0)

    status = "PASS" if passed else "FAIL"
    details = (
        "%s: Transit rho=%.3f g/cm3, Spectroscopic rho=%.3f g/cm3. "
        "Ratio: %.2f (%.1f%% difference). %s"
        % (status, transit_rho_cgs, spectro_rho_cgs, density_ratio,
           density_diff_pct,
           "consistent (planet-like)" if passed else "inconsistent (EB indicator)")
    )

    return {
        "test_name": "Stellar Density Check",
        "passed": passed,
        "score": round(score, 3),
        "details": details,
        "transit_density_gcm3": round(transit_rho_cgs, 4),
        "spectro_density_gcm3": round(spectro_rho_cgs, 4),
        "density_ratio": round(density_ratio, 3),
    }


def test_catalog_crossmatch(known_planets, detected_period, tolerance=0.05):
    matches = []
    for p in known_planets:
        known_period = p.get("pl_orbper")
        if known_period:
            try:
                kp = float(known_period)
                if abs(kp - detected_period) / kp < tolerance:
                    matches.append(p)
            except (ValueError, TypeError):
                pass

    passed = len(matches) == 0
    score = 1.0 if passed else 0.0
    status = "PASS" if passed else "FAIL"
    if matches:
        names = ", ".join(m.get("pl_name", "unknown") for m in matches)
        details = "%s: MATCH FOUND with known planets: %s. Signal is NOT new." % (status, names)
    else:
        details = "%s: No matching known planets within %.0f%% period tolerance. Signal could be new." % (status, tolerance * 100)
    return {
        "test_name": "Catalog Cross-match",
        "passed": passed,
        "score": score,
        "details": details,
        "matches": matches,
    }


def run_all_tests(time_arr, flux_arr, analysis, known_planets=None,
                   multi_band_data=None, stellar_info=None):
    transit = analysis.get("transit", {})
    period = transit.get("period")
    t0 = transit.get("t0", 0.0)
    snr = analysis.get("snr", 0.0)
    depth_ppm = analysis.get("depth_ppm", 0)
    n_transits = transit.get("n_transits", 0)
    depth_frac = depth_ppm / 1e6

    results = []
    results.append(test_snr_threshold(snr))
    results.append(test_recurrence(n_transits))
    results.append(test_physical_plausibility(depth_ppm, period, snr))

    if period and len(time_arr) > 10:
        results.append(test_stellar_variability(time_arr, flux_arr, period, t0))
        results.append(test_odd_even_depth(time_arr, flux_arr, period, t0))
        results.append(test_secondary_eclipse(time_arr, flux_arr, period, t0, depth_frac))
        results.append(test_vshape(time_arr, flux_arr, period, t0))
        results.append(test_bootstrap_fap(time_arr, flux_arr, period, t0, snr, n_bootstrap=80))
        results.append(test_phase_curve(time_arr, flux_arr, period, t0, depth_frac))
        results.append(test_harmonic_alias(time_arr, flux_arr, period, t0, snr))
        results.append(test_transit_shape_metrics(time_arr, flux_arr, period, t0, depth_ppm))
        results.append(test_window_function(period, time_arr[0] if len(time_arr) > 0 else None,
                                            time_arr[-1] if len(time_arr) > 0 else None))
    else:
        for name in ["Stellar Variability", "Odd-Even Depth", "Secondary Eclipse",
                      "V-Shape Test", "Bootstrap FAP", "Phase Curve Analysis",
                      "Harmonic Alias", "Transit Shape Metrics", "Window Function"]:
            results.append({"test_name": name, "passed": True, "score": 0.5, "details": "Skipped: insufficient data."})

    if multi_band_data and len(multi_band_data) >= 2 and period:
        results.append(test_chromaticity(multi_band_data, period, t0))
    else:
        results.append({"test_name": "Chromaticity", "passed": True, "score": 0.5,
                         "details": "No multi-band data available. Skipped."})

    if stellar_info and period and depth_ppm:
        teff = stellar_info.get("teff") or stellar_info.get("pl_star_teff")
        logg = stellar_info.get("logg") or stellar_info.get("pl_starlogg")
        rs = stellar_info.get("radius_rsun") or stellar_info.get("pl_star_radius")
        results.append(test_stellar_density_consistency(period, depth_ppm, teff, logg, rs))
    else:
        results.append({"test_name": "Stellar Density Check", "passed": True, "score": 0.5,
                         "details": "No stellar parameters available. Skipped."})

    if known_planets is not None and period:
        results.append(test_catalog_crossmatch(known_planets, period))
    else:
        results.append({"test_name": "Catalog Cross-match", "passed": True, "score": 0.5, "details": "Skipped: no catalog data available."})

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_score = float(np.mean([r["score"] for r in results]))

    return {
        "tests": results,
        "passed_count": passed_count,
        "total_tests": total,
        "all_passed": passed_count == total,
        "average_score": round(avg_score, 3),
    }
