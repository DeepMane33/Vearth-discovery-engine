"""Publication-Quality Plot Generation.

Generates figures suitable for inclusion in a research paper:
  - Phase-folded light curve with transit model overlay
  - BLS periodogram
  - Odd-even depth comparison
  - Residuals analysis
"""
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _fold(time, period, t0=0.0):
    return (time - t0) % period / period


def _fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def plot_phase_folded(time, flux, period, t0=0.0, depth_ppm=0, snr=0):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    phase = _fold(time, period, t0)
    sort_idx = np.argsort(phase)
    phase_s = phase[sort_idx]
    flux_s = np.asarray(flux, float)[sort_idx]
    flux_s = flux_s / np.median(flux_s)

    ax.scatter(phase_s, flux_s, s=1, color="#475569", alpha=0.4, rasterized=True)

    in_transit = (phase_s < 0.05) | (phase_s > 0.95)
    if in_transit.sum() > 0:
        ax.scatter(phase_s[in_transit], flux_s[in_transit],
                   s=8, color="#dc2626", alpha=0.7, label="in-transit", zorder=5)

    ax.axhline(1.0, color="#94a3b8", ls="--", lw=0.8, alpha=0.5)
    if depth_ppm > 0:
        depth_frac = depth_ppm / 1e6
        ax.axhline(1.0 - depth_frac, color="#dc2626", ls=":", lw=1.0, alpha=0.6,
                   label=f"depth = {depth_ppm} ppm")

    ax.set_xlabel("Orbital Phase", fontsize=10)
    ax.set_ylabel("Relative Flux", fontsize=10)
    ax.set_title(f"Phase-Folded Light Curve  (P = {period:.4f} d, S/N = {snr:.1f})", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(-0.05, 1.05)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_bls_periodogram(periods, powers, best_period=None, best_power=None):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
    ax.plot(periods, powers, color="#1e40af", lw=0.8)
    ax.fill_between(periods, 0, powers, alpha=0.15, color="#1e40af")

    if best_period and best_power:
        ax.axvline(best_period, color="#dc2626", ls="--", lw=1.0, alpha=0.8,
                   label=f"Best: P = {best_period:.4f} d")
        ax.plot(best_period, best_power, "o", color="#dc2626", ms=8, zorder=5)

    ax.set_xlabel("Period (days)", fontsize=10)
    ax.set_ylabel("BLS Power", fontsize=10)
    ax.set_title("Box Least Squares Periodogram", fontsize=11)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_odd_even(time, flux, period, t0=0.0):
    phase = _fold(time, period, t0)
    in_transit = (phase < 0.05) | (phase > 0.95)
    transit_times = time[in_transit]
    transit_flux_vals = flux[in_transit]

    if len(transit_times) < 4:
        return None

    epochs = np.round((transit_times - t0) / period).astype(int)
    odd_mask = epochs % 2 != 0
    even_mask = epochs % 2 == 0

    if odd_mask.sum() < 2 or even_mask.sum() < 2:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), dpi=120)

    odd_phase = phase[in_transit][odd_mask]
    even_phase = phase[in_transit][even_mask]
    odd_flux = transit_flux_vals[odd_mask]
    even_flux = transit_flux_vals[even_mask]

    ax1.scatter(odd_phase, odd_flux, s=10, color="#2563eb", alpha=0.6, label="Odd")
    ax1.scatter(even_phase, even_flux, s=10, color="#dc2626", alpha=0.6, label="Even")
    ax1.set_xlabel("Phase", fontsize=9)
    ax1.set_ylabel("Relative Flux", fontsize=9)
    ax1.set_title("Odd-Even Transits", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=8)

    odd_depth = 1.0 - np.median(odd_flux)
    even_depth = 1.0 - np.median(even_flux)
    bars = ax2.bar(["Odd", "Even"],
                   [odd_depth * 1e6, even_depth * 1e6],
                   color=["#2563eb", "#dc2626"], alpha=0.7, width=0.5)
    ax2.set_ylabel("Depth (ppm)", fontsize=9)
    ax2.set_title("Depth Comparison", fontsize=10)
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_residuals(time, flux, period, t0=0.0):
    phase = _fold(time, period, t0)
    flux_norm = np.asarray(flux, float) / np.median(flux)

    in_transit = (phase < 0.05) | (phase > 0.95)
    out_transit = ~in_transit

    if out_transit.sum() < 10:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), dpi=120,
                                    height_ratios=[2, 1], sharex=True)

    sorted_idx = np.argsort(phase)
    phase_s = phase[sorted_idx]
    flux_s = flux_norm[sorted_idx]

    ax1.scatter(phase_s, flux_s, s=1, color="#475569", alpha=0.4, rasterized=True)
    ax1.axhline(1.0, color="#94a3b8", ls="--", lw=0.8, alpha=0.5)
    ax1.set_ylabel("Relative Flux", fontsize=9)
    ax1.set_title("Light Curve with Residuals", fontsize=10)
    ax1.tick_params(labelsize=8)

    residuals = flux_norm - 1.0
    ax2.scatter(phase_s, residuals, s=1, color="#475569", alpha=0.3, rasterized=True)
    ax2.axhline(0, color="#94a3b8", ls="--", lw=0.8, alpha=0.5)
    ax2.set_xlabel("Orbital Phase", fontsize=9)
    ax2.set_ylabel("Residuals", fontsize=9)
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_multi_planet(all_planets):
    """Generate a comparison chart of all detected planets."""
    n = len(all_planets)
    if n < 2:
        return None

    fig, axes = plt.subplots(2, min(n, 3), figsize=(5 * min(n, 3), 6), dpi=120)
    if min(n, 3) == 1:
        axes = axes.reshape(2, 1)

    colors = ["#1e40af", "#dc2626", "#059669", "#d97706", "#7c3aed"]
    params = ["depth_ppm", "period_days", "snr"]
    labels = ["Transit Depth (ppm)", "Orbital Period (d)", "Signal-to-Noise"]
    planets_shown = min(n, 3)

    for col in range(planets_shown):
        pm = all_planets[col]
        ax = axes[0, col]
        analysis = pm.get("analysis", {})
        vals = [analysis.get(p, 0) or 0 for p in params]
        bars = ax.bar(labels, vals, color=colors[col], alpha=0.7)
        ax.set_title("Planet %d" % pm["planet_number"], fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    for col in range(planets_shown):
        ax = axes[1, col]
        pm = all_planets[col]
        physics = pm.get("physics", {})
        params_r = ["planet_radius_rearth", "equilibrium_temp_k", "semi_major_axis_au"]
        labels_r = ["Radius (Re)", "Teq (K)", "a (AU)"]
        vals_r = [physics.get(p, 0) or 0 for p in params_r]
        bars = ax.bar(labels_r, vals_r, color=colors[col], alpha=0.5)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        for bar, val in zip(bars, vals_r):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Multi-Planet Comparison (%d planets detected)" % n, fontsize=12, y=1.02)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_monte_carlo(mc_results):
    """Generate Monte Carlo posterior distribution plots."""
    has_period = mc_results.get("period_mean") is not None
    has_depth = mc_results.get("depth_mean") is not None

    if not has_period and not has_depth:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    if has_period:
        ax = axes[0]
        mean = mc_results["period_mean"]
        std = mc_results["period_std"]
        lo, hi = mc_results["period_1sigma"]
        x = np.linspace(mean - 4 * std, mean + 4 * std, 200) if std > 0 else np.array([mean])
        y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi)) if std > 0 else np.ones_like(x)
        ax.fill_between(x, 0, y, alpha=0.3, color="#1e40af")
        ax.plot(x, y, color="#1e40af", lw=1.5)
        if lo is not None and hi is not None:
            ax.axvspan(lo, hi, alpha=0.15, color="#dc2626", label="1-sigma")
        ax.axvline(mean, color="#1e40af", ls="--", lw=1.0, label=f"mean={mean:.4f} d")
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Probability Density")
        ax.set_title("Period Posterior (%d iterations)" % mc_results["iterations"])
        ax.legend(fontsize=8)

    if has_depth:
        ax = axes[1]
        mean = mc_results["depth_mean"]
        std = mc_results["depth_std"]
        lo, hi = mc_results["depth_1sigma"]
        x = np.linspace(mean - 4 * std, mean + 4 * std, 200) if std > 0 else np.array([mean])
        y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi)) if std > 0 else np.ones_like(x)
        ax.fill_between(x, 0, y, alpha=0.3, color="#059669")
        ax.plot(x, y, color="#059669", lw=1.5)
        if lo is not None and hi is not None:
            ax.axvspan(lo, hi, alpha=0.15, color="#dc2626", label="1-sigma")
        ax.axvline(mean, color="#059669", ls="--", lw=1.0, label=f"mean={mean:.1f} ppm")
        ax.set_xlabel("Transit Depth (ppm)")
        ax.set_ylabel("Probability Density")
        ax.set_title("Depth Posterior (%d iterations)" % mc_results["iterations"])
        ax.legend(fontsize=8)

    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_all_plots(time_arr, flux_arr, analysis):
    transit = analysis.get("transit", {})
    period = transit.get("period")
    t0 = transit.get("t0", 0.0)
    snr = analysis.get("snr", 0)
    depth_ppm = analysis.get("depth_ppm", 0)

    plots = {}

    if period and len(time_arr) > 20:
        plots["phase_folded"] = plot_phase_folded(time_arr, flux_arr, period, t0, depth_ppm, snr)
        plots["residuals"] = plot_residuals(time_arr, flux_arr, period, t0)
        plots["odd_even"] = plot_odd_even(time_arr, flux_arr, period, t0)

    if analysis.get("bls_periods") is not None:
        plots["bls_periodogram"] = plot_bls_periodogram(
            analysis["bls_periods"],
            analysis["bls_powers"],
            period,
            transit.get("snr"),
        )

    return plots
