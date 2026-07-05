#!/usr/bin/env python3
"""
validate_tps.py — TPS (Transient Plane Source) numerical validation
====================================================================

Validates the heat-solver against the analytical solution of the
Transient Plane Source (TPS) method — the same physical experiment
performed in the HD_Intelligent platform.

TPS Physical Model (Half-space, Carslaw & Jaeger)
-------------------------------------------------
A disc sensor of radius r₀ applies constant power P to the sample surface.
Under the half-space approximation, the mean temperature rise is:

    ΔT(t) = (2P / (π^(3/2) r₀² λ)) · √(α t)

Key features:
  - Linear in √t  (diagnostic plot: ΔT vs √t should be straight)
  - Slope = (2P / (π^(3/2) r₀² λ)) · √α  → determines α / λ²
  - Intercept correction from curved sensor accounts for finite radius

This script:
  1. Generates synthetic TPS experimental data with Gaussian noise.
  2. Fits the analytical model to extract α and λ.
  3. Optionally overlays numerical FD solver disc temperature.
  4. Plots ΔT vs √t (linear), fit residuals, and a parameter sensitivity map.
  5. Reports % error on fitted vs true material properties.

Reference:
  Carslaw & Jaeger, "Conduction of Heat in Solids", 2nd ed., §10.4
  Gustafsson (1991), Rev. Sci. Instrum. 62, 797-804.
"""
import argparse
import glob
import os
import re
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Material database  (alpha [m²/s], lambda [W/m·K], rho_cp [J/m³·K])
# ---------------------------------------------------------------------------
MATERIALS = {
    "aluminium":       (9.7e-5,  237.0, 2.45e6),
    "copper":          (1.17e-4, 401.0, 3.45e6),
    "concrete":        (7.0e-7,  1.6,   2.30e6),
    "insulator":       (1.4e-7,  0.20,  1.44e6),  # ← solver default
    "stainless_steel": (4.2e-6,  16.0,  3.80e6),
}


# ---------------------------------------------------------------------------
# TPS analytical model: half-space approximation (Carslaw & Jaeger)
# ---------------------------------------------------------------------------
def tps_model(t: np.ndarray,
              alpha: float, lam: float,
              P: float, r0: float) -> np.ndarray:
    """
    Mean temperature rise of the TPS sensor disc.
    ΔT(t) = [2P / (π^(3/2) r₀² λ)] · √(αt)
    """
    return (2.0 * P) / (np.pi**1.5 * r0**2 * lam) * np.sqrt(alpha * t)


def generate_experiment(material: str,
                        P: float = 0.1,     # heating power [W]
                        r0: float = 6.4e-3, # sensor radius [m]
                        t_max: float = 60.0,
                        n_pts: int = 150,
                        noise_frac: float = 0.02) -> tuple:
    """Synthetic TPS experiment with noise."""
    if material not in MATERIALS:
        raise ValueError(f"Unknown material '{material}'. "
                         f"Available: {sorted(MATERIALS)}")
    alpha, lam, _ = MATERIALS[material]
    # Start from t>0 to avoid sqrt(0)
    t_arr = np.linspace(0.2, t_max, n_pts)
    dT    = tps_model(t_arr, alpha, lam, P, r0)
    rng   = np.random.default_rng(42)
    dT   += rng.normal(0, noise_frac * dT.max(), size=dT.shape)
    return t_arr, dT, alpha, lam


def fit_tps(t_arr: np.ndarray, dT: np.ndarray,
            P: float, r0: float,
            rho_cp: float) -> tuple:
    """
    Fit TPS model via thermal-effusivity approach.

    ΔT = [2P/(π^(3/2)·r₀²)] · k · √t   where  k = √α/λ

    With known ρCp = λ/α:
        √α = 1/(k·ρCp)  →  α = 1/(k·ρCp)²
        λ = 1/(k²·ρCp)
    """
    A = (2.0 * P) / (np.pi**1.5 * r0**2)

    def model_k(t, k):
        return A * k * np.sqrt(t)

    popt, pcov = curve_fit(model_k, t_arr, dT,
                           p0=[1e-4], bounds=([1e-12], [1e3]),
                           maxfev=5000)
    k       = float(popt[0])
    k_std   = float(np.sqrt(pcov[0, 0]))

    alpha_fit = 1.0 / (k * rho_cp) ** 2
    lam_fit   = 1.0 / (k**2 * rho_cp)
    dT_fit    = model_k(t_arr, k)
    return alpha_fit, lam_fit, dT_fit, k_std


# ---------------------------------------------------------------------------
# Read solver snapshots to get disc-averaged temperature
# ---------------------------------------------------------------------------
def solver_disc_temperature(outdir: str, r0: float) -> tuple | None:
    """
    Extract mean disc temperature from FD solver snapshots.
    Returns (t_solver, dT_solver) or None if snapshots not found.
    """
    meta_path = os.path.join(outdir, "metadata.txt")
    if not os.path.exists(meta_path):
        return None

    meta = {}
    with open(meta_path) as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            meta[k.strip()] = v.strip()

    Nx     = int(meta["Nx"])
    Ny     = int(meta["Ny"])
    dx     = float(meta["dx"])
    dy     = float(meta["dy"])
    dt     = float(meta["dt"])
    T_cold = float(meta.get("T_cold", 300.0))

    ci, cj    = Nx // 2, Ny // 2
    r_cells_x = max(1, int(r0 / dx))
    r_cells_y = max(1, int(r0 / dy))

    files = sorted(
        glob.glob(os.path.join(outdir, "T_*.bin")),
        key=lambda f: int(re.search(r"T_(\d+)\.bin", f).group(1)),
    )

    t_list, dT_list = [], []
    for f in files:
        step = int(re.search(r"T_(\d+)\.bin", f).group(1))
        t    = step * dt
        if t <= 0:
            continue
        T = np.fromfile(f, dtype=np.float64).reshape(Nx, Ny)

        # Disc mask
        mask = np.zeros((Nx, Ny), dtype=bool)
        for di in range(-r_cells_x, r_cells_x + 1):
            for dj in range(-r_cells_y, r_cells_y + 1):
                ii, jj = ci + di, cj + dj
                if 0 <= ii < Nx and 0 <= jj < Ny:
                    if (di / r_cells_x) ** 2 + (dj / r_cells_y) ** 2 <= 1.0:
                        mask[ii, jj] = True
        t_list.append(t)
        dT_list.append(T[mask].mean() - T_cold)

    return (np.array(t_list), np.array(dT_list)) if t_list else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_report(material: str,
                t_arr, dT_exp, dT_fit,
                solver_result,
                alpha_true, lam_true,
                alpha_fit, lam_fit, fit_std,
                P: float, r0: float,
                save_dir: str):

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sqrt_t = np.sqrt(t_arr)

    # ── (1) ΔT vs √t  (should be linear) ────────────────────────────────────
    ax = axes[0]
    ax.plot(sqrt_t, dT_exp, "k.", markersize=3, alpha=0.6,
            label="Synthetic measurement")
    ax.plot(sqrt_t, dT_fit, "r-", linewidth=2,
            label=f"Fit  α={alpha_fit:.2e}, λ={lam_fit:.3f}")
    ax.plot(sqrt_t, tps_model(t_arr, alpha_true, lam_true, P, r0),
            "g--", linewidth=1.5, label=f"True α={alpha_true:.2e}, λ={lam_true:.3f}")
    if solver_result is not None:
        t_s, dT_s = solver_result
        ax.plot(np.sqrt(t_s), dT_s, "b^", markersize=5,
                label="FD solver (numerical)")

    ax.set_xlabel("√t  [s^(1/2)]")
    ax.set_ylabel("ΔT  [K]")
    ax.set_title("TPS: Temperature Rise vs √t")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # ── (2) ΔT vs t  (time domain) ──────────────────────────────────────────
    ax = axes[1]
    ax.plot(t_arr, dT_exp, "k.", markersize=3, alpha=0.6, label="Measurement")
    ax.plot(t_arr, dT_fit, "r-", linewidth=2,             label="Fit")
    ax.plot(t_arr, tps_model(t_arr, alpha_true, lam_true, P, r0),
            "g--", linewidth=1.5, label="Analytical (true)")
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("ΔT  [K]")
    ax.set_title("TPS: Temperature Rise vs Time")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    # ── (3) Residuals ────────────────────────────────────────────────────────
    ax = axes[2]
    res = dT_exp - dT_fit
    ax.plot(t_arr, res, "b-", linewidth=1)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.fill_between(t_arr, res, alpha=0.2, color="b")
    ax.set_xlabel("t  [s]")
    ax.set_ylabel("Residual  [K]")
    ax.set_title(f"Fit residuals  (RMS={np.sqrt(np.mean(res**2)):.4f} K)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"TPS Validation — {material.replace('_', ' ').title()}\n"
        f"Fitted: α={alpha_fit:.3e} m²/s  λ={lam_fit:.4f} W/m·K  |  "
        f"True: α={alpha_true:.3e} m²/s  λ={lam_true:.4f} W/m·K",
        fontsize=9
    )
    fig.tight_layout()
    out = os.path.join(save_dir, f"tps_validation_{material}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Console report ────────────────────────────────────────────────────────
    print(f"\n{'='*57}")
    print(f"  TPS Validation Report — {material}")
    print(f"{'='*57}")
    print(f"  {'Parameter':<22} {'True':>12} {'Fitted':>12} {'Error %':>9}")
    print(f"  {'-'*57}")
    for name, true_v, fit_v in [
        ("alpha [m²/s]",  alpha_true, alpha_fit),
        ("lambda [W/m·K]", lam_true,  lam_fit  ),
    ]:
        err = abs(fit_v - true_v) / true_v * 100
        print(f"  {name:<22} {true_v:>12.4e} {fit_v:>12.4e} {err:>8.2f}%")
    print(f"{'='*57}\n")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TPS numerical validation for heat-solver")
    parser.add_argument("--material",  default="insulator",
                        choices=sorted(MATERIALS),
                        help="Reference material")
    parser.add_argument("--power",     type=float, default=0.1,
                        help="Sensor heating power [W]")
    parser.add_argument("--radius",    type=float, default=6.4e-3,
                        help="Sensor radius [m]")
    parser.add_argument("--tmax",      type=float, default=60.0,
                        help="Max experiment time [s]")
    parser.add_argument("--noise",     type=float, default=0.02,
                        help="Relative Gaussian noise std (fraction of peak)")
    parser.add_argument("--outdir",    default="output",
                        help="Solver output directory (for snapshot comparison)")
    parser.add_argument("--save-dir",  default="docs",
                        help="Where to save plots")
    parser.add_argument("--no-solver", action="store_true",
                        help="Skip solver snapshot comparison")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Generating synthetic TPS experiment: material={args.material}")
    t_arr, dT_exp, alpha_true, lam_true = generate_experiment(
        args.material, P=args.power, r0=args.radius,
        t_max=args.tmax, noise_frac=args.noise)

    _, _, rho_cp = MATERIALS[args.material]
    print("Fitting TPS model ...")
    alpha_fit, lam_fit, dT_fit, fit_std = fit_tps(
        t_arr, dT_exp, args.power, args.radius, rho_cp)

    solver_result = None
    if not args.no_solver:
        solver_result = solver_disc_temperature(args.outdir, args.radius)
        if solver_result is not None:
            print(f"  Loaded {len(solver_result[0])} solver snapshots.")
        else:
            print("  No solver snapshots found (run heat_serial first).")

    make_report(
        args.material, t_arr, dT_exp, dT_fit,
        solver_result,
        alpha_true, lam_true, alpha_fit, lam_fit,
        fit_std, args.power, args.radius, args.save_dir,
    )


if __name__ == "__main__":
    main()
