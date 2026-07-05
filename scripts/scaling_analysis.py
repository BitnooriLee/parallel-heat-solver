#!/usr/bin/env python3
"""
scaling_analysis.py — Weak/Strong scaling analysis + Amdahl's law
==================================================================

Reads bench_results.csv written by bench_scaling (or manual timing runs),
then produces:

  1. Strong scaling:  Speedup & Efficiency vs. number of threads/ranks
     with Amdahl's ideal curve overlaid.
  2. Weak scaling:    Efficiency vs. number of threads/ranks
     with Gustafsson's (iso-granular) ideal.
  3. Communication/Computation ratio vs. ranks (MPI data).
  4. Roofline model estimate vs. grid size.

Usage:
  python scripts/scaling_analysis.py --csv bench_results.csv
  python scripts/scaling_analysis.py --csv mpi_timings.csv --mpi
  python scripts/scaling_analysis.py --demo    # generate synthetic data + plot
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Amdahl / Gustafsson models
# ---------------------------------------------------------------------------
def amdahl_speedup(p: np.ndarray, f_parallel: float) -> np.ndarray:
    """
    Amdahl's law:  S(p) = 1 / [(1 - f) + f/p]
    f_parallel ∈ [0, 1]: parallel fraction
    """
    return 1.0 / ((1.0 - f_parallel) + f_parallel / p)


def gustafsson_speedup(p: np.ndarray, f_serial: float) -> np.ndarray:
    """
    Gustafsson's (scaled speedup):  S(p) = p - f_serial*(p - 1)
    f_serial: serial fraction at single processor
    """
    return p - f_serial * (p - 1.0)


def fit_amdahl(p: np.ndarray, speedup: np.ndarray) -> float:
    """Fit Amdahl model: return best-fit parallel fraction f."""
    from scipy.optimize import minimize_scalar
    def residual(f):
        pred = amdahl_speedup(p, f)
        return np.sum((speedup - pred)**2)
    res = minimize_scalar(residual, bounds=(0.5, 1.0), method="bounded")
    return res.x


# ---------------------------------------------------------------------------
def plot_strong_scaling(df: pd.DataFrame, save_dir: str):
    df_s = df[df["mode"] == "strong"].copy()
    if df_s.empty:
        print("  No strong-scaling data found."); return

    threads  = df_s["threads"].values.astype(float)
    speedup  = df_s["speedup"].values
    eff      = df_s["efficiency"].values

    # Fit Amdahl
    try:
        f_par = fit_amdahl(threads, speedup)
    except Exception:
        f_par = 0.90

    p_fine   = np.linspace(1, threads.max(), 200)
    ideal_s  = p_fine              # perfect linear speedup
    amdahl_s = amdahl_speedup(p_fine, f_par)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Speedup ─────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(p_fine, ideal_s,  "k--",  linewidth=1,   label="Ideal (linear)")
    ax.plot(p_fine, amdahl_s, "r--",  linewidth=1.5,
            label=f"Amdahl  f={f_par:.3f}")
    ax.plot(threads, speedup, "bo-",  linewidth=2,  markersize=6,
            label="Measured")
    ax.set_xlabel("Threads / Cores")
    ax.set_ylabel("Speedup S(p)")
    ax.set_title("Strong Scaling — Speedup")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, threads.max() * 1.1)
    ax.set_ylim(0.5, None)

    # ── Efficiency ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Ideal")
    ax.plot(threads, eff, "bs-", linewidth=2, markersize=6, label="Measured")
    ax.set_xlabel("Threads / Cores")
    ax.set_ylabel("Parallel Efficiency E(p) = S(p)/p")
    ax.set_title("Strong Scaling — Efficiency")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Strong Scaling  |  Grid: {df_s['Nx'].iloc[0]}²  "
        f"|  Amdahl parallel fraction f = {f_par:.4f}",
        fontsize=10
    )
    fig.tight_layout()
    out = os.path.join(save_dir, "strong_scaling.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved: {out}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Strong Scaling Summary  (Amdahl f={f_par:.4f})")
    print(f"{'='*60}")
    print(f"  {'Threads':>8} {'Time(s)':>10} {'Speedup':>9} {'Efficiency':>12} {'GFLOP/s':>9}")
    print(f"  {'-'*60}")
    for _, row in df_s.iterrows():
        print(f"  {int(row['threads']):>8} {row['time_s']:>10.4f} "
              f"{row['speedup']:>9.3f} {row['efficiency']:>11.3f} "
              f"{row['gflops']:>9.3f}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
def plot_weak_scaling(df: pd.DataFrame, save_dir: str):
    df_w = df[df["mode"] == "weak"].copy()
    if df_w.empty:
        print("  No weak-scaling data found."); return

    threads = df_w["threads"].values.astype(float)
    eff     = df_w["efficiency"].values
    t       = df_w["time_s"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Efficiency ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="Ideal")
    ax.plot(threads, eff, "go-", linewidth=2, markersize=6, label="Measured")
    ax.set_xlabel("Threads / Cores")
    ax.set_ylabel("Weak-scaling Efficiency")
    ax.set_title("Weak Scaling — Efficiency")
    ax.set_ylim(0, 1.2)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # ── Wall time per fixed work unit ────────────────────────────────────────
    ax = axes[1]
    ax.axhline(t[0], color="k", linestyle="--", linewidth=1, label="Ideal (constant)")
    ax.plot(threads, t, "gs-", linewidth=2, markersize=6, label="Measured")
    ax.set_xlabel("Threads / Cores")
    ax.set_ylabel("Wall time [s]  (per unit work)")
    ax.set_title("Weak Scaling — Wall Time")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    fig.suptitle("Weak Scaling  |  Work ∝ Cores  (iso-granular)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(save_dir, "weak_scaling.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
def plot_comm_compute(df: pd.DataFrame, save_dir: str):
    """MPI communication/computation ratio vs ranks."""
    needed = {"ranks", "comm_s", "compute_s"}
    if not needed.issubset(df.columns):
        print("  No comm/compute columns — skipping."); return

    df = df.sort_values("ranks")
    ranks   = df["ranks"].values
    ratio   = df["comm_s"].values / df["compute_s"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(ranks)), ratio, tick_label=ranks, color="steelblue", alpha=0.8)
    ax.set_xlabel("MPI Ranks")
    ax.set_ylabel("Comm / Compute ratio")
    ax.set_title("Communication vs. Computation  (MPI halo exchange)")
    ax.axhline(0.10, color="r", linestyle="--", linewidth=1, label="10% threshold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(save_dir, "comm_compute_ratio.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
def generate_demo_data() -> pd.DataFrame:
    """
    Generate synthetic scaling data that mimics a real 2D heat solver on
    a typical shared-memory machine.

    Assumptions:
      - Serial baseline: 512×512 grid, ~4.2 s
      - Parallel fraction f ≈ 0.97  (serial overhead: BC apply, swap)
      - Weak-scaling efficiency drops ~3%/2× doubling due to NUMA/cache effects
    """
    f_par  = 0.975
    t_serial = 4.20  # seconds

    # Strong scaling: fixed 512² grid
    threads_strong = [1, 2, 4, 8, 16]
    rows_strong = []
    for p in threads_strong:
        ideal_t = t_serial / p
        t = t_serial / amdahl_speedup(np.array([float(p)]), f_par)[0]
        # Add small random jitter
        t *= np.random.default_rng(p).uniform(0.97, 1.02)
        n  = (512-2)**2
        rows_strong.append({
            "mode": "strong", "threads": p,
            "Nx": 512, "Ny": 512, "n_cells": n,
            "steps": 500, "time_s": t,
            "speedup": t_serial / t,
            "efficiency": t_serial / (t * p),
            "gflops": 7e-9 * n * 500 / t,
        })

    # Weak scaling: base 512×512/thread → scale grid
    t_base = t_serial
    threads_weak = [1, 2, 4, 8, 16]
    rows_weak = []
    for p in threads_weak:
        side = int(512 * np.sqrt(p))
        t    = t_base * (1 + 0.03 * np.log2(p))  # mild overhead
        n    = (side-2)**2
        rows_weak.append({
            "mode": "weak", "threads": p,
            "Nx": side, "Ny": side, "n_cells": n,
            "steps": 500, "time_s": t,
            "speedup": t / t,
            "efficiency": t_base / t,
            "gflops": 7e-9 * n * 500 / t,
        })

    return pd.DataFrame(rows_strong + rows_weak)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Scaling analysis for parallel heat solver")
    parser.add_argument("--csv",      default=None,
                        help="Benchmark CSV file")
    parser.add_argument("--save-dir", default="docs",
                        help="Output directory for plots")
    parser.add_argument("--demo",     action="store_true",
                        help="Generate and plot synthetic demo data")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.demo:
        print("Generating synthetic demo data ...")
        df = generate_demo_data()
        csv_out = os.path.join(args.save_dir, "demo_scaling.csv")
        df.to_csv(csv_out, index=False)
        print(f"Demo CSV: {csv_out}")
    elif args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: {args.csv} not found.", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.csv)
    else:
        # Try default location
        default = "bench_results.csv"
        if os.path.exists(default):
            df = pd.read_csv(default)
        else:
            print("No data provided.  Use --csv or --demo.", file=sys.stderr)
            sys.exit(1)

    print(f"Loaded {len(df)} rows.")
    plot_strong_scaling(df, args.save_dir)
    plot_weak_scaling(df, args.save_dir)
    plot_comm_compute(df, args.save_dir)

    print("\nAll plots saved to:", args.save_dir)


if __name__ == "__main__":
    main()
