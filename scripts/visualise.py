#!/usr/bin/env python3
"""
visualise.py — Heatmap visualiser for parallel-heat-solver output
=================================================================

Reads binary snapshot files (T_<step>.bin) and metadata.txt written by
heat_serial / heat_openmp / heat_mpi, then produces:

  1. A single heatmap PNG  (--snapshot)
  2. A side-by-side animation GIF   (--animate)
  3. A temperature profile along x=Ly/2  (--profile)
  4. A convergence plot: max ΔT per step  (--convergence)

Usage:
  python scripts/visualise.py --outdir output [--snapshot step]
  python scripts/visualise.py --outdir output --animate --fps 10
  python scripts/visualise.py --outdir output --profile
"""
import argparse
import glob
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# Custom "plasma-like" colormap that reads well in print
CMAP = plt.cm.plasma


# ---------------------------------------------------------------------------
def read_metadata(outdir: str) -> dict:
    meta = {}
    path = os.path.join(outdir, "metadata.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata.txt not found in {outdir}")
    with open(path) as f:
        for line in f:
            k, _, v = line.strip().partition("=")
            meta[k.strip()] = v.strip()
    return meta


def read_snapshot(outdir: str, step: int, Nx: int, Ny: int) -> np.ndarray:
    fname = os.path.join(outdir, f"T_{step}.bin")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Snapshot not found: {fname}")
    T = np.fromfile(fname, dtype=np.float64).reshape(Nx, Ny)
    return T


def list_steps(outdir: str) -> list[int]:
    pattern = os.path.join(outdir, "T_*.bin")
    files   = sorted(glob.glob(pattern),
                     key=lambda f: int(re.search(r"T_(\d+)\.bin", f).group(1)))
    return [int(re.search(r"T_(\d+)\.bin", f).group(1)) for f in files]


# ---------------------------------------------------------------------------
def plot_snapshot(T: np.ndarray, meta: dict, step: int, out_png: str):
    Nx   = int(meta["Nx"])
    Ny   = int(meta["Ny"])
    dx   = float(meta["dx"])
    dy   = float(meta["dy"])
    dt   = float(meta["dt"])
    alpha= float(meta["alpha"])

    x = np.linspace(0, dx*(Nx-1), Nx)
    y = np.linspace(0, dy*(Ny-1), Ny)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.pcolormesh(y*1e2, x*1e2, T, cmap=CMAP, shading="auto")
    cb = fig.colorbar(im, ax=ax, label="Temperature [K]")
    ax.set_xlabel("y [cm]")
    ax.set_ylabel("x [cm]")
    ax.set_title(
        f"Heat diffusion  —  step {step}  "
        f"(t = {step*dt:.4f} s)\n"
        f"α = {alpha:.2e} m²/s,  grid {Nx}×{Ny}"
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_png}")


# ---------------------------------------------------------------------------
def animate(outdir: str, steps: list[int], meta: dict, fps: int, out_gif: str):
    Nx = int(meta["Nx"])
    Ny = int(meta["Ny"])
    dx = float(meta["dx"])
    dy = float(meta["dy"])
    dt = float(meta["dt"])

    T0     = read_snapshot(outdir, steps[0], Nx, Ny)
    T_all  = [T0] + [read_snapshot(outdir, s, Nx, Ny) for s in steps[1:]]

    vmin = min(T.min() for T in T_all)
    vmax = max(T.max() for T in T_all)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(T0, cmap=CMAP, vmin=vmin, vmax=vmax, shading="auto")
    cb = fig.colorbar(im, ax=ax, label="T [K]")
    title = ax.set_title("")
    ax.set_xlabel("j (y-direction)")
    ax.set_ylabel("i (x-direction)")

    def update(frame):
        T = T_all[frame]
        im.set_array(T.ravel())
        t = steps[frame] * dt
        title.set_text(f"step {steps[frame]:5d}   t = {t:.4f} s")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(steps),
                                   interval=1000//fps, blit=True)
    ani.save(out_gif, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Animation saved: {out_gif}")


# ---------------------------------------------------------------------------
def plot_profile(outdir: str, steps: list[int], meta: dict, out_png: str):
    """Temperature profile along the centre row for each snapshot."""
    Nx = int(meta["Nx"])
    Ny = int(meta["Ny"])
    dx = float(meta["dx"])
    dt = float(meta["dt"])

    x = np.linspace(0, dx*(Nx-1), Nx) * 1e2  # [cm]
    ci = Ny // 2  # centre column

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap_seq = plt.cm.viridis(np.linspace(0, 1, len(steps)))

    for step, color in zip(steps, cmap_seq):
        T = read_snapshot(outdir, step, Nx, Ny)
        ax.plot(x, T[:, ci], color=color,
                label=f"t={step*dt:.3f}s" if step in steps[::max(1,len(steps)//5)] else "")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                norm=plt.Normalize(0, steps[-1]*dt))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time [s]")

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title(f"Temperature profile along y = L/2\nGrid {Nx}×{Ny}, α={float(meta['alpha']):.2e} m²/s")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_png}")


# ---------------------------------------------------------------------------
def plot_convergence(outdir: str, steps: list[int], meta: dict, out_png: str):
    """Max temperature vs step — shows diffusion convergence to steady state."""
    Nx = int(meta["Nx"])
    Ny = int(meta["Ny"])
    dt = float(meta["dt"])

    T_max = []
    T_mean = []
    for step in steps:
        T = read_snapshot(outdir, step, Nx, Ny)
        T_max.append(T.max())
        T_mean.append(T.mean())

    t_arr = np.array(steps) * dt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(t_arr, T_max, "r-o", markersize=3, label="T_max")
    axes[0].plot(t_arr, T_mean, "b-s", markersize=3, label="T_mean")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Convergence: Max & Mean Temperature")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ΔT_max between consecutive snapshots
    dT = np.abs(np.diff(T_max))
    axes[1].semilogy(t_arr[1:], dT, "g-^", markersize=3)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("|ΔT_max| [K] (log scale)")
    axes[1].set_title("Temperature change per interval")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_png}")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Heat-solver output visualiser")
    parser.add_argument("--outdir",      default="output",          help="Solver output directory")
    parser.add_argument("--snapshot",    type=int, default=None,    help="Plot single snapshot at this step")
    parser.add_argument("--animate",     action="store_true",       help="Create GIF animation")
    parser.add_argument("--profile",     action="store_true",       help="Plot temperature profiles")
    parser.add_argument("--convergence", action="store_true",       help="Plot convergence curves")
    parser.add_argument("--fps",         type=int, default=8,       help="Frames per second for animation")
    parser.add_argument("--save-dir",    default="docs",            help="Where to save plots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    try:
        meta  = read_metadata(args.outdir)
        steps = list_steps(args.outdir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not steps:
        print("No snapshots found.", file=sys.stderr)
        sys.exit(1)

    Nx = int(meta["Nx"])
    Ny = int(meta["Ny"])

    if args.snapshot is not None:
        T   = read_snapshot(args.outdir, args.snapshot, Nx, Ny)
        out = os.path.join(args.save_dir, f"snapshot_{args.snapshot}.png")
        plot_snapshot(T, meta, args.snapshot, out)

    elif not any([args.animate, args.profile, args.convergence]):
        # Default: plot first and last snapshots
        for step in [steps[0], steps[-1]]:
            T   = read_snapshot(args.outdir, step, Nx, Ny)
            out = os.path.join(args.save_dir, f"snapshot_{step}.png")
            plot_snapshot(T, meta, step, out)

    if args.animate:
        out = os.path.join(args.save_dir, "heat_diffusion.gif")
        animate(args.outdir, steps, meta, args.fps, out)

    if args.profile:
        out = os.path.join(args.save_dir, "temperature_profile.png")
        plot_profile(args.outdir, steps, meta, out)

    if args.convergence:
        out = os.path.join(args.save_dir, "convergence.png")
        plot_convergence(args.outdir, steps, meta, out)


if __name__ == "__main__":
    main()
