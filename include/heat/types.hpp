// types.hpp — Shared configuration and type definitions for the heat solver
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>

namespace heat {

// ---------------------------------------------------------------------------
// Solver configuration — passed to every solver variant unchanged
// ---------------------------------------------------------------------------
struct Config {
    // ── Grid ────────────────────────────────────────────────────────────────
    int    Nx      = 256;     // Global grid points in x (rows)
    int    Ny      = 256;     // Global grid points in y (cols)
    double Lx      = 1.0;    // Physical domain length in x  [m]
    double Ly      = 1.0;    // Physical domain length in y  [m]

    // ── Physics ─────────────────────────────────────────────────────────────
    // Thermal diffusivity [m²/s].
    //   Concrete    ≈ 7e-7
    //   Aluminium   ≈ 9.7e-5
    //   Default: generic insulating material used in TPS experiments
    double alpha = 1.4e-7;

    // ── Time integration ────────────────────────────────────────────────────
    double dt      = 0.0;    // 0 → auto-select CFL-stable step
    int    n_steps = 2000;   // Number of time steps to advance

    // ── Boundary / initial conditions ───────────────────────────────────────
    double T_cold = 300.0;   // Ambient / initial temperature [K]
    double T_hot  = 500.0;   // Hot-spot temperature (Gaussian centre)  [K]
    double T_bc   = 300.0;   // Dirichlet boundary temperature [K]

    // ── Output ──────────────────────────────────────────────────────────────
    int         out_every = 200;     // Write snapshot every N steps (0 = off)
    std::string out_dir   = "output";

    // ── Derived quantities (always computed, never stored) ──────────────────
    double dx()  const { return Lx / (Nx - 1); }
    double dy()  const { return Ly / (Ny - 1); }

    // CFL-stable time step: r = α·dt/h² ≤ 0.25 in 2D
    double dt_cfl() const {
        double h = std::min(dx(), dy());
        return 0.24 * h * h / alpha;   // safety margin below 0.25
    }

    double effective_dt() const { return (dt > 0.0) ? dt : dt_cfl(); }

    // Fourier numbers per axis (must satisfy r_x + r_y ≤ 0.5 for stability)
    double r_x() const { return alpha * effective_dt() / (dx() * dx()); }
    double r_y() const { return alpha * effective_dt() / (dy() * dy()); }

    void validate() const {
        double r = r_x() + r_y();
        if (r > 0.5)
            throw std::runtime_error(
                "CFL violated: r_x + r_y = " + std::to_string(r) + " > 0.5");
    }
};

// ---------------------------------------------------------------------------
// Lightweight timing helper used by all solver variants
// ---------------------------------------------------------------------------
struct TimingStats {
    double total_s    = 0.0;  // Total wall time [s]
    double compute_s  = 0.0;  // Time in stencil computation [s]
    double comm_s     = 0.0;  // Time in MPI communication (halo exchange) [s]
    double io_s       = 0.0;  // Time in file I/O [s]

    double comm_fraction() const {
        return (total_s > 0) ? comm_s / total_s : 0.0;
    }
    double compute_fraction() const {
        return (total_s > 0) ? compute_s / total_s : 0.0;
    }
    // GFLOP/s: 7 floating-point ops per interior cell per step
    double gflops(long long n_cells, int steps) const {
        return (compute_s > 0)
            ? 7.0e-9 * n_cells * steps / compute_s
            : 0.0;
    }
};

}  // namespace heat
