// main.cpp — Serial heat-solver driver
//
//  Usage:
//    ./heat_serial [options]
//
//  Options:
//    --nx N       Grid points in x          (default: 256)
//    --ny N       Grid points in y          (default: same as Nx)
//    --lx F       Domain length in x [m]    (default: 1.0)
//    --ly F       Domain length in y [m]    (default: same as Lx)
//    --alpha F    Thermal diffusivity [m²/s](default: 1.4e-7)
//    --steps N    Number of time steps      (default: 2000)
//    --out N      Snapshot interval (0=off) (default: 200)
//    --outdir S   Output directory           (default: "output")
//    --T_hot F    Hot-spot temperature [K]  (default: 500)
//    --T_cold F   Cold temperature [K]      (default: 300)
// ---------------------------------------------------------------------------
#include "heat/fd_solver.hpp"
#include "heat/types.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

static void usage(const char* prog) {
    std::printf("Usage: %s [--nx N] [--ny N] [--lx F] [--alpha F] "
                "[--steps N] [--out N] [--outdir S] "
                "[--T_hot F] [--T_cold F]\n", prog);
}

static heat::Config parse_args(int argc, char** argv) {
    heat::Config cfg;
    for (int i = 1; i < argc; ++i) {
        auto get_next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", argv[i]);
                std::exit(1);
            }
            return argv[++i];
        };
        if      (!std::strcmp(argv[i], "--nx"))     cfg.Nx      = std::atoi(get_next());
        else if (!std::strcmp(argv[i], "--ny"))     cfg.Ny      = std::atoi(get_next());
        else if (!std::strcmp(argv[i], "--lx"))     cfg.Lx      = std::atof(get_next());
        else if (!std::strcmp(argv[i], "--ly"))     cfg.Ly      = std::atof(get_next());
        else if (!std::strcmp(argv[i], "--alpha"))  cfg.alpha   = std::atof(get_next());
        else if (!std::strcmp(argv[i], "--steps"))  cfg.n_steps = std::atoi(get_next());
        else if (!std::strcmp(argv[i], "--out"))    cfg.out_every = std::atoi(get_next());
        else if (!std::strcmp(argv[i], "--outdir")) cfg.out_dir = get_next();
        else if (!std::strcmp(argv[i], "--T_hot"))  cfg.T_hot   = std::atof(get_next());
        else if (!std::strcmp(argv[i], "--T_cold")) cfg.T_cold  = std::atof(get_next());
        else if (!std::strcmp(argv[i], "--help"))   { usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]); std::exit(1);
        }
    }
    // Sync defaults
    if (cfg.Ly  == 1.0 && cfg.Lx != 1.0) cfg.Ly = cfg.Lx;
    if (cfg.Ny  == 256 && cfg.Nx != 256)  cfg.Ny = cfg.Nx;
    cfg.T_bc = cfg.T_cold;
    return cfg;
}

int main(int argc, char** argv) {
    heat::Config cfg = parse_args(argc, argv);

    try {
        cfg.validate();
    } catch (const std::runtime_error& e) {
        std::fprintf(stderr, "Config error: %s\n", e.what());
        return 1;
    }

    std::printf("=== Serial Heat Solver ===\n");
    std::printf("  Grid    : %d × %d\n",    cfg.Nx, cfg.Ny);
    std::printf("  Domain  : %.2f × %.2f m\n", cfg.Lx, cfg.Ly);
    std::printf("  alpha   : %.2e m²/s\n",  cfg.alpha);
    std::printf("  dt      : %.4e s  (CFL r=%.4f)\n",
                cfg.effective_dt(), cfg.r_x() + cfg.r_y());
    std::printf("  Steps   : %d\n", cfg.n_steps);
    std::printf("  Output  : %s (every %d steps)\n\n",
                cfg.out_dir.c_str(), cfg.out_every);

    heat::FDSolver2D solver(cfg, /*use_omp=*/false);
    solver.run();

    const auto& s = solver.timing();
    long long   n = (long long)(cfg.Nx - 2) * (cfg.Ny - 2);

    std::printf("=== Results ===\n");
    std::printf("  Total time  : %.4f s\n",  s.total_s);
    std::printf("  Compute     : %.4f s  (%.1f%%)\n",
                s.compute_s, 100.0 * s.compute_fraction());
    std::printf("  I/O + misc  : %.4f s\n",  s.io_s);
    std::printf("  Performance : %.3f GFLOP/s\n",
                s.gflops(n, cfg.n_steps));
    std::printf("  Throughput  : %.2f Mcell-steps/s\n",
                1e-6 * n * cfg.n_steps / s.total_s);

    return 0;
}
