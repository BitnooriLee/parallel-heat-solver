// main.cpp — OpenMP heat-solver driver
//
//  Same CLI as heat_serial, plus:
//    --threads N   Number of OpenMP threads (default: OMP_NUM_THREADS)
//
//  The solver uses the same FDSolver2D with use_omp=true so the stencil loop
//  is decorated with #pragma omp parallel for.
// ---------------------------------------------------------------------------
#include "heat/fd_solver.hpp"
#include "heat/types.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

static void usage(const char* prog) {
    std::printf("Usage: %s [--nx N] [--ny N] [--alpha F] [--steps N] "
                "[--out N] [--outdir S] [--threads N]\n", prog);
}

static heat::Config parse_args(int argc, char** argv, int& n_threads) {
    heat::Config cfg;
    n_threads = 0;  // 0 = use OMP_NUM_THREADS
    for (int i = 1; i < argc; ++i) {
        auto get_next = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "Missing value\n"); std::exit(1); }
            return argv[++i];
        };
        if      (!std::strcmp(argv[i], "--nx"))      cfg.Nx      = std::atoi(get_next());
        else if (!std::strcmp(argv[i], "--ny"))      cfg.Ny      = std::atoi(get_next());
        else if (!std::strcmp(argv[i], "--lx"))      cfg.Lx      = std::atof(get_next());
        else if (!std::strcmp(argv[i], "--alpha"))   cfg.alpha   = std::atof(get_next());
        else if (!std::strcmp(argv[i], "--steps"))   cfg.n_steps = std::atoi(get_next());
        else if (!std::strcmp(argv[i], "--out"))     cfg.out_every = std::atoi(get_next());
        else if (!std::strcmp(argv[i], "--outdir"))  cfg.out_dir = get_next();
        else if (!std::strcmp(argv[i], "--threads")) n_threads   = std::atoi(get_next());
        else if (!std::strcmp(argv[i], "--T_hot"))   cfg.T_hot   = std::atof(get_next());
        else if (!std::strcmp(argv[i], "--T_cold"))  cfg.T_cold  = std::atof(get_next());
        else if (!std::strcmp(argv[i], "--help"))    { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr, "Unknown: %s\n", argv[i]); usage(argv[0]); std::exit(1); }
    }
    if (cfg.Ny == 256 && cfg.Nx != 256) cfg.Ny = cfg.Nx;
    cfg.T_bc = cfg.T_cold;
    return cfg;
}

int main(int argc, char** argv) {
    int n_threads = 0;
    heat::Config cfg = parse_args(argc, argv, n_threads);

#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
    int actual_threads = omp_get_max_threads();
#else
    int actual_threads = 1;
    std::fprintf(stderr, "WARNING: compiled without OpenMP; running serially.\n");
#endif

    try { cfg.validate(); }
    catch (const std::runtime_error& e) {
        std::fprintf(stderr, "Config error: %s\n", e.what()); return 1;
    }

    std::printf("=== OpenMP Heat Solver ===\n");
    std::printf("  Grid    : %d × %d\n",   cfg.Nx, cfg.Ny);
    std::printf("  Threads : %d\n",        actual_threads);
    std::printf("  alpha   : %.2e m²/s\n", cfg.alpha);
    std::printf("  dt      : %.4e s  (CFL r=%.4f)\n",
                cfg.effective_dt(), cfg.r_x() + cfg.r_y());
    std::printf("  Steps   : %d\n\n",      cfg.n_steps);

    heat::FDSolver2D solver(cfg, /*use_omp=*/true);
    solver.run();

    const auto& s = solver.timing();
    long long   n = (long long)(cfg.Nx - 2) * (cfg.Ny - 2);

    std::printf("=== Results ===\n");
    std::printf("  Total time  : %.4f s\n",  s.total_s);
    std::printf("  Compute     : %.4f s  (%.1f%%)\n",
                s.compute_s, 100.0 * s.compute_fraction());
    std::printf("  Performance : %.3f GFLOP/s\n", s.gflops(n, cfg.n_steps));
    std::printf("  Throughput  : %.2f Mcell-steps/s\n",
                1e-6 * n * cfg.n_steps / s.total_s);

    return 0;
}
