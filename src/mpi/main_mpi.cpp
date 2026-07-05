// main_mpi.cpp — MPI + OpenMP heat-solver driver
//
//  Usage:
//    mpirun -np <N> ./heat_mpi [options]
//
//  Options: (same as heat_serial, plus)
//    --threads N    OpenMP threads per rank  (default: OMP_NUM_THREADS)
//    --profile      Print per-rank timing breakdown
//
//  Output:
//    Rank 0 prints a summary with:
//      • Total wall time, compute time, communication time
//      • Communication / computation ratio
//      • Parallel efficiency vs. serial baseline (if --serial-time given)
//      • GFLOP/s
// ---------------------------------------------------------------------------
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "heat/fd_solver.hpp"
#include "heat/types.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

static void usage(const char* prog) {
    std::printf(
        "Usage: mpirun -np N %s [--nx N] [--ny N] [--lx F] [--alpha F]\n"
        "                        [--steps N] [--out N] [--outdir S]\n"
        "                        [--threads N] [--T_hot F] [--T_cold F]\n"
        "                        [--serial-time F] [--profile]\n",
        prog);
}

struct Args {
    heat::Config cfg;
    int    n_threads    = 0;
    double serial_time  = 0.0;  // for computing speedup
    bool   profile      = false;
};

static Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        auto nxt = [&]() -> const char* {
            if (i+1 >= argc) { std::fprintf(stderr,"Missing arg\n"); std::exit(1); }
            return argv[++i];
        };
        if      (!std::strcmp(argv[i],"--nx"))          a.cfg.Nx       = std::atoi(nxt());
        else if (!std::strcmp(argv[i],"--ny"))          a.cfg.Ny       = std::atoi(nxt());
        else if (!std::strcmp(argv[i],"--lx"))          a.cfg.Lx       = std::atof(nxt());
        else if (!std::strcmp(argv[i],"--ly"))          a.cfg.Ly       = std::atof(nxt());
        else if (!std::strcmp(argv[i],"--alpha"))       a.cfg.alpha    = std::atof(nxt());
        else if (!std::strcmp(argv[i],"--steps"))       a.cfg.n_steps  = std::atoi(nxt());
        else if (!std::strcmp(argv[i],"--out"))         a.cfg.out_every= std::atoi(nxt());
        else if (!std::strcmp(argv[i],"--outdir"))      a.cfg.out_dir  = nxt();
        else if (!std::strcmp(argv[i],"--T_hot"))       a.cfg.T_hot    = std::atof(nxt());
        else if (!std::strcmp(argv[i],"--T_cold"))      a.cfg.T_cold   = std::atof(nxt());
        else if (!std::strcmp(argv[i],"--threads"))     a.n_threads    = std::atoi(nxt());
        else if (!std::strcmp(argv[i],"--serial-time")) a.serial_time  = std::atof(nxt());
        else if (!std::strcmp(argv[i],"--profile"))     a.profile      = true;
        else if (!std::strcmp(argv[i],"--help"))        { usage(argv[0]); std::exit(0); }
        else { std::fprintf(stderr,"Unknown: %s\n",argv[i]); usage(argv[0]); std::exit(1); }
    }
    if (a.cfg.Ny == 256 && a.cfg.Nx != 256) a.cfg.Ny = a.cfg.Nx;
    a.cfg.T_bc = a.cfg.T_cold;
    return a;
}

int main(int argc, char** argv) {
#ifdef USE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "WARNING: MPI thread support < MPI_THREAD_FUNNELED\n");
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Args a = parse(argc, argv);

#ifdef _OPENMP
    if (a.n_threads > 0) omp_set_num_threads(a.n_threads);
    int omp_threads = omp_get_max_threads();
#else
    int omp_threads = 1;
#endif

    try { a.cfg.validate(); }
    catch (const std::runtime_error& e) {
        if (rank == 0) std::fprintf(stderr, "Config error: %s\n", e.what());
        MPI_Finalize(); return 1;
    }

    if (rank == 0) {
        std::printf("=== MPI+OpenMP Heat Solver ===\n");
        std::printf("  Grid      : %d × %d  (global)\n", a.cfg.Nx, a.cfg.Ny);
        std::printf("  MPI ranks : %d\n",   size);
        std::printf("  OMP threads/rank: %d\n", omp_threads);
        std::printf("  Total cores: %d\n",  size * omp_threads);
        std::printf("  alpha     : %.2e m²/s\n", a.cfg.alpha);
        std::printf("  dt        : %.4e s  (CFL r=%.4f)\n",
                    a.cfg.effective_dt(), a.cfg.r_x() + a.cfg.r_y());
        std::printf("  Steps     : %d\n\n", a.cfg.n_steps);
    }

    double t_total = 0.0;
    heat::TimingStats stats_copy;
    {
        // Solver destructor (MPI_Type_free, MPI_Comm_free) must run
        // before MPI_Finalize(), so we scope it explicitly.
        double t0 = MPI_Wtime();
        heat::MPIFDSolver2D solver(a.cfg, MPI_COMM_WORLD);
        solver.run();
        t_total    = MPI_Wtime() - t0;
        stats_copy = solver.timing();
    }  // ← solver and CartDomain freed here

    if (rank == 0) {
        const auto& s = stats_copy;
        long long   n = (long long)a.cfg.Nx * a.cfg.Ny;

        std::printf("=== Results ===\n");
        std::printf("  Wall time   : %.4f s\n",  t_total);
        std::printf("  Compute     : %.4f s  (%.1f%%)\n",
                    s.compute_s, 100.0 * s.compute_fraction());
        std::printf("  Comm (halo) : %.4f s  (%.1f%%)\n",
                    s.comm_s,    100.0 * s.comm_fraction());
        std::printf("  Comm/Comp ratio: %.3f\n", s.comm_s / std::max(s.compute_s, 1e-12));
        std::printf("  Performance : %.3f GFLOP/s\n",
                    s.gflops(n, a.cfg.n_steps));
        std::printf("  Throughput  : %.2f Mcell-steps/s\n",
                    1e-6 * n * a.cfg.n_steps / t_total);

        if (a.serial_time > 0) {
            double speedup    = a.serial_time / t_total;
            double p_total    = (double)(size * omp_threads);
            double efficiency = speedup / p_total;
            std::printf("\n=== Scaling vs Serial ===\n");
            std::printf("  Serial time   : %.4f s\n",  a.serial_time);
            std::printf("  Speedup       : %.2fx\n",   speedup);
            std::printf("  Efficiency    : %.1f%%\n",  100.0 * efficiency);
            std::printf("  Amdahl ideal  : p=%.0f → max speedup %.2fx\n",
                        p_total, p_total);
        }
    }

    MPI_Finalize();
#else
    (void)argc; (void)argv;
    std::fprintf(stderr, "heat_mpi not compiled with USE_MPI\n");
    return 1;
#endif
    return 0;
}
