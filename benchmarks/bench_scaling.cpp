// bench_scaling.cpp — Weak / Strong scaling benchmark harness
//
//  Measures wall-clock time of the serial and OpenMP heat solvers across
//  multiple grid sizes and thread counts.  Outputs CSV for plotting.
//
//  For MPI scaling, run heat_mpi directly and collect times externally
//  (see scripts/scaling_analysis.py for automated data collection).
//
//  Usage:
//    ./bench_scaling [--mode strong|weak] [--max-threads N] [--out FILE]
//
//  Strong scaling:
//    Fixed grid size (default: 1024×1024), vary OMP threads 1→max.
//    Measures: speedup, efficiency.
//
//  Weak scaling:
//    Grid size scales as sqrt(threads) × base so work per thread is constant.
//    Measures: efficiency (should be ~1.0 for perfectly parallel code).
//
//  Output columns:  threads, grid, n_cells, time_s, speedup, efficiency, gflops
// ---------------------------------------------------------------------------
#include "heat/fd_solver.hpp"
#include "heat/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
static double run_once(int Nx, int Ny, int threads, int steps) {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    heat::Config cfg;
    cfg.Nx      = Nx;
    cfg.Ny      = Ny;
    cfg.n_steps = steps;
    cfg.out_every = 0;  // no I/O during benchmark
    cfg.alpha   = 1.4e-7;
    cfg.T_cold  = 300.0;
    cfg.T_hot   = 500.0;
    cfg.T_bc    = 300.0;

    bool use_omp = (threads > 1);
    heat::FDSolver2D solver(cfg, use_omp);
    solver.run();
    return solver.timing().total_s;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Defaults
    std::string mode     = "strong";
    int max_threads      = 8;
    int base_N           = 1024;  // strong scaling grid
    int base_N_weak      = 512;   // weak scaling base (per thread)
    int steps            = 500;
    std::string out_file = "bench_results.csv";

    for (int i = 1; i < argc; ++i) {
        auto nxt = [&]() -> const char* {
            if (i+1>=argc) { std::fprintf(stderr,"Missing arg\n"); std::exit(1); }
            return argv[++i];
        };
        if      (!std::strcmp(argv[i],"--mode"))       mode        = nxt();
        else if (!std::strcmp(argv[i],"--max-threads"))max_threads = std::atoi(nxt());
        else if (!std::strcmp(argv[i],"--grid"))       base_N      = std::atoi(nxt());
        else if (!std::strcmp(argv[i],"--steps"))      steps       = std::atoi(nxt());
        else if (!std::strcmp(argv[i],"--out"))        out_file    = nxt();
        else { std::fprintf(stderr,"Unknown: %s\n",argv[i]); std::exit(1); }
    }

#ifdef _OPENMP
    int hw_threads = omp_get_max_threads();
    max_threads = std::min(max_threads, hw_threads);
#else
    max_threads = 1;
    if (mode == "strong") std::fprintf(stderr, "WARNING: no OpenMP, only serial run.\n");
#endif

    // Generate thread counts: 1, 2, 4, 8, … up to max_threads
    std::vector<int> thread_counts = {1};
    for (int t = 2; t <= max_threads; t *= 2) thread_counts.push_back(t);
    if (thread_counts.back() != max_threads && max_threads > 1)
        thread_counts.push_back(max_threads);

    std::FILE* csv = std::fopen(out_file.c_str(), "w");
    if (!csv) { std::perror("Cannot open output file"); return 1; }
    std::fprintf(csv, "mode,threads,Nx,Ny,n_cells,steps,time_s,speedup,efficiency,gflops\n");

    std::printf("=== Scaling Benchmark  mode=%s ===\n\n", mode.c_str());
    std::printf("%-10s %-8s %-8s %-12s %-10s %-10s %-8s\n",
                "Threads", "Grid", "Time(s)", "Speedup", "Efficiency", "GFLOP/s", "Mode");
    std::printf("%s\n", std::string(72,'-').c_str());

    double t1 = -1.0;  // serial baseline time

    for (int threads : thread_counts) {
        int Nx, Ny;

        if (mode == "strong") {
            // Fixed problem size
            Nx = Ny = base_N;
        } else {
            // Weak: total cells ∝ threads  (base_N² per thread)
            int side = static_cast<int>(base_N_weak * std::sqrt((double)threads));
            Nx = Ny = side;
        }

        // Warm-up run (smaller) to prime caches
        run_once(std::min(Nx/4, 128), std::min(Ny/4, 128), threads, 50);

        // Timed run (average of 3 repetitions)
        double total = 0.0;
        int reps = (Nx <= 512) ? 3 : 1;
        for (int r = 0; r < reps; ++r)
            total += run_once(Nx, Ny, threads, steps);
        double t = total / reps;

        if (threads == 1 && mode == "strong") t1 = t;

        double speedup    = (t1 > 0 && mode == "strong") ? t1 / t : 1.0;
        double efficiency = speedup / threads;
        long long n       = (long long)(Nx-2)*(Ny-2);
        // 7 flops / cell / step
        double gflops     = (t > 0) ? 7.0e-9 * n * steps / t : 0.0;

        std::printf("%-10d %-8d %-8.3f %-12.2f %-10.2f %-8.3f %s\n",
                    threads, Nx, t, speedup, efficiency, gflops, mode.c_str());

        std::fprintf(csv, "%s,%d,%d,%d,%lld,%d,%.6f,%.4f,%.4f,%.4f\n",
                     mode.c_str(), threads, Nx, Ny, n, steps, t,
                     speedup, efficiency, gflops);
    }

    std::fclose(csv);
    std::printf("\nResults written to: %s\n", out_file.c_str());
    return 0;
}
