// fd_solver.hpp — Finite-difference solver interfaces
//
//  Three concrete solvers are provided:
//    FDSolver2D         – serial (or optionally OpenMP) FTCS solver
//    MPIFDSolver2D      – MPI+OpenMP hybrid solver with ghost-cell exchange
//
//  All use the same explicit FTCS (Forward-Time Centred-Space) scheme:
//
//    T^{n+1}_{i,j} = T^n_{i,j}
//                  + r_x * (T^n_{i+1,j} + T^n_{i-1,j} - 2 T^n_{i,j})
//                  + r_y * (T^n_{i,j+1} + T^n_{i,j-1} - 2 T^n_{i,j})
//
//  Stability: r_x + r_y ≤ 0.5
// ---------------------------------------------------------------------------
#pragma once
#include <vector>
#include <cstdio>
#include "types.hpp"
#include "domain.hpp"

namespace heat {

// ===========================================================================
// Serial / OpenMP solver
// ===========================================================================
class FDSolver2D {
public:
    explicit FDSolver2D(const Config& cfg, bool use_omp = false);

    void initialize();
    void run(int steps = -1);   // -1 → use cfg.n_steps

    const std::vector<double>& temperature() const { return T_; }
    const TimingStats&         timing()      const { return stats_; }
    const Config&              config()      const { return cfg_; }

    // Direct access for benchmarks (avoids a copy)
    double T_at(int i, int j) const { return T_[i * cfg_.Ny + j]; }

private:
    Config             cfg_;
    bool               use_omp_;
    std::vector<double> T_;
    std::vector<double> Tnew_;
    TimingStats        stats_;

    void apply_boundary(std::vector<double>& T);
    void stencil_serial(const std::vector<double>& T,
                              std::vector<double>& Tnew);
    void stencil_omp   (const std::vector<double>& T,
                              std::vector<double>& Tnew);
    void write_snapshot(int step) const;
};

// ===========================================================================
// MPI + OpenMP hybrid solver
// ===========================================================================
#ifdef USE_MPI
class MPIFDSolver2D {
public:
    MPIFDSolver2D(const Config& cfg, MPI_Comm world = MPI_COMM_WORLD);
    ~MPIFDSolver2D();

    void initialize();
    void run(int steps = -1);

    const TimingStats& timing() const { return stats_; }

    // Gather full temperature field to rank 0 (for validation / output)
    // Returns non-empty vector only on rank 0.
    std::vector<double> gather_global() const;

private:
    Config      cfg_;
    CartDomain  dom_;
    int         lnx_;    // local_nx (dom_.local_nx shorthand)
    int         lny_;    // local_ny
    int         stride_; // local_ny + 2

    std::vector<double> T_;    // ghost-padded local field
    std::vector<double> Tnew_;
    TimingStats stats_;

    // Helpers
    void halo_exchange();
    void apply_local_boundary();
    void stencil_update();
    void write_snapshot(int step) const;

    double& T   (int i, int j)       { return T_   [i * stride_ + j]; }
    double& Tn  (int i, int j)       { return Tnew_[i * stride_ + j]; }
    double  T   (int i, int j) const { return T_   [i * stride_ + j]; }
};
#endif

}  // namespace heat
