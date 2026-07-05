// fd_solver.cpp — Serial 2D FTCS finite-difference solver
//
//  ∂T/∂t = α ∇²T   on  [0,Lx] × [0,Ly]
//  BC:  Dirichlet T = T_bc on all edges
//  IC:  T = T_cold everywhere, Gaussian hot spot at centre
// ---------------------------------------------------------------------------
#include "heat/fd_solver.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace heat {

// ---------------------------------------------------------------------------
FDSolver2D::FDSolver2D(const Config& cfg, bool use_omp)
    : cfg_(cfg), use_omp_(use_omp)
{
    cfg_.validate();
    T_   .resize(cfg_.Nx * cfg_.Ny, cfg_.T_cold);
    Tnew_.resize(cfg_.Nx * cfg_.Ny, cfg_.T_cold);
}

// ---------------------------------------------------------------------------
// Initial condition: uniform T_cold + Gaussian hot spot at domain centre
// ---------------------------------------------------------------------------
void FDSolver2D::initialize() {
    const int Nx = cfg_.Nx, Ny = cfg_.Ny;
    const double sigma = 0.05 * std::min(cfg_.Lx, cfg_.Ly);  // 5 % of domain
    const double cx = cfg_.Lx / 2.0;
    const double cy = cfg_.Ly / 2.0;

    for (int i = 0; i < Nx; ++i) {
        double x = i * cfg_.dx();
        for (int j = 0; j < Ny; ++j) {
            double y  = j * cfg_.dy();
            double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            T_[i * Ny + j] =
                cfg_.T_cold + (cfg_.T_hot - cfg_.T_cold)
                            * std::exp(-r2 / (2.0 * sigma * sigma));
        }
    }
    apply_boundary(T_);
    Tnew_ = T_;
}

// ---------------------------------------------------------------------------
void FDSolver2D::apply_boundary(std::vector<double>& T) {
    const int Nx = cfg_.Nx, Ny = cfg_.Ny;
    for (int j = 0;  j < Ny; ++j) { T[0        * Ny + j] = cfg_.T_bc;
                                     T[(Nx-1)   * Ny + j] = cfg_.T_bc; }
    for (int i = 1; i < Nx-1; ++i) { T[i * Ny + 0     ] = cfg_.T_bc;
                                      T[i * Ny + (Ny-1)] = cfg_.T_bc; }
}

// ---------------------------------------------------------------------------
// Serial stencil — 7 flops per interior cell
// ---------------------------------------------------------------------------
void FDSolver2D::stencil_serial(const std::vector<double>& T,
                                       std::vector<double>& Tnew) {
    const int    Nx = cfg_.Nx, Ny = cfg_.Ny;
    const double rx = cfg_.r_x(), ry = cfg_.r_y();

    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            double lap_x = T[(i+1)*Ny+j] + T[(i-1)*Ny+j] - 2.0*T[i*Ny+j];
            double lap_y = T[i*Ny+(j+1)] + T[i*Ny+(j-1)] - 2.0*T[i*Ny+j];
            Tnew[i*Ny+j] = T[i*Ny+j] + rx*lap_x + ry*lap_y;
        }
    }
}

// ---------------------------------------------------------------------------
// OpenMP stencil — same logic, parallelised over i-loop
// ---------------------------------------------------------------------------
void FDSolver2D::stencil_omp(const std::vector<double>& T,
                                     std::vector<double>& Tnew) {
#ifdef _OPENMP
    const int    Nx = cfg_.Nx, Ny = cfg_.Ny;
    const double rx = cfg_.r_x(), ry = cfg_.r_y();

    #pragma omp parallel for schedule(static) default(none) \
        shared(T, Tnew) firstprivate(Nx, Ny, rx, ry)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            double lap_x = T[(i+1)*Ny+j] + T[(i-1)*Ny+j] - 2.0*T[i*Ny+j];
            double lap_y = T[i*Ny+(j+1)] + T[i*Ny+(j-1)] - 2.0*T[i*Ny+j];
            Tnew[i*Ny+j] = T[i*Ny+j] + rx*lap_x + ry*lap_y;
        }
    }
#else
    stencil_serial(T, Tnew);
#endif
}

// ---------------------------------------------------------------------------
void FDSolver2D::write_snapshot(int step) const {
    if (cfg_.out_every <= 0) return;
    std::filesystem::create_directories(cfg_.out_dir);

    // Binary dump: raw double array, row-major
    std::string fname = cfg_.out_dir + "/T_" + std::to_string(step) + ".bin";
    std::ofstream f(fname, std::ios::binary);
    f.write(reinterpret_cast<const char*>(T_.data()),
            static_cast<std::streamsize>(T_.size() * sizeof(double)));

    // Metadata on first write
    if (step == 0) {
        std::ofstream meta(cfg_.out_dir + "/metadata.txt");
        meta << "Nx="    << cfg_.Nx          << "\n"
             << "Ny="    << cfg_.Ny          << "\n"
             << "dx="    << cfg_.dx()        << "\n"
             << "dy="    << cfg_.dy()        << "\n"
             << "dt="    << cfg_.effective_dt() << "\n"
             << "alpha=" << cfg_.alpha       << "\n"
             << "T_cold=" << cfg_.T_cold     << "\n"
             << "T_hot="  << cfg_.T_hot      << "\n";
    }
}

// ---------------------------------------------------------------------------
void FDSolver2D::run(int steps) {
    if (steps < 0) steps = cfg_.n_steps;
    initialize();

    using Clock = std::chrono::high_resolution_clock;
    auto t_start = Clock::now();

    if (cfg_.out_every > 0) write_snapshot(0);

    auto stencil = use_omp_
                 ? &FDSolver2D::stencil_omp
                 : &FDSolver2D::stencil_serial;

    for (int s = 1; s <= steps; ++s) {
        auto tc0 = Clock::now();
        (this->*stencil)(T_, Tnew_);
        auto tc1 = Clock::now();

        std::swap(T_, Tnew_);
        apply_boundary(T_);

        stats_.compute_s +=
            std::chrono::duration<double>(tc1 - tc0).count();

        if (cfg_.out_every > 0 && s % cfg_.out_every == 0)
            write_snapshot(s);
    }

    stats_.total_s = std::chrono::duration<double>(Clock::now() - t_start).count();
    stats_.io_s    = stats_.total_s - stats_.compute_s;
}

}  // namespace heat
