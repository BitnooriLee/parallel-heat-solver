// fd_solver_mpi.cpp — MPI + OpenMP hybrid heat solver implementation
//
//  Domain decomposition:  2D Cartesian (CartDomain)
//  Halo exchange:         MPI_Sendrecv (mpi_halo_exchange)
//  Stencil:               FTCS, OpenMP-parallelised inner loop
//
//  Ghost-padded local array layout  [(lnx+2) × (lny+2)]:
//
//    i=0          : south ghost row  (from south neighbour)
//    i=1 .. lnx   : interior rows
//    i=lnx+1      : north ghost row  (from north neighbour)
//    j=0          : west ghost col   (from west  neighbour)
//    j=1 .. lny   : interior cols
//    j=lny+1      : east ghost col   (from east  neighbour)
//
//  Boundary ranks apply Dirichlet BC directly into ghost slots.
// ---------------------------------------------------------------------------
#include "heat/fd_solver.hpp"
#include "heat/domain.hpp"

#ifdef USE_MPI
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace heat {

// Forward declaration (implemented in halo_exchange.cpp)
void mpi_halo_exchange(std::vector<double>& T, const CartDomain& dom);

// ===========================================================================
// Constructor
// ===========================================================================
MPIFDSolver2D::MPIFDSolver2D(const Config& cfg, MPI_Comm world)
    : cfg_(cfg)
{
    cfg_.validate();
    dom_.init(cfg_, world);

    lnx_    = dom_.local_nx;
    lny_    = dom_.local_ny;
    stride_ = dom_.stride;  // lny_ + 2

    int sz = (lnx_ + 2) * stride_;
    T_   .assign(sz, cfg_.T_cold);
    Tnew_.assign(sz, cfg_.T_cold);
}

MPIFDSolver2D::~MPIFDSolver2D() {
    dom_.finalize();
}

// ===========================================================================
// Initial condition: Gaussian hot spot centred on the global domain
// ===========================================================================
void MPIFDSolver2D::initialize() {
    const double sigma = 0.05 * std::min(cfg_.Lx, cfg_.Ly);
    const double cx    = cfg_.Lx / 2.0;
    const double cy    = cfg_.Ly / 2.0;
    const double dx    = cfg_.dx();
    const double dy    = cfg_.dy();

    for (int i = 1; i <= lnx_; ++i) {
        double x = (dom_.offset_x + i - 1) * dx;
        for (int j = 1; j <= lny_; ++j) {
            double y  = (dom_.offset_y + j - 1) * dy;
            double r2 = (x - cx)*(x - cx) + (y - cy)*(y - cy);
            T(i, j)   = cfg_.T_cold
                      + (cfg_.T_hot - cfg_.T_cold)
                      * std::exp(-r2 / (2.0 * sigma * sigma));
        }
    }
    apply_local_boundary();
    Tnew_ = T_;
}

// ===========================================================================
// Dirichlet BC: set boundary ghost cells to T_bc
// Only ranks on the global domain edge apply the BC.
// ===========================================================================
void MPIFDSolver2D::apply_local_boundary() {
    // South boundary (i = 0 ghost row → i = 1 interior becomes boundary)
    if (dom_.on_south_boundary())
        for (int j = 0; j <= lny_ + 1; ++j) T(0,       j) = cfg_.T_bc;

    // North boundary
    if (dom_.on_north_boundary())
        for (int j = 0; j <= lny_ + 1; ++j) T(lnx_+1,  j) = cfg_.T_bc;

    // West boundary
    if (dom_.on_west_boundary())
        for (int i = 0; i <= lnx_ + 1; ++i) T(i,       0) = cfg_.T_bc;

    // East boundary
    if (dom_.on_east_boundary())
        for (int i = 0; i <= lnx_ + 1; ++i) T(i, lny_+1)  = cfg_.T_bc;
}

// ===========================================================================
// FTCS stencil — OpenMP-parallelised over i
// ===========================================================================
void MPIFDSolver2D::stencil_update() {
    const double rx = cfg_.r_x();
    const double ry = cfg_.r_y();

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) default(none) \
        shared(rx, ry)
#endif
    for (int i = 1; i <= lnx_; ++i) {
        for (int j = 1; j <= lny_; ++j) {
            double lap_x = T(i+1,j) + T(i-1,j) - 2.0*T(i,j);
            double lap_y = T(i,j+1) + T(i,j-1) - 2.0*T(i,j);
            Tn(i, j)     = T(i,j) + rx*lap_x + ry*lap_y;
        }
    }
}

// ===========================================================================
// Write binary snapshot (only rank 0 writes the gathered global field)
// ===========================================================================
void MPIFDSolver2D::write_snapshot(int step) const {
    if (cfg_.out_every <= 0) return;
    auto global = const_cast<MPIFDSolver2D*>(this)->gather_global();
    if (dom_.rank != 0) return;

    std::filesystem::create_directories(cfg_.out_dir);
    std::string fname = cfg_.out_dir + "/T_" + std::to_string(step) + ".bin";
    std::ofstream f(fname, std::ios::binary);
    f.write(reinterpret_cast<const char*>(global.data()),
            static_cast<std::streamsize>(global.size() * sizeof(double)));

    if (step == 0) {
        std::ofstream meta(cfg_.out_dir + "/metadata.txt");
        meta << "Nx="     << cfg_.Nx            << "\n"
             << "Ny="     << cfg_.Ny            << "\n"
             << "dx="     << cfg_.dx()          << "\n"
             << "dy="     << cfg_.dy()          << "\n"
             << "dt="     << cfg_.effective_dt()<< "\n"
             << "alpha="  << cfg_.alpha         << "\n"
             << "T_cold=" << cfg_.T_cold        << "\n"
             << "T_hot="  << cfg_.T_hot         << "\n"
             << "np="     << dom_.size          << "\n"
             << "dims="   << dom_.dims[0] << "x" << dom_.dims[1] << "\n";
    }
}

// ===========================================================================
// Gather global field to rank 0 using MPI_Gatherv
// Each rank sends its local interior (lnx × lny) in row-major order.
// Rank 0 reassembles using offset_x / offset_y.
// ===========================================================================
std::vector<double> MPIFDSolver2D::gather_global() const {
    // Pack local interior into a contiguous send buffer
    std::vector<double> send_buf(lnx_ * lny_);
    for (int i = 0; i < lnx_; ++i)
        for (int j = 0; j < lny_; ++j)
            send_buf[i * lny_ + j] = T(i+1, j+1);

    // Gather sizes from every rank
    int send_cnt = lnx_ * lny_;
    std::vector<int> recv_cnts(dom_.size), displs(dom_.size);
    MPI_Gather(&send_cnt, 1, MPI_INT,
               recv_cnts.data(), 1, MPI_INT,
               0, dom_.cart_comm);

    std::vector<double> recv_buf;
    if (dom_.rank == 0) {
        int total = 0;
        for (int r = 0; r < dom_.size; ++r) { displs[r] = total; total += recv_cnts[r]; }
        recv_buf.resize(total);
    }

    MPI_Gatherv(send_buf.data(), send_cnt, MPI_DOUBLE,
                recv_buf.data(), recv_cnts.data(), displs.data(), MPI_DOUBLE,
                0, dom_.cart_comm);

    // Rank 0: scatter into the global Nx×Ny array
    std::vector<double> global;
    if (dom_.rank == 0) {
        global.assign(cfg_.Nx * cfg_.Ny, cfg_.T_bc);

        // For each rank, recover its dims and offset via coords
        for (int r = 0; r < dom_.size; ++r) {
            int rcoords[2];
            MPI_Cart_coords(dom_.cart_comm, r, 2, rcoords);

            int bnx   = cfg_.Nx / dom_.dims[0];
            int rem_x = cfg_.Nx % dom_.dims[0];
            int bny   = cfg_.Ny / dom_.dims[1];
            int rem_y = cfg_.Ny % dom_.dims[1];

            int rlnx   = bnx + (rcoords[0] < rem_x ? 1 : 0);
            int rlny   = bny + (rcoords[1] < rem_y ? 1 : 0);
            int roffx  = rcoords[0]*bnx + std::min(rcoords[0], rem_x);
            int roffy  = rcoords[1]*bny + std::min(rcoords[1], rem_y);

            const double* src = recv_buf.data() + displs[r];
            for (int i = 0; i < rlnx; ++i)
                for (int j = 0; j < rlny; ++j)
                    global[(roffx + i) * cfg_.Ny + (roffy + j)] =
                        src[i * rlny + j];
        }
    }
    return global;
}

// ===========================================================================
// Main time-stepping loop
// ===========================================================================
void MPIFDSolver2D::run(int steps) {
    if (steps < 0) steps = cfg_.n_steps;
    initialize();

    if (cfg_.out_every > 0) write_snapshot(0);

    for (int s = 1; s <= steps; ++s) {
        // ── Halo exchange ─────────────────────────────────────────────────
        double t_comm0 = MPI_Wtime();
        mpi_halo_exchange(T_, dom_);
        stats_.comm_s += MPI_Wtime() - t_comm0;

        // ── Apply BC (overwrites stale ghost cells on physical boundaries) ─
        apply_local_boundary();

        // ── Stencil update ────────────────────────────────────────────────
        double t_comp0 = MPI_Wtime();
        stencil_update();
        stats_.compute_s += MPI_Wtime() - t_comp0;

        std::swap(T_, Tnew_);

        if (cfg_.out_every > 0 && s % cfg_.out_every == 0)
            write_snapshot(s);
    }

    // Reduce timing to rank 0 (report max across ranks = bottleneck)
    double local_total   = stats_.comm_s + stats_.compute_s;
    double local_comm    = stats_.comm_s;
    double local_compute = stats_.compute_s;

    MPI_Reduce(&local_total,   &stats_.total_s,   1, MPI_DOUBLE, MPI_MAX, 0, dom_.cart_comm);
    MPI_Reduce(&local_comm,    &stats_.comm_s,    1, MPI_DOUBLE, MPI_MAX, 0, dom_.cart_comm);
    MPI_Reduce(&local_compute, &stats_.compute_s, 1, MPI_DOUBLE, MPI_MAX, 0, dom_.cart_comm);
}

}  // namespace heat
#endif  // USE_MPI
