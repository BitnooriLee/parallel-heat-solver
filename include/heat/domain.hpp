// domain.hpp — MPI Cartesian domain decomposition structures
#pragma once
#include <array>
#include "types.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace heat {

// ---------------------------------------------------------------------------
// CartDomain: describes how the 2-D global grid is partitioned across ranks.
//
//  Global grid: Nx × Ny
//  Process grid: dims[0] × dims[1]  (auto-computed by MPI_Dims_create)
//
//  Each rank (coords[0], coords[1]) owns an interior sub-domain of size:
//      local_nx × local_ny
//  plus a 1-cell-wide ghost layer on every side (total: (lnx+2)×(lny+2))
//
//  Ghost-cell exchange pattern:
//      NORTH/SOUTH (i-direction) : rows — contiguous in row-major layout
//      EAST /WEST  (j-direction) : cols — strided, use MPI_Type_vector
// ---------------------------------------------------------------------------
struct CartDomain {
#ifdef USE_MPI
    MPI_Comm cart_comm  = MPI_COMM_NULL;
    int      rank       = 0;
    int      size       = 1;
    int      coords[2]  = {0, 0};   // (rx, ry) in process grid
    int      dims[2]    = {1, 1};   // process-grid shape
    int      periods[2] = {0, 0};   // non-periodic (Dirichlet BC)

    // Neighbour ranks (MPI_PROC_NULL when on a physical boundary)
    int north = MPI_PROC_NULL;   // rx + 1
    int south = MPI_PROC_NULL;   // rx - 1
    int east  = MPI_PROC_NULL;   // ry + 1
    int west  = MPI_PROC_NULL;   // ry - 1

    // Local sub-domain size (interior cells only, excl. ghost cells)
    int local_nx = 0;
    int local_ny = 0;

    // Global index of the first interior cell this rank owns
    int offset_x = 0;
    int offset_y = 0;

    // MPI derived type for exchanging a single ghost column
    // (one element per row, stride = local_ny+2)
    MPI_Datatype col_type = MPI_DATATYPE_NULL;

    // Stride of the local 2-D array (local_ny + 2)
    int stride = 0;

    // ── Lifecycle ────────────────────────────────────────────────────────────
    void init(const Config& cfg, MPI_Comm world = MPI_COMM_WORLD);
    void finalize();

    // ── Convenience ──────────────────────────────────────────────────────────
    // True when this rank sits on the given global boundary
    bool on_south_boundary() const { return south == MPI_PROC_NULL; }
    bool on_north_boundary() const { return north == MPI_PROC_NULL; }
    bool on_west_boundary()  const { return west  == MPI_PROC_NULL; }
    bool on_east_boundary()  const { return east  == MPI_PROC_NULL; }

    // Flat index into (local_nx+2)×(local_ny+2) ghost-padded array
    int idx(int i, int j) const { return i * stride + j; }
#else
    // Dummy non-MPI fallback so headers compile without MPI
    int rank       = 0;
    int size       = 1;
    int local_nx   = 0;
    int local_ny   = 0;
    int offset_x   = 0;
    int offset_y   = 0;
    int stride     = 0;
#endif
};

}  // namespace heat
