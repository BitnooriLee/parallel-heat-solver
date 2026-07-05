// domain_decomp.cpp — 2D Cartesian MPI domain decomposition
//
//  Creates an MPI Cartesian communicator, computes each rank's local
//  subdomain dimensions and global offsets, and registers an MPI derived
//  datatype for efficient column (j-direction) ghost-cell exchange.
//
//  Process grid layout (dims[0]=px rows, dims[1]=py cols):
//
//    rank(px-1, 0) … rank(px-1, py-1)   ← north
//          …                …
//    rank(0,   0) … rank(0,   py-1)     ← south
//
//  coords[0] = row-rank (x / north-south direction)
//  coords[1] = col-rank (y / east-west  direction)
// ---------------------------------------------------------------------------
#include "heat/domain.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>

#ifdef USE_MPI

namespace heat {

void CartDomain::init(const Config& cfg, MPI_Comm world) {
    MPI_Comm_size(world, &size);
    MPI_Comm_rank(world, &rank);

    // ── 1.  Choose process grid dims[0] × dims[1] ─────────────────────────
    dims[0] = dims[1] = 0;
    MPI_Dims_create(size, 2, dims);
    // Prefer more rows than columns to minimise strided column exchanges
    if (dims[0] < dims[1]) std::swap(dims[0], dims[1]);

    periods[0] = periods[1] = 0;  // non-periodic (Dirichlet on all sides)

    // ── 2.  Create Cartesian communicator ────────────────────────────────
    MPI_Cart_create(world, 2, dims, periods, /*reorder=*/1, &cart_comm);
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // ── 3.  Discover neighbours (MPI_PROC_NULL at physical boundaries) ────
    MPI_Cart_shift(cart_comm, 0, +1, &south, &north);
    MPI_Cart_shift(cart_comm, 1, +1, &west,  &east );

    // ── 4.  Compute local subdomain size ─────────────────────────────────
    //  Distribute remainder evenly (last rank in each direction gets extras)
    int base_nx  = cfg.Nx / dims[0];
    int rem_nx   = cfg.Nx % dims[0];
    int base_ny  = cfg.Ny / dims[1];
    int rem_ny   = cfg.Ny % dims[1];

    // Assign one extra cell to the higher-indexed ranks that carry remainder
    local_nx = base_nx + (coords[0] < rem_nx ? 1 : 0);
    local_ny = base_ny + (coords[1] < rem_ny ? 1 : 0);

    // Global offset of this rank's first interior cell
    offset_x = coords[0] * base_nx + std::min(coords[0], rem_nx);
    offset_y = coords[1] * base_ny + std::min(coords[1], rem_ny);

    // Stride for the ghost-padded 2-D local array (local_ny + 2 per row)
    stride = local_ny + 2;

    // ── 5.  Register MPI column datatype for east/west exchange ──────────
    //  Column: one element per row, stride = (local_ny + 2) doubles
    MPI_Type_vector(
        local_nx + 2,      // count    : rows (incl. ghost rows)
        1,                 // blocklength
        stride,            // stride in elements
        MPI_DOUBLE,
        &col_type);
    MPI_Type_commit(&col_type);
}

void CartDomain::finalize() {
    if (col_type != MPI_DATATYPE_NULL) {
        MPI_Type_free(&col_type);
        col_type = MPI_DATATYPE_NULL;
    }
    if (cart_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&cart_comm);
        cart_comm = MPI_COMM_NULL;
    }
}

}  // namespace heat
#endif  // USE_MPI
