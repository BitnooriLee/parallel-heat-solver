// halo_exchange.cpp — Ghost-cell synchronisation via MPI_Sendrecv
//
//  Two exchange passes:
//    1.  North / South (i-direction):
//          Row data is contiguous in row-major layout → plain MPI_DOUBLE buffer
//
//    2.  East / West (j-direction):
//          Column data is strided → use the pre-registered MPI_Type_vector
//          (CartDomain::col_type)
//
//  MPI_Sendrecv is used in place of separate MPI_Send/MPI_Recv to avoid
//  potential deadlocks and reduce latency by overlapping send+recv.
//
//  Timing is injected by the caller (MPIFDSolver2D::halo_exchange) via
//  MPI_Wtime.
// ---------------------------------------------------------------------------
#include "heat/fd_solver.hpp"
#include "heat/domain.hpp"

#ifdef USE_MPI
#include <mpi.h>
#include <vector>

namespace heat {

// ---------------------------------------------------------------------------
// Called from MPIFDSolver2D::halo_exchange().
// Modifies the ghost cells of `T` in-place.
// ---------------------------------------------------------------------------
void mpi_halo_exchange(std::vector<double>& T,
                       const CartDomain&    dom)
{
    const int lnx    = dom.local_nx;
    const int lny    = dom.local_ny;
    const int stride = dom.stride;   // local_ny + 2
    const int tag    = 42;

    // ── 1.  North / South row exchange ───────────────────────────────────
    //  Each row has `stride` contiguous doubles.
    //
    //  Send my top-interior row (i = lnx)    to north  → fills north's south ghost
    //  Recv into my north ghost (i = lnx+1)  from north's south-interior row
    //  Send my bot-interior row (i = 1  )    to south  → fills south's north ghost
    //  Recv into my south ghost (i = 0  )    from south's top-interior row

    MPI_Sendrecv(
        &T[(lnx    ) * stride + 0], stride, MPI_DOUBLE, dom.north, tag,
        &T[(lnx + 1) * stride + 0], stride, MPI_DOUBLE, dom.north, tag,
        dom.cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(
        &T[1 * stride + 0], stride, MPI_DOUBLE, dom.south, tag,
        &T[0 * stride + 0], stride, MPI_DOUBLE, dom.south, tag,
        dom.cart_comm, MPI_STATUS_IGNORE);

    // ── 2.  East / West column exchange ──────────────────────────────────
    //  col_type = MPI_Type_vector(lnx+2, 1, stride, MPI_DOUBLE)
    //
    //  Send my right-interior col (j = lny  ) to east  → fills east's west ghost
    //  Recv into my east ghost    (j = lny+1) from east's left-interior col
    //  Send my left-interior col  (j = 1    ) to west  → fills west's east ghost
    //  Recv into my west ghost    (j = 0    ) from west's right-interior col

    MPI_Sendrecv(
        &T[0 * stride + lny    ], 1, dom.col_type, dom.east, tag,
        &T[0 * stride + lny + 1], 1, dom.col_type, dom.east, tag,
        dom.cart_comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(
        &T[0 * stride + 1], 1, dom.col_type, dom.west, tag,
        &T[0 * stride + 0], 1, dom.col_type, dom.west, tag,
        dom.cart_comm, MPI_STATUS_IGNORE);
}

}  // namespace heat
#endif  // USE_MPI
