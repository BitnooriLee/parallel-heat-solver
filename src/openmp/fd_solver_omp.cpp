// fd_solver_omp.cpp — OpenMP-parallelised FD solver
//
//  This is a thin wrapper: FDSolver2D already contains both a serial and an
//  OpenMP stencil method (stencil_serial / stencil_omp).  The OpenMP target
//  is built with USE_OMP and _OPENMP defined, so the same source file is
//  compiled differently for the two targets.
//
//  The inner loop is decorated with:
//    #pragma omp parallel for schedule(static) — static chunking
//  which gives each thread a contiguous band of rows → excellent cache reuse.
// ---------------------------------------------------------------------------
// No new code here — OpenMP solver shares src/serial/fd_solver.cpp.
// The CMakeLists target 'heat_openmp' links this file and passes
// -DUSE_OMP to enable the OpenMP path inside fd_solver.cpp.

// This translation unit intentionally left as a reminder that the OpenMP
// parallelism lives inside FDSolver2D::stencil_omp() in fd_solver.cpp.
// The heat_openmp binary is just heat_serial recompiled with OpenMP enabled.
