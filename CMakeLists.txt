cmake_minimum_required(VERSION 3.20)
project(parallel_heat_solver VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Optimisation flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address,undefined")

# ── Options ─────────────────────────────────────────────────────────────────
option(ENABLE_OPENMP "Enable OpenMP for intra-domain parallelism" ON)
option(ENABLE_MPI    "Enable MPI for domain decomposition"        ON)
option(ENABLE_ASAN   "Enable AddressSanitizer (debug)"            OFF)

# ── Find packages ───────────────────────────────────────────────────────────
if(ENABLE_OPENMP)
  find_package(OpenMP)
  if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found — building without OpenMP support")
    set(ENABLE_OPENMP OFF)
  endif()
endif()

if(ENABLE_MPI)
  find_package(MPI)
  if(NOT MPI_CXX_FOUND)
    message(WARNING "MPI not found — building without MPI support")
    set(ENABLE_MPI OFF)
  endif()
endif()

include_directories(include)

# ── Helper: common compile options ──────────────────────────────────────────
function(target_common_options tgt)
  target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
  if(ENABLE_ASAN)
    target_compile_options(${tgt} PRIVATE -fsanitize=address,undefined)
    target_link_options   (${tgt} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()

# ===========================================================================
# 1. Serial solver
# ===========================================================================
add_executable(heat_serial
  src/serial/main.cpp
  src/serial/fd_solver.cpp
)
target_common_options(heat_serial)
message(STATUS "Target heat_serial: serial FTCS solver")

# ===========================================================================
# 2. OpenMP solver — same source as serial, recompiled with OpenMP
# ===========================================================================
add_executable(heat_openmp
  src/openmp/main.cpp
  src/serial/fd_solver.cpp   # reuse; _OPENMP activates the OMP stencil
)
target_common_options(heat_openmp)
if(ENABLE_OPENMP)
  target_link_libraries(heat_openmp PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(heat_openmp PRIVATE USE_OMP)
  message(STATUS "Target heat_openmp: OpenMP stencil parallelism ENABLED")
else()
  message(STATUS "Target heat_openmp: OpenMP NOT found — serial fallback")
endif()

# ===========================================================================
# 3. MPI + OpenMP hybrid solver
# ===========================================================================
if(ENABLE_MPI)
  add_executable(heat_mpi
    src/mpi/main_mpi.cpp
    src/mpi/domain_decomp.cpp
    src/mpi/halo_exchange.cpp
    src/mpi/fd_solver_mpi.cpp
    src/serial/fd_solver.cpp   # needed for FDSolver2D (serial gather path)
  )
  target_common_options(heat_mpi)
  target_compile_definitions(heat_mpi PRIVATE USE_MPI)
  target_link_libraries(heat_mpi PRIVATE MPI::MPI_CXX)

  if(ENABLE_OPENMP)
    target_link_libraries(heat_mpi PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(heat_mpi PRIVATE USE_OMP)
    message(STATUS "Target heat_mpi: MPI domain decomp + OpenMP stencil ENABLED")
  else()
    message(STATUS "Target heat_mpi: MPI ENABLED, OpenMP NOT found")
  endif()
else()
  message(STATUS "MPI not found — skipping heat_mpi target")
endif()

# ===========================================================================
# 4. Scaling benchmark
# ===========================================================================
add_executable(bench_scaling
  benchmarks/bench_scaling.cpp
  src/serial/fd_solver.cpp
)
target_common_options(bench_scaling)
if(ENABLE_OPENMP)
  target_link_libraries(bench_scaling PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(bench_scaling PRIVATE USE_OMP)
  message(STATUS "Target bench_scaling: with OpenMP")
else()
  message(STATUS "Target bench_scaling: serial only")
endif()

# ===========================================================================
# Summary
# ===========================================================================
message(STATUS "")
message(STATUS "=== parallel-heat-solver build summary ===")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ std    : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenMP     : ${ENABLE_OPENMP}")
message(STATUS "  MPI        : ${ENABLE_MPI}")
message(STATUS "")
