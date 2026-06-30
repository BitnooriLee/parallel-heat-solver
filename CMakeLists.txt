cmake_minimum_required(VERSION 3.20)
project(parallel_heat_solver VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

option(ENABLE_OPENMP "Enable OpenMP for intra-domain parallelism" ON)
option(ENABLE_MPI    "Enable MPI for domain decomposition" ON)

if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

if(ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

include_directories(include)

# --- Targets ---

# 1. Serial baseline (2D FD)
add_executable(heat_serial
  src/serial/main.cpp
  src/serial/fd_solver.cpp
)

# 2. OpenMP (parallelise stencil loop)
add_executable(heat_openmp
  src/openmp/main.cpp
  src/openmp/fd_solver_omp.cpp
)
if(ENABLE_OPENMP)
  target_link_libraries(heat_openmp PRIVATE OpenMP::OpenMP_CXX)
endif()

# 3. MPI domain decomposition + halo exchange
add_executable(heat_mpi
  src/mpi/main_mpi.cpp
  src/mpi/domain_decomp.cpp
  src/mpi/halo_exchange.cpp
  src/mpi/fd_solver_mpi.cpp
)
if(ENABLE_MPI)
  target_link_libraries(heat_mpi PRIVATE MPI::MPI_CXX)
  if(ENABLE_OPENMP)
    target_link_libraries(heat_mpi PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# --- Benchmarks ---
add_executable(bench_scaling benchmarks/bench_scaling.cpp src/serial/fd_solver.cpp)
