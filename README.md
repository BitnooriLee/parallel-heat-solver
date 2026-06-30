# parallel-heat-solver

> MPI+OpenMP finite-difference solver for 3D heat diffusion equation — domain decomposition with halo exchange

## Overview

A high-performance parallel solver for the 3D heat diffusion equation using finite differences. The solver uses **MPI domain decomposition** across nodes and **OpenMP** for intra-domain parallelism, following the classic halo-exchange pattern.

Results are validated against real TPS (Transient Plane Source) experimental measurements from [HD_Intelligent](https://github.com/BitnooriLee/HD_Intelligent) (private), closing the loop between numerical simulation and physical experiment.

---

## Governing Equation

```
∂T/∂t = α ∇²T

  α  — thermal diffusivity (m²/s)
  T  — temperature field
```

Discretised with explicit finite differences on a structured 3D grid.

---

## HPC Techniques

| Technique | Where |
|---|---|
| **MPI domain decomposition** | 3D grid split across ranks |
| **Halo exchange** | `MPI_Sendrecv` for ghost cell sync |
| **OpenMP** | Stencil loop parallelism within each subdomain |
| **Weak / strong scaling** | Benchmarks with Amdahl analysis |
| **Profiling** | `mpiP` + `gprof`, communication vs computation ratio |

---

## Project Structure

```
parallel-heat-solver/
├── src/
│   ├── serial/          # Baseline 2D/3D FD solver
│   ├── openmp/          # OpenMP stencil loop
│   └── mpi/             # Domain decomposition + halo exchange
├── include/heat/        # Shared headers
├── benchmarks/          # Scaling experiments
├── scripts/             # Python: visualisation, TPS validation
└── docs/                # Scaling plots, validation reports
```

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=ON -DENABLE_OPENMP=ON
make -j$(nproc)
```

### Dependencies

- CMake ≥ 3.20
- C++17 compiler
- Open MPI ≥ 4.0
- OpenMP (bundled with compiler)
- Python 3.x + matplotlib (for visualisation scripts)

---

## Run

```bash
# Serial (2D, 512×512 grid)
./build/heat_serial --grid 512 --steps 1000 --alpha 1.4e-7

# OpenMP (8 threads)
OMP_NUM_THREADS=8 ./build/heat_openmp --grid 512 --steps 1000

# MPI (4 ranks, 2D decomposition)
mpirun -np 4 ./build/heat_mpi --grid 1024 --steps 5000

# Visualise output
python scripts/visualise.py output/T_final.csv
```

---

## Validation

> TODO: Simulation results will be validated against TPS experimental measurements once the solver is implemented.
>
> Planned: compare simulated α and k against real measurements from the HD_Intelligent platform.

---

## Scaling Results

> TODO: Weak/strong scaling benchmarks will be added after MPI implementation is complete.
>
> Planned: 1 / 4 / 8 / 16 cores, grid sizes 256³ to 1024³.
