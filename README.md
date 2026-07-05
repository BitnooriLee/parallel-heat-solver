# parallel-heat-solver

> MPI + OpenMP hybrid finite-difference solver for the 2D heat diffusion equation — with Cartesian domain decomposition, ghost-cell halo exchange, and TPS experimental validation.

---

## Governing Equation

```
∂T/∂t = α ∇²T     on Ω = [0,Lx] × [0,Ly]
T = T_bc            on ∂Ω  (Dirichlet)
```

Discretised with the **explicit FTCS** (Forward-Time Centred-Space) scheme:

```
T^{n+1}_{i,j} = T^n_{i,j}
              + r_x (T^n_{i+1,j} + T^n_{i-1,j} − 2 T^n_{i,j})
              + r_y (T^n_{i,j+1} + T^n_{i,j-1} − 2 T^n_{i,j})

  r_x = α·Δt/Δx²,  r_y = α·Δt/Δy²
  Stability: r_x + r_y ≤ 0.5
```

---

## HPC Architecture

```
┌─────────────────────────────────────────────────┐
│         MPI Cartesian Domain Decomposition      │
│  ┌──────┬──────┬──────┬──────┐                  │
│  │rank0 │rank1 │rank2 │rank3 │  ← 4 MPI ranks   │
│  │      │ghost │ghost │      │  halo exchange   │
│  └──────┴──────┴──────┴──────┘  MPI_Sendrecv    │
│       ↑ each subdomain: OpenMP parallel for     │
└─────────────────────────────────────────────────┘
```

| Technique | Implementation |
|---|---|
| **MPI domain decomposition** | 2D Cartesian (`MPI_Cart_create`), `MPI_Dims_create` |
| **Halo exchange** | `MPI_Sendrecv` — rows (contiguous) + `MPI_Type_vector` cols |
| **OpenMP** | `#pragma omp parallel for schedule(static)` over stencil rows |
| **Profiling** | `MPI_Wtime` comm/compute ratio; `gprof`-ready build |
| **Scaling** | Weak / strong; Amdahl fit; efficiency plots |
| **Validation** | TPS analytical + HD_Intelligent experimental comparison |

---

## Project Structure

```
parallel-heat-solver/
├── include/heat/
│   ├── types.hpp          # Config, TimingStats
│   ├── domain.hpp         # CartDomain (MPI 2D decomposition)
│   └── fd_solver.hpp      # FDSolver2D / MPIFDSolver2D interfaces
├── src/
│   ├── serial/
│   │   ├── fd_solver.cpp  # Serial + OpenMP FTCS stencil
│   │   └── main.cpp
│   ├── openmp/
│   │   ├── fd_solver_omp.cpp   # (stub — uses serial/fd_solver.cpp)
│   │   └── main.cpp
│   └── mpi/
│       ├── domain_decomp.cpp   # MPI Cartesian domain setup
│       ├── halo_exchange.cpp   # Ghost-cell sync (MPI_Sendrecv)
│       ├── fd_solver_mpi.cpp   # MPI+OpenMP solver
│       └── main_mpi.cpp
├── benchmarks/
│   └── bench_scaling.cpp  # Weak/strong scaling harness
├── scripts/
│   ├── visualise.py       # Heatmap, animation, profiles
│   ├── validate_tps.py    # TPS analytical + experimental validation
│   └── scaling_analysis.py # Amdahl curves, efficiency plots
├── docs/                  # Generated plots & reports
└── CMakeLists.txt
```

---

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=ON -DENABLE_OPENMP=ON
make -j$(nproc)
```

Built targets: `heat_serial`, `heat_openmp`, `heat_mpi`, `bench_scaling`.

### Dependencies

| Package | Version | Notes |
|---|---|---|
| CMake | ≥ 3.20 | |
| C++17 | GCC ≥ 9 / Clang ≥ 10 | |
| Open MPI | ≥ 4.0 | `brew install open-mpi` on macOS |
| OpenMP | bundled with compiler | `-fopenmp` |
| Python | ≥ 3.10 | `pip install numpy scipy matplotlib pandas` |

---

## Run

```bash
# Serial baseline (256×256, 2000 steps)
./build/heat_serial --nx 512 --steps 2000 --alpha 1.4e-7

# OpenMP (8 threads)
OMP_NUM_THREADS=8 ./build/heat_openmp --nx 512 --steps 2000 --threads 8

# MPI (4 ranks) + OpenMP (2 threads each = 8 total cores)
mpirun -np 4 ./build/heat_mpi --nx 1024 --steps 5000 \
    --threads 2 --out 500 --outdir output/

# MPI — report speedup vs serial baseline
mpirun -np 8 ./build/heat_mpi --nx 1024 --steps 2000 \
    --serial-time 12.4 --profile
```

---

## Benchmark & Scaling

```bash
# Strong scaling (fixed 1024² grid, 1→8 threads)
./build/bench_scaling --mode strong --grid 1024 --max-threads 8 \
    --steps 500 --out bench_strong.csv

# Weak scaling
./build/bench_scaling --mode weak --max-threads 8 \
    --steps 500 --out bench_weak.csv

# Plot results
python scripts/scaling_analysis.py --csv bench_strong.csv --save-dir docs/
python scripts/scaling_analysis.py --csv bench_weak.csv   --save-dir docs/

# Or generate demo plots without running the solver
python scripts/scaling_analysis.py --demo --save-dir docs/
```

---

## Visualisation

```bash
# Run solver first to generate snapshots
./build/heat_serial --nx 256 --steps 2000 --out 200 --outdir output/

# Single heatmap
python scripts/visualise.py --outdir output/ --snapshot 2000 --save-dir docs/

# Animation (GIF)
python scripts/visualise.py --outdir output/ --animate --fps 10 --save-dir docs/

# Temperature profile + convergence
python scripts/visualise.py --outdir output/ --profile --convergence --save-dir docs/
```

---

## TPS Validation

Validates the solver against the **Transient Plane Source** analytical solution —
the same measurement principle used in the [HD_Intelligent](https://github.com/BitnooriLee/HD_Intelligent) platform.

```bash
# Validate with synthetic TPS data (insulating material, default)
python scripts/validate_tps.py --material insulator --outdir output/ --save-dir docs/

# Aluminium
python scripts/validate_tps.py --material aluminium --no-solver

# Available materials: aluminium, copper, concrete, insulator, stainless_steel
```

**Sample output:**
```
=======================================================
  TPS Validation Report — insulator
=======================================================
  Parameter            True          Fitted    Error %
  -------------------------------------------------------
  alpha [m²/s]     1.4000e-07    1.3982e-07     0.13%
  lambda [W/mK]    2.0000e-01    2.0018e-01     0.09%
=======================================================
```

---

## Scaling Results

> Generated by `bench_scaling` + `scaling_analysis.py`.

| Threads | Grid | Time (s) | Speedup | Efficiency | GFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 512² | 4.20 | 1.00× | 100% | 0.32 |
| 2 | 512² | 2.18 | 1.93× | 96% | 0.61 |
| 4 | 512² | 1.13 | 3.72× | 93% | 1.18 |
| 8 | 512² | 0.60 | 7.02× | 88% | 2.22 |

> Amdahl parallel fraction: **f = 0.975** — BC application and buffer swaps form the serial bottleneck.

---

## Connection to HD_Intelligent (TPS Experiments)

This solver closes the loop between **numerical simulation** and **physical measurement**:

```
Physical sample  →  TPS sensor  →  HD_Intelligent   →  α, λ (measured)
                                                              ↕  compare
  ∂T/∂t = α∇²T  →  heat_mpi    →  T(x,y,t) field   →  α, λ (simulated)
```

The `validate_tps.py` script fits the TPS analytical model to both the
experimental temperature-rise curve and the solver's mean-disc temperature,
then computes the percentage error in extracted α and λ.
