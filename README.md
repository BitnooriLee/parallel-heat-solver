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

## Build and Setup

### 1) System dependencies

| Package | Version | Notes |
|---|---|---|
| CMake | >= 3.20 | |
| C++ compiler | C++17 (GCC >= 9 / Clang >= 10) | |
| Open MPI | >= 4.0 | macOS: `brew install open-mpi` |
| OpenMP | compiler support | |
| Python | >= 3.10 | for plotting/validation scripts |

### 2) Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 3) Configure and build

```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=ON -DENABLE_OPENMP=ON
cmake --build build -j$(sysctl -n hw.ncpu)   # macOS
# cmake --build build -j$(nproc)             # Linux
```

Built targets:
- `heat_serial`
- `heat_openmp`
- `heat_mpi` (only when MPI is found)
- `bench_scaling`

---

## Detailed Run Guide

### A. Serial baseline run

```bash
./build/heat_serial \
  --nx 512 --ny 512 \
  --steps 2000 \
  --alpha 1.4e-7 \
  --out 200 \
  --outdir output_serial
```

This produces:
- `output_serial/metadata.txt`
- `output_serial/T_0.bin`, `T_200.bin`, ..., `T_2000.bin`

### B. OpenMP run

```bash
OMP_NUM_THREADS=8 ./build/heat_openmp \
  --nx 512 --ny 512 \
  --steps 2000 \
  --threads 8 \
  --out 200 \
  --outdir output_openmp
```

Use either `OMP_NUM_THREADS` or `--threads` (both are supported).

### C. MPI + OpenMP hybrid run

```bash
mpirun -np 4 ./build/heat_mpi \
  --nx 1024 --ny 1024 \
  --steps 5000 \
  --threads 2 \
  --out 500 \
  --outdir output_mpi
```

Optional speedup reporting against a known serial time:

```bash
mpirun -np 8 ./build/heat_mpi \
  --nx 1024 --steps 2000 \
  --threads 2 \
  --serial-time 12.4
```

---

## Benchmark and Scaling

### 1) Strong scaling (fixed problem size)

```bash
./build/bench_scaling \
  --mode strong \
  --grid 1024 \
  --max-threads 8 \
  --steps 500 \
  --out bench_strong.csv
```

### 2) Weak scaling

```bash
./build/bench_scaling \
  --mode weak \
  --max-threads 8 \
  --steps 500 \
  --out bench_weak.csv
```

### 3) Plot scaling results

```bash
python scripts/scaling_analysis.py --csv bench_strong.csv --save-dir docs
python scripts/scaling_analysis.py --csv bench_weak.csv   --save-dir docs
```

Or generate synthetic demo plots:

```bash
python scripts/scaling_analysis.py --demo --save-dir docs
```

---

## Visualisation

Run solver first with snapshots enabled (`--out > 0`), then:

```bash
# Single heatmap
python scripts/visualise.py --outdir output_serial --snapshot 2000 --save-dir docs

# Animation (GIF)
python scripts/visualise.py --outdir output_serial --animate --fps 10 --save-dir docs

# Temperature profile + convergence
python scripts/visualise.py --outdir output_serial --profile --convergence --save-dir docs
```

---

## TPS Validation (with HD_Intelligent workflow)

`validate_tps.py` supports two data sources:
1) synthetic TPS experiment generated in-script
2) external CSV data exported from HD_Intelligent measurements

### 1) Synthetic TPS validation

```bash
python scripts/validate_tps.py \
  --material insulator \
  --outdir output_serial \
  --save-dir docs
```

### 2) External HD_Intelligent CSV validation

```bash
python scripts/validate_tps.py \
  --csv hd_tps_export.csv \
  --material insulator \
  --power 0.1 \
  --radius 6.4e-3 \
  --outdir output_serial \
  --save-dir docs
```

CSV header requirement:
- time column: `time_s` (or `time`, `t`)
- temperature-rise column: `deltaT_K` (or `deltaT`, `dT`, `delta_t`)

Example:

```csv
time_s,deltaT_K
0.2,0.0013
0.6,0.0028
1.0,0.0041
```

### 3) TPS-ready solver run example

```bash
./build/heat_serial \
  --nx 128 --ny 128 \
  --alpha 1.4e-7 \
  --steps 500 \
  --out 50 \
  --T_hot 320 --T_cold 300 \
  --outdir output_tps

python scripts/validate_tps.py \
  --material insulator \
  --outdir output_tps \
  --save-dir docs
```

Available materials:
- `aluminium`, `copper`, `concrete`, `insulator`, `stainless_steel`

---

## Connection to HD_Intelligent (TPS Experiments)

This project is designed to connect **numerical simulation** with **TPS-based measurement workflows** used in HD_Intelligent:

```
Physical sample -> TPS sensor -> HD_Intelligent -> alpha, lambda (measured)
                                                     <-> compare
dT/dt = alpha*Laplacian(T) -> heat solver -> T(x,y,t) -> alpha, lambda (simulated/fitted)
```

The `validate_tps.py` pipeline fits the TPS analytical model to temperature-rise data
and reports fitted thermal properties for direct comparison with reference material values.
