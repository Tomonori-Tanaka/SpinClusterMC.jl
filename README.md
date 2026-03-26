# SpinClusterMC.jl

[![Build Status](https://github.com/Tomonori-Tanaka/SpinClusterMC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Tomonori-Tanaka/SpinClusterMC.jl/actions/workflows/CI.yml?query=branch%3Amain)

Monte Carlo simulation of spin systems using **Spin-Cluster Expansion (SCE)** models.
Reads interaction parameters from [Magesty.jl](https://github.com/Tomonori-Tanaka/Magesty.jl) XML output and runs classical spin simulations via the [Carlo.jl](https://github.com/lukas-Weber/Carlo.jl) framework.

## Features

- **Metropolis Monte Carlo** — independent simulations at arbitrary temperatures
- **Parallel Tempering** — replica-exchange MC for efficient sampling across a temperature range
- **Geodesic spin proposal** — configurable random walk on the unit sphere for improved acceptance at low temperature
- **Supercell tiling** — tile the primitive cell to any `(n₁, n₂, n₃)` supercell
- **HDF5 checkpointing** — resume interrupted runs via Carlo.jl
- **MPI parallelization** — distribute tasks (Metropolis) or replicas (parallel tempering) across MPI ranks
- **Efficient energy evaluation** — preallocated tensor contractions and cached spherical harmonics in the Metropolis hot path

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Tomonori-Tanaka/SpinClusterMC.jl")
```

> **Julia ≥ 1.12** is required.

## Quick start

The main MC type is `JPhiSpinMC`. A minimal job script looks like:

```julia
using Carlo
using Carlo.JobTools
using SpinClusterMC.JPhiMagestyCarlo

const k_B = 8.617333262e-5  # eV/K

tm = TaskMaker()
tm.seed            = 1234
tm.sweeps          = 100_000   # measurement sweeps
tm.thermalization  = 5_000     # equilibration sweeps (not measured)
tm.binsize         = 50        # bin size for error estimation
tm.spin_theta_max  = 0.5       # geodesic proposal half-angle (radians); nothing → uniform
tm.supercell       = (1, 1, 1) # tile primitive cell; e.g. (2,2,2) → 8× more atoms
tm.xml_path        = "path/to/jphi.xml"

for T_K in range(100.0, 1500.0, length = 8)
    tm.T = k_B * T_K   # temperature in eV (same units as jphi.xml)
    task(tm)
end

job = JobInfo(
    splitext(@__FILE__)[1],
    JPhiSpinMC;
    run_time        = "04:00:00",
    checkpoint_time = "01:00:00",
    tasks           = make_tasks(tm),
)
start(job, ARGS)
```

Run it with:

```bash
julia example_job.jl run          # single process (tasks run sequentially)
mpiexec -n <nprocs> julia example_job.jl run   # MPI (tasks run in parallel)
```

### Carlo.jl commands

| Command | Description |
|---------|-------------|
| `run` | Start a new run or resume from the latest checkpoint |
| `run --restart` | Discard all checkpoints and rerun from scratch |
| `status` | Print progress (completed sweeps, ETA) without running |
| `merge` | Merge results of an incomplete simulation |
| `delete` | Clean up a simulation directory |

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `T` | `Float64` | Temperature in the same energy unit as `JPhi@unit` in `jphi.xml` (typically eV) |
| `xml_path` | `String` | Path to the Magesty `jphi.xml` file |
| `sweeps` | `Int` | Number of measurement sweeps |
| `thermalization` | `Int` | Equilibration sweeps (excluded from measurements) |
| `binsize` | `Int` | Bin size for autocorrelation error estimation |
| `spin_theta_max` | `Float64` or `nothing` | Geodesic proposal half-angle in radians; `nothing` → uniform random spin on full sphere |
| `supercell` / `repeat` | `Tuple{Int,Int,Int}` | Tile the primitive cell `(n₁,n₂,n₃)` times |
| `seed` | `Int` | Random seed |

### Temperature units

Temperatures must be in the energy unit declared by `JPhi@unit` in `jphi.xml` (usually eV).
Using `k_B = 8.617333262e-5 eV/K`, pass `T = k_B * T_K` for a temperature in Kelvin.

## Output

Carlo writes the following files next to the job script:

| File | Contents |
|------|----------|
| `<job>.results.json` | Measured observables per task (Energy, SpecificHeat, …) |
| `<job>.h5` | HDF5 checkpoint file; updated every `checkpoint_time` |

### Reading results

```julia
using Carlo.ResultTools, DataFrames

df = DataFrame(ResultTools.dataframe("example_job.results.json"))
T_K = df.T ./ 8.617333262e-5   # convert eV → K
# df.Energy, df.SpecificHeat, df.Magnetization², ...
```

## Examples

See the [examples/](examples/) directory for ready-to-run job scripts.

## Related packages

- [Magesty.jl](https://github.com/Tomonori-Tanaka/Magesty.jl) — generates the `jphi.xml` SCE model files read by this package
- [Carlo.jl](https://github.com/lukas-Weber/Carlo.jl) — Monte Carlo framework (checkpointing, MPI, parallel tempering)
