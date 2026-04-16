# SpinClusterMC.jl

```@meta
CurrentModule = SpinClusterMC.JPhiMagestyCarlo
```

SpinClusterMC.jl is a Julia package for classical Monte Carlo simulation of spin systems
described by a **Spin-Cluster Expansion (SCE)** Hamiltonian.
It reads interaction parameters from [Magesty.jl](https://github.com/Tomonori-Tanaka/Magesty.jl)
XML output and runs simulations via the [Carlo.jl](https://github.com/lukas-Weber/Carlo.jl) framework.

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
Pkg.add(url = "https://github.com/Tomonori-Tanaka/SpinClusterMC.jl")
```

> **Julia ≥ 1.12** is required.

## Workflow overview

```
Magesty.jl  ──→  jphi.xml  ──→  SpinClusterMC.jl  ──→  results.json
  (SCE fit)        (SCE model)      (MC simulation)      (observables)
```

1. Fit an SCE model in [Magesty.jl](https://github.com/Tomonori-Tanaka/Magesty.jl) and export to `jphi.xml`.
2. Write a Carlo.jl job script pointing at `jphi.xml` (see [Tutorial](tutorial.md)).
3. Run the job with `julia job.jl run` (single-process) or `mpiexec -n N julia job.jl run` (MPI).
4. Read `job.results.json` with `Carlo.ResultTools` for post-processing.

## Quick links

- [Tutorial](tutorial.md) — step-by-step job script examples
- [Examples](examples.md) — ready-to-run scripts for bcc Fe
- [API Reference](api.md) — full type and function documentation
- [Technical Notes](technical_notes.md) — Metropolis algorithm and energy evaluation details

## Related packages

- [Magesty.jl](https://github.com/Tomonori-Tanaka/Magesty.jl) — generates the `jphi.xml` SCE model files
- [Carlo.jl](https://github.com/lukas-Weber/Carlo.jl) — Monte Carlo framework (checkpointing, MPI, parallel tempering)
