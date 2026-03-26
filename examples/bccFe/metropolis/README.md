# bcc Fe — Metropolis

Runs 8 independent Metropolis simulations at temperatures from 100 K to 1500 K.
Each temperature is a separate Carlo task and can be checkpointed/resumed independently.

## Running

Single process (tasks run sequentially):
```bash
julia example_job.jl run
```

With MPI (tasks run in parallel; use any `nprocs ≥ 1`):
```bash
mpiexec -n <nprocs> julia example_job.jl run
```

Resume, restart, or check progress:
```bash
julia example_job.jl status
julia example_job.jl run             # resume from checkpoint
julia example_job.jl run --restart   # discard checkpoint and rerun
```

## Parameters

```julia
tm.sweeps         = 100000   # Number of measurement sweeps
tm.thermalization = 5000     # Equilibration sweeps (excluded from measurements)
tm.binsize        = 50       # Bin size for autocorrelation error estimation
tm.spin_theta_max = 0.5      # Geodesic proposal half-angle (radians);
                             #   nothing → uniform random spin on full sphere
tm.supercell      = (1,1,1)  # Tile the primitive cell (n1,n2,n3)
                             #   e.g. (2,2,2) → 16×8 = 128 atoms
tm.T              = ...      # Temperature in eV
tm.xml_path       = ...      # Path to Magesty jphi.xml
```

## Visualization

Reads `example_job.results.json` and plots specific heat vs temperature.

Run from the Julia REPL to keep the plot window open (recommended):
```julia
include("visualize.jl")
```

Or from the terminal (the plot window may close immediately on exit):
```bash
julia visualize.jl
```

```julia
using Carlo.ResultTools, DataFrames

df = DataFrame(ResultTools.dataframe("example_job.results.json"))
T_K = df.T ./ 8.617333262e-5   # convert eV → K
# df.SpecificHeat, df.Energy, ...
```
