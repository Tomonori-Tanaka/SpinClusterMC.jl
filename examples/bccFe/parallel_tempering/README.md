# bcc Fe — Parallel Tempering

Runs a single parallel tempering chain spanning 8 temperatures from 100 K to 1500 K.
All replicas belong to one Carlo task; replica exchanges are attempted every `interval` sweeps.

## MPI rank requirement

Carlo assigns one MPI rank per replica, plus one scheduler rank.
The total number of MPI processes must satisfy:

```
nprocs = k × N_temps + 1
```

where `N_temps = length(values)` is the number of temperatures and `k ≥ 1` is the number
of concurrent PT runs (typically `k = 1`).

For this example (`N_temps = 8`):

| k | nprocs |
|---|--------|
| 1 | **9**  |
| 2 | 17     |
| 3 | 25     |

Passing any other number of processes will cause Carlo to raise an error at startup.

## Running

```bash
mpiexec -n 9 julia example_job.jl run
```

Resume, restart, or check progress:
```bash
julia example_job.jl status
mpiexec -n 9 julia example_job.jl run             # resume from checkpoint
mpiexec -n 9 julia example_job.jl run --restart   # discard checkpoint and rerun
```

## Parameters

```julia
tm.sweeps         = 500000   # Sweeps per replica (longer runs improve exchange statistics)
tm.thermalization = 5000     # Equilibration sweeps
tm.binsize        = 100      # Bin size for error estimation

tm.parallel_tempering = (
    mc        = JPhiSpinMC,  # MC type for each replica
    parameter = :T,           # Parameter to exchange (temperature)
    values    = Ts_mc,        # Vector of temperatures in eV (length = N_temps)
    interval  = 20,           # Attempt replica exchange every N sweeps
)

# In JobInfo:
ranks_per_run = length(Ts_mc)   # Must equal N_temps
```

## Visualization

PT observables are stored as vectors (one entry per temperature).
Reads `example_job_pt.results.json` and plots specific heat with error bars.

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

df  = DataFrame(ResultTools.dataframe("example_job_pt.results.json"))
pt  = df.parallel_tempering[1]
T_K = Float64.(pt["values"]) ./ 8.617333262e-5   # convert eV → K

cv     = getfield.(df.SpecificHeat[1], :val)
cv_err = getfield.(df.SpecificHeat[1], :err)
```
