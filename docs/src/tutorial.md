# Tutorial

This tutorial walks through the typical workflow for running Monte Carlo simulations
with SpinClusterMC.jl.

## Prerequisites

- A `jphi.xml` file produced by [Magesty.jl](https://github.com/Tomonori-Tanaka/Magesty.jl).
  See the [Magesty.jl tutorial](https://Tomonori-Tanaka.github.io/Magesty.jl/tutorial/) for how to generate it.
- Julia ≥ 1.12 with SpinClusterMC.jl and Carlo.jl installed.

## 1. Metropolis MC

The simplest job runs independent Metropolis simulations at a set of temperatures.

```julia
using Carlo
using Carlo.JobTools
using SpinClusterMC.JPhiMagestyCarlo

const k_B = 8.617333262e-5  # eV/K

tm = TaskMaker()
tm.seed           = 1234
tm.sweeps         = 100_000   # measurement sweeps
tm.thermalization = 5_000     # equilibration sweeps (excluded from measurements)
tm.binsize        = 50        # bin length for autocorrelation error estimation
tm.spin_theta_max = 0.5       # geodesic proposal half-angle (radians)
tm.supercell      = (2, 2, 2) # tile the primitive XML cell to 8× more atoms
tm.xml_path       = "path/to/jphi.xml"

for T_K in range(100.0, 1500.0, length = 8)
    tm.T = k_B * T_K   # temperature in eV
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

Run it:

```bash
julia job.jl run                        # single process
mpiexec -n 8 julia job.jl run          # MPI: tasks run in parallel
```

Each temperature is a separate Carlo task and can be checkpointed and resumed independently.

## 2. Parallel Tempering

Parallel tempering couples replicas at different temperatures with periodic swap attempts,
which improves ergodicity and is especially useful near phase transitions.

```julia
using Carlo
using Carlo.JobTools
using SpinClusterMC.JPhiMagestyCarlo

const k_B = 8.617333262e-5

tm = TaskMaker()
tm.seed           = 1234
tm.sweeps         = 500_000
tm.thermalization = 5_000
tm.binsize        = 100
tm.spin_theta_max = 0.5
tm.xml_path       = "path/to/jphi.xml"
tm.supercell      = (2, 2, 2)

Ts_K  = range(100.0, 1500.0; length = 8)
Ts_mc = collect(k_B .* Ts_K)

tm.parallel_tempering = (
    mc        = JPhiSpinMC,
    parameter = :T,
    values    = Ts_mc,
    interval  = 20,           # swap attempt every 20 sweeps
)

task(tm)   # one task = one full PT chain containing all temperatures

job = JobInfo(
    splitext(@__FILE__)[1],
    ParallelTemperingMC;
    run_time        = "04:00:00",
    checkpoint_time = "01:00:00",
    tasks           = make_tasks(tm),
    ranks_per_run   = length(Ts_mc),  # one MPI rank per replica
)
start(job, ARGS)
```

Run with exactly one MPI rank per temperature:

```bash
mpiexec -n 8 julia job.jl run
```

## 3. Setting an initial spin configuration

By default, all spins are drawn uniformly at random on the unit sphere.
To start from a specific configuration (e.g., ferromagnetic), pass `initial_spins`:

```julia
base_n_atoms = 16   # number of atoms in the base cell (without tiling)

# Ferromagnetic: all spins along +z
s0 = zeros(3, base_n_atoms)
s0[3, :] .= 1.0
tm.initial_spins = s0
```

The `3 × base_n_atoms` matrix is tiled periodically over the full supercell.
Each column is renormalized to a unit vector automatically.

## 4. Restricting interaction body sizes

To run with only pair interactions (body size 2), set:

```julia
tm.enabled_bodies = [2]
```

This raises an `ArgumentError` if body size 2 is not present in the XML, or if the
selection leaves no active interactions.

## 5. Carlo.jl commands

Every job script accepts a command as its last argument:

| Command | Description |
|---------|-------------|
| `run` | Start a new run or resume from the latest checkpoint |
| `run --restart` | Discard all checkpoints and rerun from scratch |
| `status` | Print progress (completed sweeps, ETA) without running |
| `merge` | Merge results of an incomplete simulation |
| `delete` | Clean up a simulation directory |

```bash
julia job.jl run
julia job.jl status
julia job.jl run --restart
```

## 6. Reading results

Carlo writes a JSON results file next to the job script:

```julia
using Carlo.ResultTools, DataFrames

df = DataFrame(ResultTools.dataframe("job.results.json"))
T_K = df.T ./ 8.617333262e-5   # convert eV → K

# Available columns (Metropolis):
# df.Energy, df.SpecificHeat, df.Magnetization, df.Susceptibility, df.BinderRatio, ...
```

For parallel tempering, observables are vectors across the temperature chain:

```julia
T_mc = Float64.(df.parallel_tempering[1]["values"])
T_K  = T_mc ./ 8.617333262e-5

cv     = getfield.(df.SpecificHeat[1], :val)
cv_err = getfield.(df.SpecificHeat[1], :err)
```

### Available observables

| Name | Formula |
|:-----|:--------|
| `Energy` | Total energy per atom (eV) |
| `SpecificHeat` | $N(\langle E^2 \rangle - \langle E \rangle^2) / T^2$ |
| `Magnetization` | Vector magnetization magnitude $|\langle \mathbf{S} \rangle|$ |
| `Susceptibility` | $N \langle m^2 \rangle / T$ |
| `BinderRatio` | $\langle m^2 \rangle^2 / \langle m^4 \rangle$ |
