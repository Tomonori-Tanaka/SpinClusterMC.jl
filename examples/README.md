# SpinClusterMC — Examples

Each subdirectory is an independent example job.
See the README in each directory for system-specific instructions.

## Available examples

### bcc Fe

| Directory | Method | Description |
|-----------|--------|-------------|
| [bccFe/metropolis](bccFe/metropolis/) | Metropolis | Independent simulations at 8 temperatures |
| [bccFe/parallel_tempering](bccFe/parallel_tempering/) | Parallel Tempering | Replica-exchange MC over 8 temperatures |

---

## Common concepts

### Temperature units

All temperatures are passed in the same energy unit as `JPhi@unit` in `jphi.xml` (typically eV).

```julia
const k_B_eV_per_K = 8.617333262e-5  # eV/K
T_eV = k_B_eV_per_K * T_K
```

### Carlo.jl commands

Every job script accepts a command as its last argument
(see [Carlo.jl paper](https://scipost.org/SciPostPhysCodeb.49) for details):

| Command | Alias | Description |
|---------|-------|-------------|
| `run` | `r` | Start a new run or resume from the latest checkpoint |
| `run --restart` | | Discard all checkpoints and rerun from scratch |
| `status` | `s` | Print progress (completed sweeps, ETA) without running |
| `merge` | `m` | Merge results of an incomplete simulation |
| `delete` | `d` | Clean up a simulation directory |

```bash
julia example_job.jl run
julia example_job.jl run --restart
julia example_job.jl status
julia example_job.jl merge
julia example_job.jl delete
```

With MPI (`nprocs` workers + 1 scheduler):
```bash
mpiexec -n <nprocs> julia example_job.jl run
```

### Output files

Carlo writes the following files next to the job script:

| File | Contents |
|------|----------|
| `<job>.results.json` | Observables (Energy, SpecificHeat, …) per task |
| `<job>.h5` | Checkpoint file (HDF5); updated every `checkpoint_time` |

### Reading results

```julia
using Carlo.ResultTools, DataFrames

df = DataFrame(ResultTools.dataframe("example_job.results.json"))
# Columns: T, sweeps, Energy, SpecificHeat, ...
```
