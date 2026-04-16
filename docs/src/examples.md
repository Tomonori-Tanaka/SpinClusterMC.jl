# Examples

Ready-to-run job scripts are in the [`examples/`](https://github.com/Tomonori-Tanaka/SpinClusterMC.jl/tree/main/examples) directory.

## bcc Fe — Metropolis

Runs 8 independent Metropolis simulations at temperatures from 100 K to 1500 K
for a BCC iron system described by an SCE Hamiltonian from Magesty.jl.

**Location:** `examples/bccFe/metropolis/`

```julia
# examples/bccFe/metropolis/example_job.jl
using Carlo
using Carlo.JobTools
using SpinClusterMC.JPhiMagestyCarlo

const k_B_eV_per_K = 8.617333262e-5

tm = TaskMaker()
tm.seed           = 1234
tm.sweeps         = 500_000
tm.thermalization = 50_000
tm.binsize        = 50
tm.spin_theta_max = 0.5
tm.supercell      = (2, 2, 2)   # 16-atom base cell → 128 atoms

xml = joinpath(@__DIR__, "jphi.xml")
for T_K in range(100.0, 1500.0, length = 8)
    tm.T        = k_B_eV_per_K * T_K
    tm.xml_path = xml
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

### Running

```bash
# Single process (tasks run sequentially)
julia example_job.jl run

# MPI (tasks run in parallel)
mpiexec -n 8 julia example_job.jl run

# Resume from checkpoint
julia example_job.jl run

# Restart from scratch
julia example_job.jl run --restart
```

### Visualization

```julia
# examples/bccFe/metropolis/visualize.jl
using Plots
using DataFrames
using Carlo.ResultTools

const k_B_eV_per_K = 8.617333262e-5

df   = DataFrame(ResultTools.dataframe("example_job.results.json"))
T_K  = df.T ./ k_B_eV_per_K

p1 = plot(T_K, df.SpecificHeat;
    xlabel = "Temperature (K)",
    ylabel = "Specific Heat",
    marker = :circle,
)

p2 = plot(T_K, df.Magnetization;
    xlabel = "Temperature (K)",
    ylabel = "Magnetization",
    marker = :circle,
)

plot(p1, p2; layout = (2, 1))
```

---

## bcc Fe — Parallel Tempering

Uses replica-exchange MC with 8 temperatures between 100 K and 1500 K.
A single Carlo task contains the full temperature chain; each MPI rank handles one replica.

**Location:** `examples/bccFe/parallel_tempering/`

```julia
# examples/bccFe/parallel_tempering/example_job.jl
using Carlo
using Carlo.JobTools
using SpinClusterMC.JPhiMagestyCarlo

const k_B_eV_per_K = 8.617333262e-5

tm = TaskMaker()
tm.seed           = 1234
tm.sweeps         = 500_000
tm.thermalization = 5_000
tm.binsize        = 100
tm.spin_theta_max = 0.5
tm.xml_path       = joinpath(@__DIR__, "jphi.xml")
tm.supercell      = (2, 2, 2)

Ts_K  = range(100.0, 1500.0; length = 8)
Ts_mc = collect(k_B_eV_per_K .* Ts_K)

tm.parallel_tempering = (
    mc        = JPhiSpinMC,
    parameter = :T,
    values    = Ts_mc,
    interval  = 20,
)

task(tm)

job = JobInfo(
    splitext(@__FILE__)[1],
    ParallelTemperingMC;
    run_time        = "04:00:00",
    checkpoint_time = "01:00:00",
    tasks           = make_tasks(tm),
    ranks_per_run   = length(Ts_mc),
)
start(job, ARGS)
```

### Running

Parallel tempering requires exactly one MPI rank per temperature:

```bash
mpiexec -n 8 julia example_job.jl run
```

### Reading PT results

```julia
using DataFrames, Carlo.ResultTools

df = DataFrame(ResultTools.dataframe("example_job.results.json"))

# PT results: observables are vectors across the temperature chain
pt    = df.parallel_tempering[1]
T_K   = Float64.(pt["values"]) ./ 8.617333262e-5

cv     = getfield.(df.SpecificHeat[1], :val)
cv_err = getfield.(df.SpecificHeat[1], :err)
mag    = getfield.(df.Magnetization[1], :val)
```

---

## Direct energy evaluation (no MC)

`SCEHamiltonian` and `sce_energy` can be used independently of Carlo.jl
for quick energy calculations on fixed spin configurations:

```julia
using SpinClusterMC.JPhiMagestyCarlo
using LinearAlgebra

h = load_sce_hamiltonian("jphi.xml"; repeat = (2, 2, 2))

n = h.n_atoms

# Ferromagnetic configuration: all spins along +z
spins_fm = zeros(3, n)
spins_fm[3, :] .= 1.0
E_fm = sce_energy(h, spins_fm)
println("FM energy: ", E_fm, " eV")

# Random spin configuration
spins_rand = randn(3, n)
for i in 1:n
    spins_rand[:, i] ./= norm(spins_rand[:, i])
end
E_rand = sce_energy(h, spins_rand)
println("Random spin energy: ", E_rand, " eV")
```
