#!/usr/bin/env julia
# example_job.jl
#
# Parallel tempering job for JPhiSpinMC.
# One task corresponds to one tempering chain.
# Continue from checkpoint if present:
#   Run (MPI):     mpiexec -n <nprocs> julia example_job.jl run
#   Run (single):                       julia example_job.jl run
# Reset history and recompute from scratch:
#   Restart:                            julia example_job.jl run --restart

using Carlo
using Carlo.JobTools
using SpinClusterMC.JPhiMagestyCarlo

const k_B_eV_per_K = 8.617333262e-5

tm = TaskMaker()
tm.seed = 1234
tm.sweeps = 500000
tm.thermalization = 5000
tm.binsize = 100
tm.spin_theta_max = 0.5

xml = joinpath(@__DIR__, "jphi.xml")
tm.xml_path = xml
tm.supercell = (1, 1, 1)

Ts_K = range(100.0, 1500.0; length = 8)
Ts_mc = collect(k_B_eV_per_K .* Ts_K)

tm.parallel_tempering = (
    mc = JPhiSpinMC,
    parameter = :T,
    values = Ts_mc,
    interval = 20,
)

# One task = one PT chain (contains all temperatures in values).
task(tm)

job = JobInfo(
    splitext(@__FILE__)[1],
    ParallelTemperingMC;
    run_time = "04:00:00",
    checkpoint_time = "01:00:00",
    tasks = make_tasks(tm),
    ranks_per_run = length(Ts_mc),
)

start(job, ARGS)
