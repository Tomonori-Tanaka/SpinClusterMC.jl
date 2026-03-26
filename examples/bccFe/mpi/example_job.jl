# example_job.jl — sample the SCE spin model from Magesty jphi.xml with Carlo
#
# Energy and jphi use the XML JPhi@unit (typically eV). Pass temperature T in the same energy units
# (e.g. T = 0.02585 eV ≈ 300 K, k_B ≈ 8.617333262e-5 eV/K).
#
# Activates this folder's Project.toml (path deps: Carlo, Magesty). First time: run once
# `julia --project=@. -e 'using Pkg; Pkg.instantiate()'` from this directory.

import Pkg
Pkg.activate(@__DIR__)

using Carlo
using Carlo.JobTools
include("JPhiMagestyCarlo.jl")
using .JPhiMagestyCarlo

const k_B_eV_per_K = 8.617333262e-5

tm = TaskMaker()
tm.seed = 42
tm.sweeps = 500000
tm.thermalization = 5000
tm.binsize = 100

xml = joinpath(@__DIR__, "jphi.xml")
# Supercell: tile the XML cell (n1,n2,n3) times in fractional stacking (e.g. 2,2,2 -> 16*8 = 128 atoms)
# tm.repeat = (2, 2, 2)
# Alias: tm.supercell = (2, 2, 2)
tm.supercell = (1, 1, 1)

for T_K in range(100.0, 1500.0, length = 8)
    tm.T = k_B_eV_per_K * T_K
    tm.xml_path = xml
    task(tm)
end

job = JobInfo(
    splitext(@__FILE__)[1],
    JPhiSpinMC;
    run_time = "04:00:00",
    checkpoint_time = "01:00:00",
    tasks = make_tasks(tm),
)
start(job, ARGS)
