using Plots
using DataFrames
using Carlo.ResultTools

# Boltzmann constant (eV/K), same as example_job.jl — df.T is T_mc in eV
const k_B_eV_per_K = 8.617333262e-5

df = DataFrame(ResultTools.dataframe("example_job.results.json"))
T_K = df.T ./ k_B_eV_per_K

p1 = plot(T_K, df.SpecificHeat;
    xlabel = "Temperature (K)",
    ylabel = "SpecificHeat",
#    group = df.Lx,
#    legendtitle = "L",
)

p2 = plot(T_K, df.Magnetization;
    xlabel = "Temperature (K)",
    ylabel = "Magnetization",
#    group = df.Lx,
#    legendtitle = "L",
)

plot(p1, p2; layout = (2, 1))
