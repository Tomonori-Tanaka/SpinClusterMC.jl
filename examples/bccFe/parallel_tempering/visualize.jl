using Plots
using DataFrames
using Carlo.ResultTools

# Boltzmann constant (eV/K), same as example_job_pt.jl
const k_B_eV_per_K = 8.617333262e-5

results_path = joinpath(@__DIR__, "example_job_pt.results.json")
df = DataFrame(ResultTools.dataframe(results_path))
# PT results have one row per chain, and temperature values are stored in
# parameters.parallel_tempering.values instead of a plain `T` column.
pt = df.parallel_tempering[1]
T_mc = Float64.(pt["values"])
T_K = T_mc ./ k_B_eV_per_K

# PT observables are vectors of Measurement values across the temperature chain.
cv = getfield.(df.SpecificHeat[1], :val)
cv_err = getfield.(df.SpecificHeat[1], :err)

plot(T_K, cv;
    yerror = cv_err,
    xlabel = "Temperature (K)",
    ylabel = "SpecificHeat",
    label = "L=$(Int(round(df.Lx[1])))",
)
