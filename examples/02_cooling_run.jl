# 02_cooling_run.jl
#
# Simulated-annealing cooling on the `bcc_2x2x2` fixture: build the MC
# state once at the highest temperature, then walk down a ladder of
# temperatures, **carrying the spin configuration across** each step so
# the system tracks its (approximate) thermal state as it cools. Energy
# and magnetization are sampled at every temperature and written to a
# CSV file.
#
#     julia --project=. examples/02_cooling_run.jl
#
# Output: `cooling_results.csv` next to this script.
#
# Pedagogical vs production notes
# -------------------------------
# * `mc.T` is mutable. Reassigning `mc.T = T_K * BOLTZMANN_EV_PER_KELVIN`
#   between temperature steps changes the Metropolis acceptance ratio
#   `exp(-ΔE / mc.T)` for subsequent sweeps; the spin matrix and the
#   running `mc.energy` are unchanged by the assignment. This same
#   mechanism is what future parallel-tempering glue would use to swap
#   temperatures between replicas.
# * The two-loop "thermalize, then measure" split at each temperature is
#   explicit so the convention is readable. Mathematically equivalent to
#   a single loop that discards the first `N_THERMAL` samples. Carlo's
#   job runner does the same thing internally via `is_thermalized(ctx)`.
# * Sampling is done by reading `mc.energy` and `mc.spins` directly into
#   raw Vectors, one record per sweep. Production runs typically call
#   `Carlo.measure!(mc, ctx)` instead, which sends samples through
#   `add_sample!(ctx.measure, …)`; `params[:binsize]` (50 below) then
#   controls how many raw samples Carlo averages into one committed bin
#   in the result file. Raw access here keeps the per-sweep data-flow
#   visible.
# * CSV output via `writedlm` is a teaching choice. Production runs use
#   Carlo's HDF5 result files, which carry per-bin samples and
#   correlation-aware error bars.

using DelimitedFiles: writedlm
using Random: MersenneTwister
using Carlo

using SpinClusterMC
using SpinClusterMC.Simple

const XML = joinpath(@__DIR__, "..", "test", "bcc_2x2x2", "jphi.xml")
const OUT = joinpath(@__DIR__, "cooling_results.csv")

# Per-atom mean-spin magnitude |m|. `n` is the supercell atom count.
function mean_magnetization(spins::AbstractMatrix{<:Real})::Float64
    n = size(spins, 2)
    mx = sum(@view spins[1, :]) / n
    my = sum(@view spins[2, :]) / n
    mz = sum(@view spins[3, :]) / n
    return sqrt(mx * mx + my * my + mz * mz)
end

# Temperature ladder in Kelvin, hot → cold.
const Ts_K = [2000.0, 1500.0, 1000.0, 700.0, 500.0, 300.0, 200.0, 100.0, 50.0]
const N_THERMAL = 500
const N_MEASURE = 1000

# Storage: T_K, ⟨E/atom⟩, ⟨|m|⟩, std(E/atom), std(|m|).
results = Matrix{Float64}(undef, length(Ts_K), 5)

# Build SCEMC once at the highest temperature. `:initial_spins = :random`
# starts the annealing from a paramagnetic state, which is the usual
# convention; switch to `:ferromagnetic` to start from a polarized state.
params = Dict{Symbol,Any}(
    :T => Ts_K[1],
    :xml_path => XML,
    :repeat => (1, 1, 1),              # default (no tiling)
    :external => nothing,              # default (no Zeeman field)
    :update_scheme => :metropolis,     # default
    :spin_theta_max => 0.3,            # radians
    :renorm_every => 500,
    :thermalization => 0,
    :binsize => 50,                    # see top-of-file note about binsize
    :seed => 42,
    :initial_spins => :random,
)
mc = SCEMC(params)
ctx = Carlo.MCContext{MersenneTwister}(params)
Carlo.init!(mc, ctx, params)

for (k, T_K) in enumerate(Ts_K)
    # Update the temperature *in place*: spins and `mc.energy` carry over
    # from the previous step, only the Metropolis weight changes.
    mc.T = T_K * Simple.BOLTZMANN_EV_PER_KELVIN

    # Phase 1: thermalize at the new temperature. No samples kept.
    for _ = 1:N_THERMAL
        Carlo.sweep!(mc, ctx)
    end

    # Phase 2: measurement. Manual raw-sample collection (see top notes).
    e_samples = Vector{Float64}(undef, N_MEASURE)
    m_samples = Vector{Float64}(undef, N_MEASURE)
    for s = 1:N_MEASURE
        Carlo.sweep!(mc, ctx)
        e_samples[s] = mc.energy / mc.h.n_atoms
        m_samples[s] = mean_magnetization(mc.spins)
    end

    e_mean = sum(e_samples) / N_MEASURE
    m_mean = sum(m_samples) / N_MEASURE
    # Plain (sample-variance) standard error. Production runs use Carlo's
    # binning-aware estimator which accounts for autocorrelation.
    e_std = sqrt(sum((e_samples .- e_mean) .^ 2) / (N_MEASURE - 1))
    m_std = sqrt(sum((m_samples .- m_mean) .^ 2) / (N_MEASURE - 1))
    results[k, :] = [T_K, e_mean, m_mean, e_std, m_std]
    println(
        "T = $(lpad(T_K, 7)) K  " *
        "E/atom = $(round(e_mean; digits = 5)) ± $(round(e_std; digits = 5))  " *
        "|m| = $(round(m_mean; digits = 4)) ± $(round(m_std; digits = 4))",
    )
end

# Write CSV with a one-line header.
header = reshape(["T_K", "E_per_atom_eV", "abs_M", "std_E", "std_M"], 1, :)
writedlm(OUT, vcat(header, results), ',')
println()
println("Wrote $(OUT)")
