# 01_quickstart.jl
#
# 30-second introduction to the Simple SCE Monte Carlo engine.
#
# Loads the `bcc_2x2x2` test fixture, builds the SCE Hamiltonian, runs a
# short Metropolis MC, and prints the initial / final energy and
# magnetization.
#
#     julia --project=. examples/01_quickstart.jl
#
# Pedagogical vs production notes
# -------------------------------
# * We drive `Carlo.init!` and `Carlo.sweep!` from a plain Julia loop
#   instead of invoking Carlo's job runner (`Carlo.start(...)`). Production
#   runs typically use the job runner, which handles scheduling,
#   thermalization gating via `is_thermalized(ctx)`, checkpointing, and
#   HDF5 result files.
# * We read `mc.energy` and `mc.spins` directly. Production runs route
#   samples through `Carlo.measure!(mc, ctx)`, which bin-averages them via
#   `params[:binsize]`.
# * Everything else (params keys, SCEMC API, geodesic proposal, drift
#   check cadence) matches what a real run would look like.

using LinearAlgebra: norm
using Random: MersenneTwister
using Carlo

using SpinClusterMC
using SpinClusterMC.Simple

const XML = joinpath(@__DIR__, "..", "test", "bcc_2x2x2", "jphi.xml")

# Per-atom mean-spin magnitude |m| of the current spin configuration.
# `m = (1/n) Σ_i S_i`, a dimensionless order parameter in [0, 1]; it does
# not carry per-site moment magnitudes (those live in `MomentModel`).
function mean_magnetization(spins::AbstractMatrix{<:Real})::Float64
    # `n` is the number of atoms in the supercell. This example uses the
    # base cell directly (`repeat = (1, 1, 1)`), so `n = base_n_atoms = 16`
    # from the bcc_2x2x2 XML.
    n = size(spins, 2)
    mx = sum(@view spins[1, :]) / n
    my = sum(@view spins[2, :]) / n
    mz = sum(@view spins[3, :]) / n
    return sqrt(mx * mx + my * my + mz * mz)
end

# Params for SCEMC. `T` is in Kelvin (the constructor converts to eV).
# Defaults marked with `# default` are spelled out for readability — they
# can be omitted; the SCEMC constructor will fill them in.
params = Dict{Symbol,Any}(
    :T => 100.0,                       # 100 K
    :xml_path => XML,
    :repeat => (1, 1, 1),              # default (no tiling: supercell == base cell)
    :external => nothing,              # default (SCE only; e.g. Simple.Zeeman([0,0,1.0]; unit=:tesla))
    :update_scheme => :metropolis,     # default and only v1 option (dispatch hook for future updates)
    :spin_theta_max => 0.3,            # geodesic proposal half-width (radians)
    :renorm_every => 100,
    # `thermalization` / `binsize` / `seed` are read by Carlo.MCContext.
    # Their effect (gating measurements via `is_thermalized`, bin-averaging
    # in `Carlo.measure!`) is bypassed in this example because we sample
    # `mc.spins` and `mc.energy` directly.
    :thermalization => 0,
    :binsize => 50,
    :seed => 42,
    :initial_spins => :ferromagnetic,   # all spins along +z
)

mc = SCEMC(params)
ctx = Carlo.MCContext{MersenneTwister}(params)
Carlo.init!(mc, ctx, params)

println("Loaded $(mc.h.n_atoms) atoms from $(basename(XML))")
println("Initial energy:          $(mc.energy) eV")
println("Initial energy / atom:   $(mc.energy / mc.h.n_atoms) eV/atom")
println("Initial |m|:             $(mean_magnetization(mc.spins))")

# Run 500 Metropolis sweeps.
n_sweeps = 500
for _ = 1:n_sweeps
    Carlo.sweep!(mc, ctx)
end

println()
println("After $(n_sweeps) sweeps at T = $(params[:T]) K:")
println("  Energy / atom:         $(mc.energy / mc.h.n_atoms) eV/atom")
println("  |m|:                   $(mean_magnetization(mc.spins))")
