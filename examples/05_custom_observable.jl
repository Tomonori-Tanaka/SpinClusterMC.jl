# 05_custom_observable.jl
#
# Add a user-defined observable to an `SCEMC` run via the
# `params[:extra_measure]` callback. We use the `ferh_4x4x4` fixture and
# compute the per-sublattice mean spin magnitude for the Fe sublattice
# (atoms 1..64) and the Rh sublattice (atoms 65..128), then run a short
# MC at low temperature so the two sublattice magnetizations are
# distinguishable.
#
#     julia --project=. examples/05_custom_observable.jl
#
# Pedagogical vs production notes
# -------------------------------
# * The Fe / Rh atom ranges below are hard-coded from the XML's
#   `<pos element="Fe">` / `<pos element="Rh">` labels (verifiable with
#   `grep '<pos' test/ferh_4x4x4/jphi.xml`). A production pipeline would
#   read the element labels from the XML and build the ranges
#   programmatically; the Simple parser does not currently expose
#   per-atom element strings, so we keep the assignment manual.
# * The callback writes to closure buffers (`fe_samples`, `rh_samples`)
#   rather than going through `Carlo.measure!(ctx, :…, value)`. The
#   `measure!`-based path would put the samples in `ctx.measure`, bin
#   them per `params[:binsize]`, and surface them in Carlo's HDF5
#   result file — preferable for real simulations. We sidestep all of
#   that here to keep the example dependency-free.

using LinearAlgebra: norm
using Random: MersenneTwister
using Carlo

using SpinClusterMC
using SpinClusterMC.Simple

const XML = joinpath(@__DIR__, "..", "test", "ferh_4x4x4", "jphi.xml")
const FE_RANGE = 1:64           # atoms tagged element="Fe" in the XML
const RH_RANGE = 65:128         # atoms tagged element="Rh" in the XML

# Mean-spin magnitude over a *subset* of atoms.
function sublattice_magnetization(spins::AbstractMatrix{<:Real}, idxs)::Float64
    n = length(idxs)
    mx = sum(spins[1, i] for i in idxs) / n
    my = sum(spins[2, i] for i in idxs) / n
    mz = sum(spins[3, i] for i in idxs) / n
    return sqrt(mx * mx + my * my + mz * mz)
end

# Closure-backed buffers the callback writes into.
const fe_samples = Float64[]
const rh_samples = Float64[]

# `extra_measure` is dispatched by `Carlo.measure!(mc, ctx)` after the
# built-in observables (:Energy, :Magnetization, …) are recorded. The
# call chain is:
#
#     Carlo.measure!(mc, ctx)
#         -> mc.extra_measure(mc, ctx)
#         -> record_sublattices!(mc, ctx)
#
# Our buffers therefore get appended once per `Carlo.measure!` call in
# the measurement loop below. The callback signature is
# `(mc::SCEMC, ctx::Carlo.MCContext) -> Nothing` per the SCEMC docstring.
function record_sublattices!(mc, ctx)
    push!(fe_samples, sublattice_magnetization(mc.spins, FE_RANGE))
    push!(rh_samples, sublattice_magnetization(mc.spins, RH_RANGE))
    return nothing
end

params = Dict{Symbol,Any}(
    :T => 50.0,                          # low T for a well-ordered run
    :xml_path => XML,
    :repeat => (1, 1, 1),                # default (no tiling)
    :external => nothing,                # default (no Zeeman field)
    :update_scheme => :metropolis,       # default
    :spin_theta_max => 0.3,              # radians
    :renorm_every => 500,
    :thermalization => 0,
    # `:binsize => 1` is a stopgap: the Simple energy code rebuilds the
    # SphericalHarmonics calculator + the full Zlm cache on every
    # `delta_local_energy` call (see `docs/design_notes.md` under
    # "src/simple/ future-work"), and ferh_4x4x4's ~840 k cluster
    # instances make each sweep slow enough that bigger bins would push
    # this example past a sensible wall time. Bump `:binsize` (and the
    # sweep counts below) once that future-work optimization lands.
    :binsize => 1,
    :seed => 42,
    :initial_spins => :ferromagnetic,
    :extra_measure => record_sublattices!,
)

mc = SCEMC(params)
ctx = Carlo.MCContext{MersenneTwister}(params)
Carlo.init!(mc, ctx, params)

println("Loaded $(mc.h.n_atoms) atoms; |Fe|=$(length(FE_RANGE)), |Rh|=$(length(RH_RANGE))")
println()

# Sweep counts below are stopgap values (1 / 1) chosen so this example
# completes in reasonable time *with the current Simple energy code*,
# which rebuilds `SphericalHarmonics` + the full Zlm cache on every
# `delta_local_energy` call. The 840 k cluster instances in ferh_4x4x4
# multiplied by 128 flips per sweep make even a handful of sweeps slow.
# Once the planned SH-cache optimization (see `docs/design_notes.md`
# under "src/simple/ future-work") lands, raise these to 200+ / 500+
# (matching examples 01 and 02) for a meaningful statistical run.

# Thermalize (no measurements; `extra_measure` is not invoked here).
for _ = 1:1
    Carlo.sweep!(mc, ctx)
end

# Measurement phase. `Carlo.measure!(mc, ctx)` fires the built-in
# observables AND `mc.extra_measure`, which is `record_sublattices!`.
n_measure = 1
for _ = 1:n_measure
    Carlo.sweep!(mc, ctx)
    Carlo.measure!(mc, ctx)
end

fe_mean = sum(fe_samples) / length(fe_samples)
rh_mean = sum(rh_samples) / length(rh_samples)
println("After $(n_measure) measured sweeps at T = $(params[:T]) K:")
println("  ⟨|m_Fe|⟩ = $(round(fe_mean; digits = 4))   (over $(length(fe_samples)) samples)")
println("  ⟨|m_Rh|⟩ = $(round(rh_mean; digits = 4))   (over $(length(rh_samples)) samples)")
println()
println(
    "Both sublattices are dimensionless |⟨S⟩| order parameters in [0, 1]; the\n" *
    "actual physical magnetization in μ_B per atom would multiply each by the\n" *
    "corresponding moment magnitude (Fe ≈ 3.0 μ_B, Rh ≈ 1.0 μ_B; FM FeRh values).",
)
