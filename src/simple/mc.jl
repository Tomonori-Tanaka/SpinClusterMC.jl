"""
Monte Carlo engine for the Simple submodule: `SCEMC <: Carlo.AbstractMC`.

The type bundles the SCE Hamiltonian, the spin configuration, the temperature,
the running energy (kept incrementally up to date by accepted moves), an
optional external term, and the proposal / housekeeping parameters. Carlo
calls `init!`, `sweep!`, `measure!`, and `register_evaluables` on instances of
this type to drive a Monte Carlo simulation.

# Temperature unit

`params[:T]` is in **Kelvin**. The constructor converts it once to eV via
`BOLTZMANN_EV_PER_KELVIN ≈ 8.6173e-5 eV/K` and stores `mc.T` in eV; the
Metropolis acceptance `exp(-ΔE / mc.T)` and the post-processing in
`register_evaluables` (`SpecificHeat = n · σ²(E) / T²`, …) all consume that
internal eV value. See the "physical conventions" section of `CLAUDE.md`
for the wider convention.

# Required params

| Key | Type | Meaning |
|---|---|---|
| `:T` | `Real` | Temperature in **Kelvin**. |
| `:xml_path` | `AbstractString` | Path to the Magesty `jphi.xml`. |

# Optional params (defaults shown)

| Key | Default | Meaning |
|---|---|---|
| `:repeat` | `(1, 1, 1)` | Supercell tile factors. |
| `:external` | `nothing` | An `ExternalTerm` (`Zeeman`, …) added on top of the SCE energy. |
| `:spin_theta_max` | `π` | Geodesic proposal half-width **in radians**; `π` ≈ uniform-sphere proposals, smaller values give finer local moves. |
| `:renorm_every` | `1000` | Sweep cadence for spin renormalization and energy drift check. |
| `:update_scheme` | `:metropolis` | Update algorithm; only `:metropolis` in v1. |
| `:initial_spins` | `:random` | Initial spin spec; see `init_spins`. |
| `:extra_measure` | `(mc, ctx) -> nothing` | Optional callback in `Carlo.measure!`. |
| `:extra_evaluables` | `(eval, params) -> nothing` | Optional hook in `register_evaluables`. |

# Observables

`measure!` records, all per-atom and dimensionless where relevant:

- `:Energy`, `:Energy2` — incremental total energy / n_atoms (eV/atom).
- `:Magnetization`, `:AbsMagnetization`, `:Magnetization2`, `:Magnetization4` —
  `|m|, |m|, |m|², |m|⁴` with `m = (1/n) Σ_i S_i`. `S_i` is the unit-spin
  direction, so `|m| ∈ [0, 1]` is a dimensionless order parameter and does
  not include site-resolved moment magnitudes (those live in `MomentModel`).

`register_evaluables` adds the standard derived quantities:

- `:SpecificHeat = n_atoms · (⟨E²⟩ - ⟨E⟩²) / T²`
- `:BinderRatio  = ⟨m²⟩² / ⟨m⁴⟩`
- `:Susceptibility = n_atoms · ⟨m²⟩ / T`

User-side extensions (sublattice magnetization, SI-unit magnetization,
structure factors …) plug in via the two callbacks.
"""

using Carlo
using LinearAlgebra: norm
using Random: AbstractRNG
using StaticArrays: SVector

"""
Boltzmann constant in eV per Kelvin (CODATA 2018:
`8.617333262 × 10⁻⁵ eV/K`). `SCEMC` uses it to convert the user-supplied
`params[:T]` (Kelvin) to the internal eV value that the Metropolis
acceptance and the per-atom evaluables consume.
"""
const BOLTZMANN_EV_PER_KELVIN = 8.617333262e-5

# Default no-op callbacks for the user hooks.
_no_extra_measure(mc, ctx) = nothing
_no_extra_evaluables(eval, params) = nothing

"""
    SCEMC{E<:Union{Nothing, ExternalTerm}} <: Carlo.AbstractMC

Monte Carlo state object held by Carlo for the duration of a simulation. See
the module docstring for the params / observable contract and the
temperature unit convention.

# Fields

- `h::SpinClusterHamiltonian` — SCE Hamiltonian (built once in the constructor).
- `spins::Matrix{Float64}` — `3 × n_atoms` unit-vector columns; mutated in
  place by `sweep!`.
- `T::Float64` — temperature in **eV** (converted from the Kelvin input).
  Mutable so parallel tempering can change it.
- `energy::Float64` — running total (SCE + external) tracked incrementally by
  `ΔE` on accepted moves; periodically reconciled by the drift check.
- `external::E` — `Nothing` or an `ExternalTerm` (`Union{Nothing, ExternalTerm}`).
- `theta_max::Float64` — geodesic proposal half-width.
- `renorm_every::Int` — cadence for spin renormalization and drift check.
- `sweep_count::Int` — number of completed sweeps (used by the cadence).
- `extra_measure::Function`, `extra_evaluables::Function` — user hooks.

# PT-future-work fields (not used in v1)

Retained so adding parallel tempering later needs only the PT glue
(`parallel_tempering_log_weight_ratio`, `change_parameter!`, serialize) — no
further field plumbing. Otherwise unused.

- `xml_path::String` — original XML location; PT serialize uses it to rebuild
  the Hamiltonian after a coordinator-rank gather.
- `repeat::NTuple{3, Int}` — the tile factors that the Hamiltonian was built
  with; same purpose as `xml_path`.
"""
mutable struct SCEMC{E <: Union{Nothing, ExternalTerm}} <: Carlo.AbstractMC
    h::SpinClusterHamiltonian
    spins::Matrix{Float64}
    T::Float64
    energy::Float64
    external::E
    theta_max::Float64
    renorm_every::Int
    sweep_count::Int
    extra_measure::Function
    extra_evaluables::Function
    # PT-future-work fields (not used in v1; kept for serialize round-trip).
    xml_path::String
    repeat::NTuple{3, Int}
end

function _params_repeat(params::AbstractDict)::NTuple{3, Int}
    rep = get(params, :repeat, (1, 1, 1))
    if rep isa NTuple{3, Int}
        return rep
    elseif rep isa AbstractVector || rep isa Tuple
        length(rep) == 3 ||
            throw(ArgumentError("params[:repeat] must have 3 entries; got $(length(rep))"))
        return (Int(rep[1]), Int(rep[2]), Int(rep[3]))
    end
    throw(
        ArgumentError(
        "params[:repeat] must be a 3-tuple or 3-vector of Int; got $(typeof(rep))"
    )
    )
end

# Convert the user-supplied params[:T] (Kelvin) to the internal eV value.
# Centralised so the constructor and `register_evaluables` use the same
# conversion at all times.
function _params_T_eV(params::AbstractDict)::Float64
    haskey(params, :T) ||
        throw(ArgumentError("SCEMC: params[:T] (temperature in Kelvin) is required"))
    T_K = Float64(params[:T])
    T_K > 0 ||
        throw(ArgumentError("SCEMC: params[:T] must be positive Kelvin; got $T_K"))
    return T_K * BOLTZMANN_EV_PER_KELVIN
end

"""
    SCEMC(params::AbstractDict)

Constructor used by Carlo. Reads `params`, builds the Hamiltonian (and the
spin matrix shape), and leaves the spin values for `Carlo.init!` to fill.
Temperature is converted from Kelvin to eV here.
"""
function SCEMC(params::AbstractDict)
    haskey(params, :xml_path) ||
        throw(ArgumentError("SCEMC: params[:xml_path] is required"))
    T_eV = _params_T_eV(params)
    xml_path = String(params[:xml_path])
    repeat = _params_repeat(params)
    external = get(params, :external, nothing)
    if !(external isa Union{Nothing, ExternalTerm})
        throw(
            ArgumentError(
            "params[:external] must be nothing or ExternalTerm; got $(typeof(external))"
        )
        )
    end
    theta_max = Float64(get(params, :spin_theta_max, π))
    renorm_every = Int(get(params, :renorm_every, 1000))
    extra_measure = get(params, :extra_measure, _no_extra_measure)
    extra_evaluables = get(params, :extra_evaluables, _no_extra_evaluables)
    scheme = Symbol(get(params, :update_scheme, :metropolis))
    scheme === :metropolis || throw(
        ArgumentError(
        "params[:update_scheme]=:$(scheme) not supported (v1: only :metropolis)"
    )
    )

    h = SpinClusterHamiltonian(xml_path; repeat = repeat)
    spins = Matrix{Float64}(undef, 3, h.n_atoms)
    return SCEMC(
        h,
        spins,
        T_eV,
        0.0,                # energy: filled by init!
        external,
        theta_max,
        renorm_every,
        0,                  # sweep_count
        extra_measure,
        extra_evaluables,
        xml_path,
        repeat
    )
end

# Dispatch helpers that fold the optional `external` field into the SCE
# energy / delta. The `if !isnothing(mc.external)` runtime form would be
# equivalent but JET's union-split analysis cannot always prove the
# `Nothing` branch is unreachable; explicit method dispatch on
# `Nothing` vs `ExternalTerm` keeps the static checker quiet without losing
# the type-stable fast path.
@inline _external_total_energy(::Nothing, spins) = 0.0
@inline _external_total_energy(ext::ExternalTerm, spins) = total_energy(ext, spins)
@inline _external_delta_local(::Nothing, spins, i, S_new) = 0.0
@inline _external_delta_local(ext::ExternalTerm, spins, i, S_new) = delta_local_energy(ext, spins, i, S_new)

# Total energy of the SCE + optional external piece. Used by init! and the
# drift check inside the renorm step; not on the per-flip hot path.
function _full_energy(mc::SCEMC)::Float64
    return total_energy(mc.h, mc.spins) +
           _external_total_energy(mc.external, mc.spins)
end

# Renormalize every spin to unit norm and reconcile `mc.energy` with a full
# recompute. Throws if the incremental tracker drifted beyond a relative
# `rtol = 1e-10` plus an absolute floor `atol = 1e-12` (mirrors `isapprox`
# semantics). The absolute floor keeps the check meaningful when SCE and
# external contributions cancel and `|E_full|` ends up near zero; the
# relative tail dominates in the usual `|E| ≥ O(1) eV` regime.
function _renorm_and_drift_check!(mc::SCEMC)
    @inbounds for i in 1:size(mc.spins, 2)
        sx = mc.spins[1, i]
        sy = mc.spins[2, i]
        sz = mc.spins[3, i]
        nrm = sqrt(sx * sx + sy * sy + sz * sz)
        if nrm > 0
            mc.spins[1, i] = sx / nrm
            mc.spins[2, i] = sy / nrm
            mc.spins[3, i] = sz / nrm
        end
    end
    E_full = _full_energy(mc)
    rtol = 1.0e-10
    atol = 1.0e-12
    diff = abs(mc.energy - E_full)
    threshold = rtol * max(abs(E_full), abs(mc.energy)) + atol
    diff < threshold || error(
        "SCEMC energy drift exceeded rtol=$rtol + atol=$atol after $(mc.sweep_count) sweeps: " *
        "incremental=$(mc.energy), full=$E_full, |Δ|=$diff (threshold=$threshold)"
    )
    mc.energy = E_full
    return nothing
end

function Carlo.init!(mc::SCEMC, ctx::Carlo.MCContext, params::AbstractDict)
    n_atoms = mc.h.n_atoms
    base_n_atoms = mc.h.base_n_atoms
    spec = get(params, :initial_spins, :random)
    initial = init_spins(spec, n_atoms, base_n_atoms; rng = ctx.rng)
    @inbounds for i in 1:n_atoms
        mc.spins[1, i] = initial[1, i]
        mc.spins[2, i] = initial[2, i]
        mc.spins[3, i] = initial[3, i]
    end
    mc.energy = _full_energy(mc)
    mc.sweep_count = 0
    return nothing
end

function Carlo.sweep!(mc::SCEMC, ctx::Carlo.MCContext)
    # v1 only supports Metropolis. Future updates plug in via dispatch at
    # this single site (one more `if` per scheme).
    metropolis_sweep!(mc, ctx)
    mc.sweep_count += 1
    if mc.renorm_every > 0 && mc.sweep_count % mc.renorm_every == 0
        _renorm_and_drift_check!(mc)
    end
    return nothing
end

"""
    Carlo.measure!(mc::SCEMC, ctx)

Record the per-atom energy moments and the dimensionless mean-spin order
parameter, then dispatch the user `extra_measure` callback for any
application-specific observables.
"""
function Carlo.measure!(mc::SCEMC, ctx::Carlo.MCContext)
    n = mc.h.n_atoms
    e_per_atom = mc.energy / n
    measure!(ctx, :Energy, e_per_atom)
    measure!(ctx, :Energy2, e_per_atom * e_per_atom)

    mx = 0.0
    my = 0.0
    mz = 0.0
    @inbounds for i in 1:n
        mx += mc.spins[1, i]
        my += mc.spins[2, i]
        mz += mc.spins[3, i]
    end
    inv_n = 1.0 / n
    mx *= inv_n
    my *= inv_n
    mz *= inv_n
    mag2 = mx * mx + my * my + mz * mz
    mag = sqrt(mag2)
    measure!(ctx, :Magnetization, mag)
    measure!(ctx, :AbsMagnetization, mag)
    measure!(ctx, :Magnetization2, mag2)
    measure!(ctx, :Magnetization4, mag2 * mag2)

    mc.extra_measure(mc, ctx)
    return nothing
end

"""
    Carlo.register_evaluables(::Type{<:SCEMC}, eval, params)

Register `:SpecificHeat`, `:BinderRatio`, `:Susceptibility`, then call the
user `extra_evaluables` hook (read from `params[:extra_evaluables]`). The
temperature is read from `params[:T]` (Kelvin) and converted the same way
as in the constructor.
"""
function Carlo.register_evaluables(
        ::Type{<:SCEMC}, eval::Carlo.AbstractEvaluator, params::AbstractDict
)
    T_eV = _params_T_eV(params)
    xml_path = String(params[:xml_path])
    repeat = _params_repeat(params)
    # n_atoms is needed for the per-atom -> total conversion in the
    # specific-heat and susceptibility formulas. The MC instance isn't
    # passed here, so we rebuild a Hamiltonian shell (cheap relative to the
    # whole simulation) to read it.
    h = SpinClusterHamiltonian(xml_path; repeat = repeat)
    n = h.n_atoms
    evaluate!(eval, :SpecificHeat, (:Energy2, :Energy)) do e2, e
        return n * (e2 - e * e) / (T_eV * T_eV)
    end
    evaluate!(eval, :BinderRatio, (:Magnetization2, :Magnetization4)) do mag2, mag4
        return mag2 * mag2 / mag4
    end
    evaluate!(eval, :Susceptibility, (:Magnetization2,)) do mag2
        return n * mag2 / T_eV
    end
    extra_evaluables = get(params, :extra_evaluables, _no_extra_evaluables)
    extra_evaluables(eval, params)
    return nothing
end
