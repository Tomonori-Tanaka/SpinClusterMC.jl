"""
External (non-SCE) energy terms and the magnetic-moment model used by them.

# External terms

The SCE Hamiltonian carried by `SpinClusterHamiltonian` describes site
couplings encoded in `jphi.xml`. Genuinely additive contributions outside that
expansion — uniform Zeeman fields, time-dependent drives, … — plug in here as
subtypes of `ExternalTerm`. Each subtype implements the same four-function
energy API as the SCE side (`total_energy`, `local_energy`,
`delta_local_energy`, `gradient`), dispatched on the term type. The MC engine
composes the contributions additively:

    E_full          = total_energy(h, spins) + total_energy(ext, spins)
    ΔE_full(i, S')  = delta_local_energy(h, spins, i, S') +
                      delta_local_energy(ext, spins, i, S')
    ∇_i E_full      = gradient(h, spins, i) + gradient(ext, spins, i)

Single-ion anisotropy is *not* an external term: it is representable inside
SCE as an `N = 1` cluster (the SCE expansion places no upper bound on `l`),
so it lives in `SpinClusterHamiltonian` like any other cluster. External
terms are reserved for contributions that do not fit the SCE expansion form
(uniform Zeeman, time-dependent drives, etc.).

# Magnetic-moment model

Physically, the magnetic moment at atom `i` is `μ_i = m_i · S_i` where
`S_i ∈ ℝ³` is the (unit) spin direction and `m_i` is the moment magnitude.
SCE expresses exchange couplings in terms of `S_i` directly — the magnitude
factors are absorbed into the SCE coefficients — so the SCE energy does not
read `m_i`. External terms (Zeeman, future driven-field couplings) *do* need
`m_i`, and in real materials those magnitudes typically differ between
sublattices (e.g., Fe vs Rh).

`MomentModel` captures that:

- `UniformMoment(m)` — all atoms share the same magnitude `m`.
- `PerSiteMoment(m)` — `m[i]` per supercell atom.
- (future) `ClusterExpansionMoment` — a Magesty-side cluster expansion that
  makes `m_i` depend on the local environment of atom `i`. The
  `moment_at(model, i, spins)` query already takes `spins` so this extension
  fits without changing the call sites.

The gradient routines below currently assume `m_i` is independent of `S`. A
cluster-dependent model will need to add a `∂m / ∂S` contribution in a
future revision.
"""

using StaticArrays: SVector

# Magnetic-moment model -----------------------------------------------------

"""
    MomentModel

Abstract supertype for site-resolved magnetic-moment magnitudes. A concrete
subtype provides

    moment_at(model::MomentModel, i::Integer, spins::AbstractMatrix{<:Real}) -> Float64

returning `m_i` at supercell atom `i`. The `spins` argument is reserved for
future cluster-environment-dependent models; the bundled `UniformMoment` and
`PerSiteMoment` ignore it.
"""
abstract type MomentModel end

"""
    UniformMoment(m::Real) <: MomentModel

Every atom has the same moment magnitude `m`. Units are not enforced here;
choose `m` and the external-term coupling so their product comes out in eV
(see `Zeeman` for the typical pairing, e.g., `m` in `μ_B` with field in
`eV/μ_B`).
"""
struct UniformMoment <: MomentModel
    m::Float64
    UniformMoment(m::Real) = new(Float64(m))
end

@inline function moment_at(
        model::UniformMoment, ::Integer, ::AbstractMatrix{<:Real}
)::Float64
    return model.m
end

"""
    PerSiteMoment(m::AbstractVector{<:Real}) <: MomentModel

`m[i]` is the moment magnitude at supercell atom `i`. The vector length must
match the supercell size at the energy-evaluation call site.
"""
struct PerSiteMoment <: MomentModel
    m::Vector{Float64}
    PerSiteMoment(m::AbstractVector{<:Real}) = new(collect(Float64, m))
end

@inline function moment_at(
        model::PerSiteMoment, i::Integer, ::AbstractMatrix{<:Real}
)::Float64
    1 ≤ i ≤ length(model.m) || throw(
        ArgumentError(
        "PerSiteMoment: atom $i out of range 1:$(length(model.m))"
    )
    )
    return model.m[i]
end

# External terms ------------------------------------------------------------

"""
    ExternalTerm

Abstract supertype for additive non-SCE energy contributions. A concrete
subtype must implement, dispatched on its own type:

- `total_energy(ext, spins) -> Float64`
- `local_energy(ext, spins, i) -> Float64`
- `delta_local_energy(ext, spins, i, S_new) -> Float64`
- `gradient(ext, spins, i) -> SVector{3, Float64}`

`spins` is a `3 × n_atoms` matrix following the same layout as the SCE
energy functions: column `i` is the Cartesian spin direction at supercell
atom `i`.
"""
abstract type ExternalTerm end

"""
Bohr magneton in eV per Tesla: `μ_B / e = ℏ / (2 m_e) ≈ 5.7883818060e-5 eV/T`.
Used to convert `Zeeman` field inputs given in Tesla to the internal
eV/μ_B representation.
"""
const BOHR_MAGNETON_EV_PER_TESLA = 5.7883818060e-5

"""
    Zeeman(field; unit=:eV_per_muB, moments=UniformMoment(1.0)) <: ExternalTerm

Zeeman coupling with energy

    E_Zeeman = - Σ_i m_i (field · S_i)

where `m_i = moment_at(moments, i, spins)` and the *internal* field is in
`eV/μ_B`. The energy is minimised when each spin aligns with `field`.

# Field-unit options

- `unit = :eV_per_muB` (default) — `field` components are already in
  `eV/μ_B`. Pair with `moments` in `μ_B`.
- `unit = :tesla` — `field` components are in Tesla. Internally multiplied
  by `BOHR_MAGNETON_EV_PER_TESLA ≈ 5.7884e-5 eV/(T·μ_B)`. Pair with
  `moments` in `μ_B`; the product gives an eV energy.

Both modes assume `moments` are in `μ_B`. To work in a different unit
system supply pre-converted numbers and use `:eV_per_muB`.

# Fields

- `field::SVector{3, Float64}` — internal field in `eV/μ_B` (after any
  conversion from the constructor's `unit` argument).
- `moments::MomentModel` — site-resolved magnitudes; see `MomentModel`.

# Examples

```julia
# 1 Tesla along +z, Fe (2.2 μ_B) / Rh (0.5 μ_B) sublattices.
moments = PerSiteMoment([i ≤ n_Fe ? 2.2 : 0.5 for i in 1:n_atoms])
z = Zeeman([0.0, 0.0, 1.0]; unit=:tesla, moments)

# Already in eV/μ_B (default).
z = Zeeman([0.0, 0.0, 5.7884e-5]; moments)   # equivalent to 1 T

# Uniform field, unit moment, no extra setup.
z = Zeeman([0.0, 0.0, 0.1])                  # 0.1 eV per unit-spin
```
"""
struct Zeeman{M <: MomentModel} <: ExternalTerm
    field::SVector{3, Float64}
    moments::M
end

function Zeeman(
        field::AbstractVector{<:Real};
        unit::Symbol = :eV_per_muB,
        moments::MomentModel = UniformMoment(1.0)
)
    length(field) == 3 ||
        throw(ArgumentError("Zeeman field must have length 3; got $(length(field))"))
    scale = if unit === :eV_per_muB
        1.0
    elseif unit === :tesla
        BOHR_MAGNETON_EV_PER_TESLA
    else
        throw(
            ArgumentError(
            "Zeeman unit must be :eV_per_muB or :tesla; got :$(unit)"
        )
        )
    end
    return Zeeman(
        SVector{3, Float64}(field[1] * scale, field[2] * scale, field[3] * scale),
        moments
    )
end

function _validate_external_spins(spins::AbstractMatrix{<:Real})
    size(spins, 1) == 3 || throw(
        ArgumentError("spins must have 3 rows (x, y, z); got $(size(spins, 1))")
    )
    return nothing
end

# When a Zeeman is paired with a PerSiteMoment, the per-site vector must
# match the spin matrix's atom count; checked once here so the inner loops
# stay tight.
function _check_moments_match_spins(::UniformMoment, ::AbstractMatrix{<:Real})
    return nothing
end

function _check_moments_match_spins(
        m::PerSiteMoment, spins::AbstractMatrix{<:Real}
)
    length(m.m) == size(spins, 2) || throw(
        ArgumentError(
        "PerSiteMoment length $(length(m.m)) != n_atoms $(size(spins, 2))"
    )
    )
    return nothing
end

"""
    total_energy(ext::Zeeman, spins) -> Float64

`E = -Σ_i m_i (field · S_i)`.
"""
function total_energy(ext::Zeeman, spins::AbstractMatrix{<:Real})::Float64
    _validate_external_spins(spins)
    _check_moments_match_spins(ext.moments, spins)
    Bx, By, Bz = ext.field[1], ext.field[2], ext.field[3]
    E = 0.0
    @inbounds for i in 1:size(spins, 2)
        m = moment_at(ext.moments, i, spins)
        E -= m * (Bx * spins[1, i] + By * spins[2, i] + Bz * spins[3, i])
    end
    return E
end

"""
    local_energy(ext::Zeeman, spins, i) -> Float64

Single-site contribution `-m_i (field · S_i)` for atom `i`.
"""
function local_energy(
        ext::Zeeman, spins::AbstractMatrix{<:Real}, i::Integer
)::Float64
    _validate_external_spins(spins)
    _check_moments_match_spins(ext.moments, spins)
    1 ≤ i ≤ size(spins, 2) ||
        throw(ArgumentError("atom $i out of range 1:$(size(spins, 2))"))
    m = moment_at(ext.moments, i, spins)
    return -m * (ext.field[1] * spins[1, i] +
            ext.field[2] * spins[2, i] +
            ext.field[3] * spins[3, i])
end

"""
    delta_local_energy(ext::Zeeman, spins, i, S_new) -> Float64

Change of `local_energy(i)` when atom `i`'s spin is replaced by `S_new`:

    ΔE = -m_i (field · (S_new - S_old))

assuming `m_i` is independent of `S` (true for `UniformMoment` and
`PerSiteMoment`).
"""
function delta_local_energy(
        ext::Zeeman,
        spins::AbstractMatrix{<:Real},
        i::Integer,
        S_new::AbstractVector{<:Real}
)::Float64
    _validate_external_spins(spins)
    _check_moments_match_spins(ext.moments, spins)
    1 ≤ i ≤ size(spins, 2) ||
        throw(ArgumentError("atom $i out of range 1:$(size(spins, 2))"))
    length(S_new) == 3 ||
        throw(ArgumentError("S_new must have length 3; got $(length(S_new))"))
    m = moment_at(ext.moments, i, spins)
    return -m * (ext.field[1] * (S_new[1] - spins[1, i]) +
            ext.field[2] * (S_new[2] - spins[2, i]) +
            ext.field[3] * (S_new[3] - spins[3, i]))
end

"""
    gradient(ext::Zeeman, spins, i) -> SVector{3, Float64}

`∂(-m_i field · S_i) / ∂S_i = -m_i field`, assuming `m_i` is independent of
`S_i` (true for `UniformMoment` and `PerSiteMoment`). A future
`ClusterExpansionMoment` would need to add `-(field · S_i) · ∂m_i/∂S_i` and
the cross-site contributions where `S_i` enters `m_j`.
"""
function gradient(
        ext::Zeeman, spins::AbstractMatrix{<:Real}, i::Integer
)::SVector{3, Float64}
    _validate_external_spins(spins)
    _check_moments_match_spins(ext.moments, spins)
    1 ≤ i ≤ size(spins, 2) ||
        throw(ArgumentError("atom $i out of range 1:$(size(spins, 2))"))
    m = moment_at(ext.moments, i, spins)
    return -m * ext.field
end
