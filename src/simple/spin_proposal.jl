"""
Spin proposals and initial-spin generation for the Simple submodule.

# Proposals

- `_rand_unit_spin(rng)` — uniform unit vector on the sphere (Marsaglia /
  z-then-azimuth method). Use as an i.i.d. proposal for Metropolis or as a
  fallback when a geodesic step degenerates.
- `_propose_spin_geodesic(rng, u, theta_max)` — small-angle rotation of `u`
  by `θ ∈ [-theta_max, theta_max]` in a uniformly-chosen tangent direction.
  Standard local proposal whose acceptance rate is tunable via `theta_max`.

Both return `SVector{3, Float64}` so they compose with the rest of the
energy code without intermediate allocations.

# Initial spins

`init_spins(spec, n_atoms, base_n_atoms; rng)` returns a `3 × n_atoms`
`Matrix{Float64}` from a variety of user-facing specs, all renormalized to
unit columns. Supported forms:

| Spec type | Meaning |
|---|---|
| `Symbol :random` | i.i.d. uniform spins on the sphere |
| `Symbol :ferromagnetic` | all spins aligned with `+ẑ` |
| `Tuple` `(sx, sy, sz)` | every atom aligned with the given direction |
| `AbstractVector{<:Real}` (length 3) | same as Tuple |
| `AbstractMatrix{<:Real}` `(3, base_n_atoms)` | tile across the supercell |
| `AbstractMatrix{<:Real}` `(3, n_atoms)` | already supercell-shaped, use as-is |

When `n_atoms == base_n_atoms` (e.g., `repeat = (1, 1, 1)`) the as-is path
takes precedence; this is unambiguous because both interpretations produce
the same result in that case (one tile = the whole supercell).

`init_spins(params::AbstractDict, n_atoms, base_n_atoms; rng)` reads
`params[:initial_spins]` and delegates, defaulting to `:random` when absent.

# Reproducibility

The `rng` kwarg controls the only source of randomness here. To get
bit-identical results across runs, always pass an explicit `rng`, e.g.
`init_spins(:random, n, n; rng = MersenneTwister(42))`. The default
`Random.default_rng()` reads task-local state and is **not** reproducible
across processes, threads, or even subsequent calls in the same session.
For the strongest guarantee across Julia minor versions, use
`MersenneTwister(seed)` (the stream is officially stability-guaranteed);
`Xoshiro` may evolve between Julia releases.

When this function is invoked from inside the MC engine (M7's
`Carlo.init!`) the caller forwards the seeded `ctx.rng`, so the MC path is
reproducible whenever Carlo is.
"""

using LinearAlgebra: norm, dot
using Random: AbstractRNG, default_rng, rand, randn
using StaticArrays: SVector

# Proposals ----------------------------------------------------------------

"""
    _rand_unit_spin(rng) -> SVector{3, Float64}

Sample a unit vector uniformly on `S²`. Uses the z + uniform-azimuth scheme
so it costs one `rand` + one `cos`/`sin` per call with no rejection.
"""
@inline function _rand_unit_spin(rng::AbstractRNG)::SVector{3, Float64}
    z = 2.0 * rand(rng) - 1.0
    ϕ = 2π * rand(rng)
    r = sqrt(max(0.0, 1.0 - z * z))
    return SVector{3, Float64}(r * cos(ϕ), r * sin(ϕ), z)
end

"""
    _propose_spin_geodesic(rng, u, theta_max) -> SVector{3, Float64}

Propose `u' = cos(θ) u + sin(θ) t` with `t` a random unit tangent at `u`
and `θ ∈ [-theta_max, theta_max]`. `theta_max` is in **radians** (e.g.,
`0.3` rad ≈ 17°, `π` rad gives near-uniform-sphere proposals). For
moderate `theta_max` (~0.3 rad) the Metropolis acceptance is much higher
than i.i.d. uniform spins because the local-energy change is small.

If the random tangent happens to be (numerically) parallel to `u` the step
falls back to a uniform-sphere sample. The `theta_max == 0` case is
short-circuited *before* that fallback so it always returns `u` unchanged.
"""
@inline function _propose_spin_geodesic(
        rng::AbstractRNG, u::SVector{3, Float64}, theta_max::Real
)::SVector{3, Float64}
    theta_max == 0 && return u
    r = SVector{3, Float64}(randn(rng), randn(rng), randn(rng))
    t_unnorm = r - dot(r, u) * u
    nrm = norm(t_unnorm)
    if nrm < 1.0e-14
        return _rand_unit_spin(rng)
    end
    t = t_unnorm / nrm
    θ = Float64(theta_max) * (2.0 * rand(rng) - 1.0)
    return cos(θ) * u + sin(θ) * t
end

# Initial spins ------------------------------------------------------------

function _normalize_direction(v::SVector{3, Float64})
    let n = norm(v)
        n > 0 || throw(ArgumentError("direction has zero norm; cannot normalize"))
        v / n
    end
end

function _fill_aligned(n_atoms::Int, dir::SVector{3, Float64})::Matrix{Float64}
    spins = Matrix{Float64}(undef, 3, n_atoms)
    @inbounds for i in 1:n_atoms
        spins[1, i] = dir[1]
        spins[2, i] = dir[2]
        spins[3, i] = dir[3]
    end
    return spins
end

function _random_unit_spins(rng::AbstractRNG, n_atoms::Int)::Matrix{Float64}
    spins = Matrix{Float64}(undef, 3, n_atoms)
    @inbounds for i in 1:n_atoms
        s = _rand_unit_spin(rng)
        spins[1, i] = s[1]
        spins[2, i] = s[2]
        spins[3, i] = s[3]
    end
    return spins
end

# Tile a 3×base matrix across the supercell. Supercell atom `ia` maps to base
# atom `((ia - 1) % base_n_atoms) + 1`, matching the convention used by
# `_supercell_atom_index` (atoms are laid out in tile-major order).
function _tile_base_matrix(
        base_spins::AbstractMatrix{<:Real}, n_atoms::Int, base_n_atoms::Int
)::Matrix{Float64}
    out = Matrix{Float64}(undef, 3, n_atoms)
    @inbounds for ia in 1:n_atoms
        ib = ((ia - 1) % base_n_atoms) + 1
        v = SVector{3, Float64}(
            Float64(base_spins[1, ib]),
            Float64(base_spins[2, ib]),
            Float64(base_spins[3, ib])
        )
        d = _normalize_direction(v)
        out[1, ia] = d[1]
        out[2, ia] = d[2]
        out[3, ia] = d[3]
    end
    return out
end

function _normalize_supercell_matrix(
        super_spins::AbstractMatrix{<:Real}, n_atoms::Int
)::Matrix{Float64}
    out = Matrix{Float64}(undef, 3, n_atoms)
    @inbounds for i in 1:n_atoms
        v = SVector{3, Float64}(
            Float64(super_spins[1, i]),
            Float64(super_spins[2, i]),
            Float64(super_spins[3, i])
        )
        d = _normalize_direction(v)
        out[1, i] = d[1]
        out[2, i] = d[2]
        out[3, i] = d[3]
    end
    return out
end

"""
    init_spins(spec, n_atoms, base_n_atoms; rng=default_rng()) -> Matrix{Float64}

Build the initial `3 × n_atoms` spin matrix from a user-facing specification.
See the module docstring for the supported spec forms. The result has
unit-norm columns regardless of the input scaling, so callers can pass
non-normalized directions freely.

When `spec::AbstractDict` is given, the implementation reads
`spec[:initial_spins]` and delegates, defaulting to `:random` if the key is
absent.

**Reproducibility**: the `rng` kwarg defaults to `Random.default_rng()`,
which is *not* reproducible across processes or sessions. To get
bit-identical initial spins, pass a seeded RNG explicitly — e.g.
`init_spins(:random, n, n; rng = MersenneTwister(42))`. See the module
docstring's "Reproducibility" section for details.
"""
function init_spins(
        spec::Symbol,
        n_atoms::Int,
        base_n_atoms::Int;
        rng::AbstractRNG = default_rng()
)::Matrix{Float64}
    if spec === :random
        return _random_unit_spins(rng, n_atoms)
    elseif spec === :ferromagnetic
        return _fill_aligned(n_atoms, SVector{3, Float64}(0.0, 0.0, 1.0))
    end
    throw(
        ArgumentError(
        ":initial_spins symbol :$(spec) not supported; expected :random or :ferromagnetic"
    )
    )
end

function init_spins(
        spec::Tuple,
        n_atoms::Int,
        base_n_atoms::Int;
        rng::AbstractRNG = default_rng()
)::Matrix{Float64}
    length(spec) == 3 || throw(
        ArgumentError("initial_spins tuple must have 3 elements; got $(length(spec))")
    )
    dir = _normalize_direction(
        SVector{3, Float64}(Float64(spec[1]), Float64(spec[2]), Float64(spec[3]))
    )
    return _fill_aligned(n_atoms, dir)
end

function init_spins(
        spec::AbstractVector{<:Real},
        n_atoms::Int,
        base_n_atoms::Int;
        rng::AbstractRNG = default_rng()
)::Matrix{Float64}
    length(spec) == 3 || throw(
        ArgumentError(
        "initial_spins vector must have 3 elements; got $(length(spec))"
    )
    )
    dir = _normalize_direction(
        SVector{3, Float64}(Float64(spec[1]), Float64(spec[2]), Float64(spec[3]))
    )
    return _fill_aligned(n_atoms, dir)
end

function init_spins(
        spec::AbstractMatrix{<:Real},
        n_atoms::Int,
        base_n_atoms::Int;
        rng::AbstractRNG = default_rng()
)::Matrix{Float64}
    size(spec, 1) == 3 || throw(
        ArgumentError(
        "initial_spins matrix must have 3 rows; got $(size(spec, 1))"
    )
    )
    ncols = size(spec, 2)
    if ncols == n_atoms
        return _normalize_supercell_matrix(spec, n_atoms)
    elseif ncols == base_n_atoms
        return _tile_base_matrix(spec, n_atoms, base_n_atoms)
    end
    throw(
        ArgumentError(
        "initial_spins matrix has $(ncols) columns; expected $(base_n_atoms) (base) or $(n_atoms) (supercell)"
    )
    )
end

function init_spins(
        params::AbstractDict,
        n_atoms::Int,
        base_n_atoms::Int;
        rng::AbstractRNG = default_rng()
)::Matrix{Float64}
    spec = get(params, :initial_spins, :random)
    return init_spins(spec, n_atoms, base_n_atoms; rng = rng)
end
