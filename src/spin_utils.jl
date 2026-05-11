# --- Spin utilities: Zlm cache, spin proposals, supercell initialization ---

# Generic accessors so energy kernels accept either Matrix{Float64} (3 × n)
# or Vector{SVector{3,Float64}} for the spin storage.
@inline _n_spins(s::AbstractMatrix)::Int = size(s, 2)
@inline _n_spins(s::AbstractVector{<:SVector{3}})::Int = length(s)

@inline _spin_at(s::AbstractMatrix, atom::Int) = @view s[:, atom]
@inline _spin_at(s::AbstractVector{<:SVector{3}}, atom::Int) = @inbounds s[atom]

"""
Convert internal `Vector{SVector{3,Float64}}` storage to `Matrix{Float64}` (3 × n)
for serialization, the public `:initial_spins` parameter, and other Matrix-shaped
boundaries.
"""
function _spins_to_matrix(spins::AbstractVector{<:SVector{3,<:Real}})::Matrix{Float64}
    n = length(spins)
    M = Matrix{Float64}(undef, 3, n)
    @inbounds for i in 1:n
        s = spins[i]
        M[1, i] = s[1]; M[2, i] = s[2]; M[3, i] = s[3]
    end
    return M
end

"""
Inverse of [`_spins_to_matrix`](@ref): build a `Vector{SVector{3,Float64}}` from a
`3 × n` matrix.  Throws if the row count is not 3.
"""
function _matrix_to_spins(M::AbstractMatrix{<:Real})::Vector{SVector{3,Float64}}
    size(M, 1) == 3 || throw(ArgumentError("spin matrix must have 3 rows, got $(size(M,1))"))
    n = size(M, 2)
    out = Vector{SVector{3,Float64}}(undef, n)
    @inbounds for i in 1:n
        out[i] = SVector{3,Float64}(M[1, i], M[2, i], M[3, i])
    end
    return out
end

"""
Return the maximum cluster size among all instances.
"""
function _max_sites_in_instances(instances::Vector{ClusterInstance})::Int
    m = 1
    for inst in instances
        n = length(inst.atoms)
        if n > m
            m = n
        end
    end
    return m
end

"""
Return the maximum angular-momentum degree `l` across all instances.
"""
function _max_l_in_instances(instances::Vector{ClusterInstance})::Int
    m = 0
    for inst in instances
        for l in inst.cbc.ls
            m = max(m, l)
        end
    end
    return m
end

"""
Map `(l, m_idx)` to a contiguous cache column index.
"""
@inline _zlm_col(l::Int, m_idx::Int)::Int = l * l + m_idx

"""
Allocate per-atom cache for all real spherical harmonics up to `max_l`.
"""
function _alloc_zlm_cache(n_atoms::Int, max_l::Int)::Matrix{Float64}
    # sum_{l=0}^{L} (2l+1) = (L+1)^2
    return zeros(Float64, n_atoms, (max_l + 1)^2)
end

# SpheriCart's flat output ordering `l*(l+1) + m + 1` (m = -l..+l) matches
# `_zlm_col(l, m_idx) = l² + m_idx` (m_idx = 1..2l+1) value-for-value, so the
# cache columns can be written straight through without remapping.
# Bit-exact agreement with Magesty's `Zₗₘ_unsafe` is verified in
# docs/zlm_convention_vs_sphericart.md (max |Δ| ≤ 3.3e-16 for l ≤ 3) for
# SpheriCart's default `:L2` normalisation, which we rely on here.
#
# Note: SpheriCart's `STATIC=true` (SVector return) only holds for `max_l ≤ 15`.
# Above that, `compute(sph, u)` allocates a `Vector{Float64}` and this code path
# regresses to a per-call heap allocation. The SCE models we support keep
# `max_l ≤ 3`, so this limit is well clear; revisit if it ever stops being true.

"""
Refresh cached `Z_lm` values for one atom from its current spin.
"""
function _update_atom_zlm_cache!(
    zlm_cache::Matrix{Float64},
    atom::Int,
    u::AbstractVector{<:Real},
    sph::SphericalHarmonics,
)
    y = compute(sph, SVector{3,Float64}(u[1], u[2], u[3]))
    @inbounds @simd for c in eachindex(y)
        zlm_cache[atom, c] = y[c]
    end
    return nothing
end

# SVector overload avoids a temporary copy when callers already have a unit vector
# as `SVector{3,Float64}` (e.g. `mc.spins[atom]`).
function _update_atom_zlm_cache!(
    zlm_cache::Matrix{Float64},
    atom::Int,
    u::SVector{3,Float64},
    sph::SphericalHarmonics,
)
    y = compute(sph, u)
    @inbounds @simd for c in eachindex(y)
        zlm_cache[atom, c] = y[c]
    end
    return nothing
end

"""
Build a per-atom `Z_lm` cache from a spin matrix or `Vector{SVector{3,Float64}}`.
Rows index atoms; columns index `(l, m)` via `_zlm_col`. Useful for standalone full-energy
evaluation, e.g. in global update algorithms or benchmarks.
"""
function _build_zlm_cache(
    spin_directions::AbstractVector{<:SVector{3,<:Real}},
    max_l::Int,
)::Matrix{Float64}
    n_atoms = length(spin_directions)
    ncols = (max_l + 1)^2
    zlm_cache = Matrix{Float64}(undef, n_atoms, ncols)
    sph = SphericalHarmonics(max_l)
    # `compute!` writes a (n_atoms × ncols) matrix in one batched call.
    compute!(zlm_cache, sph, spin_directions)
    return zlm_cache
end

function _build_zlm_cache(
    spin_directions::AbstractMatrix{<:Real},
    max_l::Int,
)::Matrix{Float64}
    size(spin_directions, 1) == 3 || throw(ArgumentError(
        "spin matrix must have 3 rows, got $(size(spin_directions, 1))"))
    n_atoms = size(spin_directions, 2)
    spins_sv = Vector{SVector{3,Float64}}(undef, n_atoms)
    @inbounds for ia in 1:n_atoms
        spins_sv[ia] = SVector{3,Float64}(spin_directions[1, ia],
                                          spin_directions[2, ia],
                                          spin_directions[3, ia])
    end
    return _build_zlm_cache(spins_sv, max_l)
end

"""
Sample a random unit vector uniformly on the sphere.
"""
@inline function _rand_unit_spin(rng)
    z = 2.0 * rand(rng) - 1.0
    ϕ = 2π * rand(rng)
    r = sqrt(max(0.0, 1.0 - z^2))
    return r * cos(ϕ), r * sin(ϕ), z
end

"""
    _propose_spin_geodesic(rng, ux, uy, uz, theta_max)

Unit-vector proposal `u' = cos(θ) u + sin(θ) t` with `t` a random unit tangent at `u` and
`θ` uniform in `[-theta_max, theta_max]`. For moderate `theta_max`, moves stay close to the
current direction and Metropolis acceptance is typically much higher than i.i.d. uniform spins.
"""
@inline function _propose_spin_geodesic(
    rng,
    ux::Float64,
    uy::Float64,
    uz::Float64,
    theta_max::Float64,
)
    rx = randn(rng)
    ry = randn(rng)
    rz = randn(rng)
    dot = rx * ux + ry * uy + rz * uz
    tx = rx - dot * ux
    ty = ry - dot * uy
    tz = rz - dot * uz
    nrm = hypot(tx, ty, tz)
    if nrm < 1e-14
        return _rand_unit_spin(rng)
    end
    invn = 1.0 / nrm
    tx *= invn
    ty *= invn
    tz *= invn
    θ = theta_max * (2.0 * rand(rng) - 1.0)
    c = cos(θ)
    s = sin(θ)
    return c * ux + s * tx, c * uy + s * ty, c * uz + s * tz
end

"""
    _tile_base_spins!(spins, initial_spins, base_n_atoms)

Fill the supercell spin matrix `spins` (3 × n_atoms) by tiling `initial_spins`
(3 × base_n_atoms).  The tiling follows the same atom-index convention as
`supercell_atom_index`: supercell atom `ia` maps to base atom
`((ia-1) % base_n_atoms) + 1`.  Each column of `initial_spins` is
renormalized to a unit vector before writing.
"""
function _tile_base_spins!(
    spins::AbstractVector{SVector{3,Float64}},
    initial_spins::AbstractMatrix{<:Real},
    base_n_atoms::Int,
)
    n_atoms = length(spins)
    size(initial_spins) == (3, base_n_atoms) || throw(ArgumentError(
        "initial_spins must be a 3×$(base_n_atoms) matrix, got $(size(initial_spins))",
    ))
    for ia in 1:n_atoms
        ib = ((ia - 1) % base_n_atoms) + 1
        sx = Float64(initial_spins[1, ib])
        sy = Float64(initial_spins[2, ib])
        sz = Float64(initial_spins[3, ib])
        nrm = hypot(sx, sy, sz)
        nrm > 0 || throw(ArgumentError("initial_spins column $ib has zero norm"))
        spins[ia] = SVector(sx / nrm, sy / nrm, sz / nrm)
    end
    return nothing
end
