# --- Spin utilities: Zlm cache, spin proposals, supercell initialization ---

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

"""
Refresh cached `Z_lm` values for one atom from its current spin.
"""
function _update_atom_zlm_cache!(
    zlm_cache::Matrix{Float64},
    atom::Int,
    u::AbstractVector{<:Real},
    max_l::Int,
)
    @inbounds for l in 0:max_l
        @simd for m_idx in 1:(2 * l + 1)
            m = m_idx - l - 1
            zlm_cache[atom, _zlm_col(l, m_idx)] = Zₗₘ_unsafe(l, m, u)
        end
    end
    return nothing
end

"""
Build a per-atom `Z_lm` cache from a spin matrix without requiring a `JPhiSpinMC` instance.
Rows index atoms; columns index `(l, m)` via `_zlm_col`. Useful for standalone full-energy
evaluation, e.g. in global update algorithms or benchmarks.
"""
function _build_zlm_cache(
    spin_directions::AbstractMatrix{<:Real},
    max_l::Int,
)::Matrix{Float64}
    n_atoms = size(spin_directions, 2)
    ncols = (max_l + 1)^2
    zlm_cache = Matrix{Float64}(undef, n_atoms, ncols)
    @inbounds for atom in 1:n_atoms
        _update_atom_zlm_cache!(zlm_cache, atom, @view(spin_directions[:, atom]), max_l)
    end
    return zlm_cache
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
    spins::Matrix{Float64},
    initial_spins::AbstractMatrix{<:Real},
    base_n_atoms::Int,
)
    n_atoms = size(spins, 2)
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
        spins[1, ia] = sx / nrm
        spins[2, ia] = sy / nrm
        spins[3, ia] = sz / nrm
    end
    return nothing
end
