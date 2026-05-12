"""
Energy API for the Simple SCE implementation. Four pure functions on
`(h::SpinClusterHamiltonian, spins::AbstractMatrix{<:Real}, ...)`:

- `total_energy(h, spins)`           — sum of every cluster contribution.
- `local_energy(h, spins, i)`        — sum of contributions whose cluster
                                        contains atom `i`.
- `delta_local_energy(h, spins, i, S_new)` — local energy change for a single
                                              spin flip at site `i`.
- `gradient(h, spins, i)`            — `∂(local_energy(i)) / ∂S_i` as a
                                        Cartesian 3-vector.

All functions recompute the tesseral spherical harmonics they need; no state
is kept between calls. This trades performance for legibility — the optimized
side caches Zlm and tensor strides, but the Simple side rebuilds them on
demand so the textual formula matches the code one-to-one.

# Energy formula (one instance)

For a cluster with `N` sites at supercell atoms `atoms[1..N]`, per-site
angular momenta `ls`, intermediate coupling path `Lseq`, final coupling `Lf`,
per-Mf weights `salc_weights`, coupling constant `J`, and multiplicity `m`:

    E_inst = J · (4π)^(N/2) · m ·
             Σ_{Mf=-Lf..Lf} salc_weights[Mf] ·
                 Σ_{m_1..m_N} T_real[m_1, …, m_N; Mf] ·
                              Π_{k=1..N} Z_{ls[k]}^{m_k}(spins[:, atoms[k]])

`T_real` is `h.cg_table[(ls, Lf, Lseq)]`, the tesseral Clebsch-Gordan tensor.
`Z_l^m` are the real (tesseral) spherical harmonics produced by SpheriCart
with `:L2` normalisation. See Magesty technical notes for the SCE derivation:
<https://Tomonori-Tanaka.github.io/Magesty.jl/technical_notes/>.
"""

using SpheriCart: SphericalHarmonics, compute, compute_with_gradients
using StaticArrays: SVector

# Linear index of (l, m) in the (l_max+1)^2 SpheriCart vector layout.
@inline _zlm_index(l::Integer, m::Integer) = l * l + l + m + 1

# (4π)^(N/2) prefactor common to every cluster contribution; matches the
# convention used by Magesty.Optimize.design_matrix_energy_element.
@inline _cluster_scaling(N::Integer) = (4 * pi)^(N / 2)

@inline function _spin_svector(spins::AbstractMatrix{<:Real}, i::Integer)
    return SVector{3, Float64}(spins[1, i], spins[2, i], spins[3, i])
end

# Evaluate Z_l^m for every atom in the supercell once and return a
# K×n_atoms matrix where K = (max_l + 1)^2. Column `a` is the SpheriCart
# output for spins[:, a].
function _compute_zlm_all(
        sph::SphericalHarmonics, spins::AbstractMatrix{<:Real}, K::Int, n_atoms::Int
)::Matrix{Float64}
    zlm = Matrix{Float64}(undef, K, n_atoms)
    for a in 1:n_atoms
        z = compute(sph, _spin_svector(spins, a))
        @inbounds for k in 1:K
            zlm[k, a] = z[k]
        end
    end
    return zlm
end

# Energy contribution of a single ClusterInstance, given a precomputed
# `zlm[(max_l+1)^2, n_atoms]` matrix. Loops over the cluster's local
# (m_1, …, m_N, Mf) index space and contracts T_real with the product of Zlm.
function _instance_energy(
        inst::ClusterInstance, zlm::AbstractMatrix{Float64}, cg_table::CGTable
)::Float64
    T = cg_table[(inst.ls, inst.Lf, inst.Lseq)]
    ls = inst.ls
    Lf = inst.Lf
    weights = inst.salc_weights
    atoms = inst.atoms
    N = length(ls)
    leading_dims = ntuple(k -> 2 * ls[k] + 1, N)

    total = 0.0
    for Mf_idx in 1:(2 * Lf + 1)
        Mf_inner = 0.0
        for I in CartesianIndices(leading_dims)
            t = T[I, Mf_idx]
            prod_Z = 1.0
            @inbounds for k in 1:N
                l = ls[k]
                m = -l + (I[k] - 1)
                prod_Z *= zlm[_zlm_index(l, m), atoms[k]]
            end
            Mf_inner += t * prod_Z
        end
        total += weights[Mf_idx] * Mf_inner
    end
    return inst.J * _cluster_scaling(N) * inst.multiplicity * total
end

# Cartesian gradient of one ClusterInstance's energy with respect to S_i,
# where atom `i` appears at position `site_k` in `inst.atoms`. The product
# Π_k Z_{ls[k]}^{m_k}(S_{atoms[k]}) becomes
#     ∂Z_{ls[site_k]}^{m_{site_k}}(S_i)/∂S_i · Π_{k ≠ site_k} Z_{ls[k]}^{m_k}(...)
# Atoms within a cluster are assumed to be distinct; if `i` appeared at more
# than one site, the product rule would need a sum over occurrences.
function _instance_gradient(
        inst::ClusterInstance,
        site_k::Int,
        zlm::AbstractMatrix{Float64},
        dzlm_i::AbstractVector{SVector{3, Float64}},
        cg_table::CGTable
)::SVector{3, Float64}
    T = cg_table[(inst.ls, inst.Lf, inst.Lseq)]
    ls = inst.ls
    Lf = inst.Lf
    weights = inst.salc_weights
    atoms = inst.atoms
    N = length(ls)
    leading_dims = ntuple(k -> 2 * ls[k] + 1, N)

    g = SVector{3, Float64}(0.0, 0.0, 0.0)
    for Mf_idx in 1:(2 * Lf + 1)
        Mf_grad = SVector{3, Float64}(0.0, 0.0, 0.0)
        for I in CartesianIndices(leading_dims)
            t = T[I, Mf_idx]
            prod_other = 1.0
            @inbounds for k in 1:N
                k == site_k && continue
                l = ls[k]
                m = -l + (I[k] - 1)
                prod_other *= zlm[_zlm_index(l, m), atoms[k]]
            end
            l_k = ls[site_k]
            m_k = -l_k + (I[site_k] - 1)
            dZ = dzlm_i[_zlm_index(l_k, m_k)]
            Mf_grad = Mf_grad + (t * prod_other) .* dZ
        end
        g = g + weights[Mf_idx] .* Mf_grad
    end
    return inst.J * _cluster_scaling(N) * inst.multiplicity .* g
end

function _validate_spin_matrix(h::SpinClusterHamiltonian, spins::AbstractMatrix{<:Real})
    size(spins, 1) == 3 || throw(
        ArgumentError("spins must have 3 rows (x, y, z); got $(size(spins, 1))")
    )
    size(spins, 2) == h.n_atoms || throw(
        ArgumentError(
        "spins has $(size(spins, 2)) columns; expected n_atoms=$(h.n_atoms)"
    )
    )
    return nothing
end

"""
    total_energy(h, spins) -> Float64

Total SCE energy ``E = Σ_{inst} E_inst`` for the supercell spin configuration
`spins` (a 3×n_atoms matrix of Cartesian spin directions). The XML's `j0`
(`ReferenceEnergy`) constant is *not* included — this package is for MC
sampling where only ΔE matters.
"""
function total_energy(
        h::SpinClusterHamiltonian, spins::AbstractMatrix{<:Real}
)::Float64
    _validate_spin_matrix(h, spins)
    sph = SphericalHarmonics(h.max_l)
    K = (h.max_l + 1)^2
    zlm = _compute_zlm_all(sph, spins, K, h.n_atoms)
    E = 0.0
    for inst in h.instances
        E += _instance_energy(inst, zlm, h.cg_table)
    end
    return E
end

"""
    local_energy(h, spins, i) -> Float64

Sum of `E_inst` over every cluster that touches supercell atom `i`. Each
cluster contributes its full energy (no division by `N`), so for a
uniform-N Hamiltonian the identity `Σ_i local_energy(i) = N · total_energy`
holds; for mixed bodies the relation generalizes to
`Σ_i local_energy(i) = Σ_inst body(inst) · E_inst`.
"""
function local_energy(
        h::SpinClusterHamiltonian, spins::AbstractMatrix{<:Real}, i::Integer
)::Float64
    _validate_spin_matrix(h, spins)
    1 ≤ i ≤ h.n_atoms ||
        throw(ArgumentError("atom $i out of range 1:$(h.n_atoms)"))
    sph = SphericalHarmonics(h.max_l)
    K = (h.max_l + 1)^2
    zlm = _compute_zlm_all(sph, spins, K, h.n_atoms)
    E = 0.0
    for idx in h.atom_to_instance_indices[i]
        E += _instance_energy(h.instances[idx], zlm, h.cg_table)
    end
    return E
end

"""
    delta_local_energy(h, spins, i, S_new) -> Float64

Change in local energy at site `i` when its spin is replaced by `S_new`:
`local_energy_after - local_energy_before`. Only clusters containing atom
`i` change, and only column `i` of the Zlm cache needs to be updated; this
function exploits both for clarity (not yet for performance).
"""
function delta_local_energy(
        h::SpinClusterHamiltonian,
        spins::AbstractMatrix{<:Real},
        i::Integer,
        S_new::AbstractVector{<:Real}
)::Float64
    _validate_spin_matrix(h, spins)
    1 ≤ i ≤ h.n_atoms ||
        throw(ArgumentError("atom $i out of range 1:$(h.n_atoms)"))
    length(S_new) == 3 ||
        throw(ArgumentError("S_new must have length 3; got $(length(S_new))"))
    sph = SphericalHarmonics(h.max_l)
    K = (h.max_l + 1)^2
    zlm_old = _compute_zlm_all(sph, spins, K, h.n_atoms)
    zlm_new = copy(zlm_old)
    z_i = compute(sph, SVector{3, Float64}(S_new[1], S_new[2], S_new[3]))
    @inbounds for k in 1:K
        zlm_new[k, i] = z_i[k]
    end
    delta = 0.0
    for idx in h.atom_to_instance_indices[i]
        inst = h.instances[idx]
        delta += _instance_energy(inst, zlm_new, h.cg_table) -
                 _instance_energy(inst, zlm_old, h.cg_table)
    end
    return delta
end

"""
    gradient(h, spins, i) -> SVector{3, Float64}

Unconstrained Cartesian gradient `∂(local_energy(h, spins, i)) / ∂S_i`,
treating the three components of `S_i` as independent (no `|S| = 1`
projection). Each cluster containing `i` contributes one term where
`Z_{ls[k]}^{m_k}(S_i)` at the site-`k` factor of the product is replaced
by its 3-vector derivative `∂Z/∂S_i`. Atoms within a single cluster are
assumed distinct.

Callers needing a spherical (tangent-plane) gradient should project this
result onto the plane orthogonal to `S_i` at the point of evaluation.
"""
function gradient(
        h::SpinClusterHamiltonian, spins::AbstractMatrix{<:Real}, i::Integer
)::SVector{3, Float64}
    _validate_spin_matrix(h, spins)
    1 ≤ i ≤ h.n_atoms ||
        throw(ArgumentError("atom $i out of range 1:$(h.n_atoms)"))
    sph = SphericalHarmonics(h.max_l)
    K = (h.max_l + 1)^2
    zlm = _compute_zlm_all(sph, spins, K, h.n_atoms)
    _, dzlm_i = compute_with_gradients(sph, _spin_svector(spins, i))
    g = SVector{3, Float64}(0.0, 0.0, 0.0)
    for idx in h.atom_to_instance_indices[i]
        inst = h.instances[idx]
        # Find every position in inst.atoms equal to i. For typical SCE
        # clusters there is exactly one; if zero, the lookup index would be
        # wrong (so we skip), if more than one we sum (product rule).
        for site_k in eachindex(inst.atoms)
            inst.atoms[site_k] == i || continue
            g = g + _instance_gradient(inst, site_k, zlm, dzlm_i, h.cg_table)
        end
    end
    return g
end
