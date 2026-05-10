# ============================================================================
# Monomial-form energy kernel
# ============================================================================
#
# Equivalent reformulation of the SALC tensor contraction in
# `_energy_from_instances_cached`. Each cluster instance evaluates
#
#     E_inst = jphi_s * mult * (4π)^(N/2) *
#              Σ_{Mf} c_{Mf} Σ_{m₁..m_N} T_{m₁..m_N, Mf} ∏_k Z_{l_k,m_k}(u_{a_k})
#
# Distributing the m-sum and Mf-sum across SALCs/cbcs and translations turns the
# Hamiltonian into a flat sum of scalar monomials in real spherical harmonics.
# Different SALC/cbc/translation contributions that produce the *same* canonical
# (atom-sorted) factor list `[(a_k, l_k, m_k)]` get their coefficients pre-summed
# at build time, so each monomial costs exactly one Z-product per energy call.
#
# Numerical equivalence with the SALC kernel holds to within floating-point
# summation order (CG-tensor partial sums are recombined in a different order).

const _MONOMIAL_COEFF_TOL = 1e-14

"""
    MonomialTable

Flattened scalar-monomial representation of the SCE Hamiltonian (excluding the
constant `j0` reference).

The energy is

    E - E_ref = Σ_t coefficient[t] * ∏_{k=offsets[t]:offsets[t+1]-1}
                                       Z_{ls[k], m_indices[k]}(u_{atoms[k]})

# Fields
- `coefficient`: pre-summed coefficient per monomial (already includes
  `jphi[s] * multiplicity * (4π)^(N/2)`).
- `offsets`: length `n_monomials + 1`. Factors of monomial `t` live in
  `atoms[offsets[t]:offsets[t+1]-1]` (likewise `ls`, `m_indices`).
- `atoms`, `ls`, `m_indices`: flattened factor lists, atom-sorted within each
  monomial.

Per the SCE convention used here each `cbc` has distinct atoms within one
cluster, and translation maps distinct base atoms to distinct supercell atoms,
so each monomial has at most one factor per atom — consumed by
[`_build_monomials_by_atom`](@ref).
"""
struct MonomialTable
    coefficient::Vector{Float64}
    offsets::Vector{Int}
    atoms::Vector{Int}
    ls::Vector{Int}
    m_indices::Vector{Int}
end

@inline _monomial_count(tbl::MonomialTable) = length(tbl.coefficient)

@inline function _max_l_in_table(tbl::MonomialTable)::Int
    m = 0
    @inbounds for l in tbl.ls
        l > m && (m = l)
    end
    return m
end

"""
    build_monomial_table(h; active_bodies=nothing, tol=$(_MONOMIAL_COEFF_TOL)) -> (MonomialTable, monomials_by_atom)

Expand every SALC/cbc/translation contribution into scalar monomials and merge
contributions that share the same canonical `(atom, l, m)` factor list.

`active_bodies` (collection of `Int`, optional): if provided, only `cbc`s with
`length(atoms) ∈ active_bodies` contribute.

`tol`: drop monomials whose accumulated coefficient has magnitude `< tol`.

Returns the table and `monomials_by_atom[a] :: Vector{Int}` — the sorted indices
of monomials whose factor list contains atom `a`. This relies on each monomial
having at most one factor per atom (see [`MonomialTable`](@ref)).
"""
function build_monomial_table(
    h::SCEHamiltonian;
    active_bodies::Union{Nothing, AbstractSet{Int}, AbstractVector{Int}} = nothing,
    tol::Float64 = _MONOMIAL_COEFF_TOL,
)::Tuple{MonomialTable, Vector{Vector{Int}}}
    active_set = active_bodies === nothing ? nothing : Set{Int}(active_bodies)
    acc = Dict{Tuple{Vector{Int}, Vector{Int}, Vector{Int}}, Float64}()

    for (s, group) in enumerate(h.salc_list)
        js = h.jphi[s]
        for cbc in group
            N = length(cbc.atoms)
            active_set === nothing || N in active_set || continue
            scaling = _cluster_scaling(N)
            base_prefactor = js * cbc.multiplicity * scaling
            dims = ntuple(k -> 2 * cbc.ls[k] + 1, N)
            Mf_size = size(cbc.coeff_tensor, N + 1)

            _foreach_translated_instance(h, cbc) do translated_atoms
                perm = sortperm(translated_atoms)
                ls_perm = cbc.ls[perm]
                atoms_perm = translated_atoms[perm]

                for mf_idx in 1:Mf_size
                    c_mf = cbc.coefficient[mf_idx]
                    c_mf == 0.0 && continue
                    for m_tuple in CartesianIndices(dims)
                        T = cbc.coeff_tensor[m_tuple.I..., mf_idx]
                        T == 0.0 && continue
                        coeff = base_prefactor * c_mf * T
                        m_perm = Vector{Int}(undef, N)
                        @inbounds for k in 1:N
                            m_perm[k] = m_tuple.I[perm[k]]
                        end
                        key = (atoms_perm, ls_perm, m_perm)
                        acc[key] = get(acc, key, 0.0) + coeff
                    end
                end
            end
        end
    end

    kept = Tuple{Vector{Int}, Vector{Int}, Vector{Int}}[]
    coeffs = Float64[]
    for (k, v) in acc
        abs(v) < tol && continue
        push!(kept, k)
        push!(coeffs, v)
    end
    # Stable, repo-version-independent ordering for reproducibility of the
    # final floating-point sum.
    order = sortperm(kept; by = k -> (k[1], k[2], k[3]))
    kept = kept[order]
    coeffs = coeffs[order]

    n_mono = length(kept)
    offsets = Vector{Int}(undef, n_mono + 1)
    offsets[1] = 1
    for t in 1:n_mono
        offsets[t + 1] = offsets[t] + length(kept[t][1])
    end
    flat_len = offsets[end] - 1
    flat_atoms = Vector{Int}(undef, flat_len)
    flat_ls = Vector{Int}(undef, flat_len)
    flat_m = Vector{Int}(undef, flat_len)
    for t in 1:n_mono
        atoms, ls, ms = kept[t]
        off = offsets[t] - 1
        @inbounds for k in eachindex(atoms)
            flat_atoms[off + k] = atoms[k]
            flat_ls[off + k] = ls[k]
            flat_m[off + k] = ms[k]
        end
    end
    table = MonomialTable(coeffs, offsets, flat_atoms, flat_ls, flat_m)
    monos_by_atom = _build_monomials_by_atom(table, h.n_atoms)
    return table, monos_by_atom
end

"""
Build per-atom monomial-index lists. Assumes each monomial has at most one
factor per atom (see [`MonomialTable`](@ref)); otherwise the same monomial
index would be inserted multiple times for a single atom.
"""
function _build_monomials_by_atom(
    tbl::MonomialTable,
    n_atoms::Int,
)::Vector{Vector{Int}}
    by_atom = [Int[] for _ in 1:n_atoms]
    @inbounds for t in 1:_monomial_count(tbl)
        for k in tbl.offsets[t]:(tbl.offsets[t + 1] - 1)
            push!(by_atom[tbl.atoms[k]], t)
        end
    end
    return by_atom
end

"""
Sum every monomial contribution against the per-atom Zₗₘ cache. Excludes the
constant `j0` reference.
"""
function _monomial_total_energy(
    tbl::MonomialTable,
    zlm_cache::Matrix{Float64},
)::Float64
    E = 0.0
    coeffs = tbl.coefficient
    offsets = tbl.offsets
    atoms = tbl.atoms
    ls = tbl.ls
    ms = tbl.m_indices
    @inbounds for t in eachindex(coeffs)
        prod = 1.0
        for k in offsets[t]:(offsets[t + 1] - 1)
            prod *= zlm_cache[atoms[k], _zlm_col(ls[k], ms[k])]
        end
        E += coeffs[t] * prod
    end
    return E
end

"""
Sum the contributions of monomials whose indices are listed in `mono_indices`,
typically the monomials that touch a particular atom.
"""
@inline function _monomial_local_energy(
    tbl::MonomialTable,
    zlm_cache::Matrix{Float64},
    mono_indices::AbstractVector{Int},
)::Float64
    E = 0.0
    coeffs = tbl.coefficient
    offsets = tbl.offsets
    atoms = tbl.atoms
    ls = tbl.ls
    ms = tbl.m_indices
    @inbounds for t in mono_indices
        prod = 1.0
        for k in offsets[t]:(offsets[t + 1] - 1)
            prod *= zlm_cache[atoms[k], _zlm_col(ls[k], ms[k])]
        end
        E += coeffs[t] * prod
    end
    return E
end

"""
    monomial_sce_energy(h, spin_directions; table=nothing, tol=$(_MONOMIAL_COEFF_TOL)) -> Float64

Total SCE energy via the monomial-expanded kernel; equivalent to
`sce_energy(h, spin_directions)` up to the order in which floating-point
contributions are summed.

If `table` is supplied, it is reused; otherwise it is built on the fly. The
zlm cache is sized from `_max_l_in_table(table)`.
"""
function monomial_sce_energy(
    h::SCEHamiltonian,
    spin_directions::AbstractMatrix{<:Real};
    table::Union{Nothing, MonomialTable} = nothing,
    tol::Float64 = _MONOMIAL_COEFF_TOL,
)::Float64
    tbl = table === nothing ? first(build_monomial_table(h; tol = tol)) : table
    max_l = _max_l_in_table(tbl)
    zlm = _build_zlm_cache(spin_directions, max_l)
    return _monomial_total_energy(tbl, zlm)
end

# Process-local cache for the monomial-form energy kernel, keyed identically to
# `_DERIVED_CACHE`. Building the monomial table is O(instances × Mf × Πdims) and
# is shared across all JPhiSpinMC instances on the same rank.
struct MonomialKernelCache
    table::MonomialTable
    by_atom::Vector{Vector{Int}}
end

const _MONOMIAL_CACHE = Dict{Tuple{String,NTuple{3,Int},Tuple}, MonomialKernelCache}()

"""
Return the cached `MonomialKernelCache` for `(xml_path, rep, active_bodies)`,
building it from the SALC list on first call.
"""
function _get_or_build_monomial_kernel(
    xml_path::String,
    rep::NTuple{3, Int},
    active_bodies::Union{Nothing, Vector{Int}},
    ham::SCEHamiltonian,
)::MonomialKernelCache
    key_bodies = active_bodies === nothing ? () : Tuple(sort(active_bodies))
    key = (xml_path, rep, key_bodies)
    haskey(_MONOMIAL_CACHE, key) && return _MONOMIAL_CACHE[key]
    tbl, by_atom = build_monomial_table(ham; active_bodies = active_bodies)
    cache = MonomialKernelCache(tbl, by_atom)
    _MONOMIAL_CACHE[key] = cache
    return cache
end
