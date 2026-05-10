# Template-based energy kernel: stores only base-cell ClusterInstances and
# reconstructs supercell atom indices on-the-fly during sweep!, avoiding the
# O(repeat_volume) memory blow-up of the full _build_cluster_instances path.

struct BaseClusterInstance
    base_atoms::Vector{Int}             # base-cell atom index for each factor
    tile_deltas::Vector{NTuple{3,Int}}  # relative tile offset of each factor from pivot (factor 1)
    ls::Vector{Int}                     # angular-momentum indices per factor
    cbc_coefficient::Vector{Float64}    # cbc.coefficient copy (Mf dimension)
    prefactor::Float64
    dims::Vector{Int}                   # 2*ls[k]+1
    strides::Vector{Int}
    coeff_flat::Vector{Float64}
    Mf_size::Int
end

struct RelatedBaseCluster
    inst_idx::Int
    pivot_k::Int  # which factor k of the instance has base_atoms[pivot_k] == b
end

# N=2-specialized variant: SVector fields enable stack allocation and unrolled
# access in the hot contraction kernel. 2-body clusters dominate in many systems
# (e.g. bccFe), so this specialization is profitable.
struct BaseClusterInstance2
    base_atoms::SVector{2,Int}
    tile_deltas::SVector{2,NTuple{3,Int}}
    ls::SVector{2,Int}
    cbc_coefficient::Vector{Float64}
    prefactor::Float64
    dims::SVector{2,Int}
    strides::SVector{3,Int}                 # N+1 = 3
    coeff_flat::Vector{Float64}
    Mf_size::Int
end

struct LocalEnergyTemplate
    base_instances::Vector{BaseClusterInstance}                    # N ≥ 3
    base_instances2::Vector{BaseClusterInstance2}                  # N = 2
    related_by_base_atom::Vector{Vector{RelatedBaseCluster}}       # → base_instances
    related2_by_base_atom::Vector{Vector{RelatedBaseCluster}}      # → base_instances2
end

# Iterate over unique base-cell cluster instances for `cbc` (ti=tj=tk=0 only).
# Calls f(base_atoms, w_list) where:
#   base_atoms[k] = h.map_sym[cbc.atoms[k], t]
#   w_list[k]     = (w1, w2, w3) tile offsets from fractional position differences
# Deduplication matches _foreach_translated_instance at ti=tj=tk=0.
function _foreach_base_instance(f, h::SCEHamiltonian, cbc)
    n1, n2, n3 = h.repeat
    n1f, n2f, n3f = Float64(n1), Float64(n2), Float64(n3)
    N = length(cbc.atoms)
    seen = Set{Tuple{Vector{Int}, Vector{Int}}}()
    for t in 1:h.n_trans
        translated_base = Int[h.map_sym[a, t] for a in cbc.atoms]
        p_ref = h.pos_frac[:, translated_base[1]]
        f_ref = (p_ref[1] * n1f, p_ref[2] * n2f, p_ref[3] * n3f)
        w_list = Vector{NTuple{3,Int}}(undef, N)
        translated_atoms = Vector{Int}(undef, N)
        for (k, ba) in enumerate(translated_base)
            p = h.pos_frac[:, ba]
            w1 = round(Int, p[1] * n1f - f_ref[1])
            w2 = round(Int, p[2] * n2f - f_ref[2])
            w3 = round(Int, p[3] * n3f - f_ref[3])
            w_list[k] = (w1, w2, w3)
            translated_atoms[k] = supercell_atom_index(
                ba, mod(w1, n1), mod(w2, n2), mod(w3, n3),
                h.base_n_atoms, h.repeat,
            )
        end
        # Dedup key mirrors _foreach_translated_instance at (ti,tj,tk)=(0,0,0).
        atoms_sorted = sort(translated_atoms)
        pair = (atoms_sorted, cbc.ls)
        pair in seen && continue
        push!(seen, pair)
        f(translated_base, w_list)
    end
end

"""
Build a `LocalEnergyTemplate` holding one `BaseClusterInstance` per unique
base-cell cluster, with per-atom lookup tables for on-the-fly supercell index
reconstruction during `sweep!`.
"""
function build_local_energy_template(h::SCEHamiltonian)::LocalEnergyTemplate
    base_instances = BaseClusterInstance[]
    base_instances2 = BaseClusterInstance2[]
    related_by_base_atom = [RelatedBaseCluster[] for _ in 1:h.base_n_atoms]
    related2_by_base_atom = [RelatedBaseCluster[] for _ in 1:h.base_n_atoms]

    coeff_flat_cache = Dict{UInt, Vector{Float64}}()

    for (s, group) in enumerate(h.salc_list)
        js = h.jphi[s]
        for cbc in group
            scaling = _cluster_scaling(length(cbc.atoms))
            N_cbc = length(cbc.atoms)
            inst_dims = [2 * l + 1 for l in cbc.ls]
            inst_strides = _compute_instance_strides(cbc.ls)
            inst_Mf_size = size(cbc.coeff_tensor, N_cbc + 1)
            inst_coeff_flat = get!(coeff_flat_cache, objectid(cbc)) do
                vec(collect(Float64, cbc.coeff_tensor))
            end
            prefactor = js * cbc.multiplicity * scaling

            _foreach_base_instance(h, cbc) do base_atoms, w_list
                # tile_deltas[k] = offset of factor k relative to factor 1 (pivot)
                w1_ref, w2_ref, w3_ref = w_list[1]
                tile_deltas = NTuple{3,Int}[
                    (w_list[k][1] - w1_ref, w_list[k][2] - w2_ref, w_list[k][3] - w3_ref)
                    for k in 1:N_cbc
                ]

                if N_cbc == 2
                    inst2 = BaseClusterInstance2(
                        SVector{2,Int}(base_atoms[1], base_atoms[2]),
                        SVector{2,NTuple{3,Int}}(tile_deltas[1], tile_deltas[2]),
                        SVector{2,Int}(cbc.ls[1], cbc.ls[2]),
                        collect(Float64, cbc.coefficient),
                        prefactor,
                        SVector{2,Int}(inst_dims[1], inst_dims[2]),
                        SVector{3,Int}(inst_strides[1], inst_strides[2], inst_strides[3]),
                        inst_coeff_flat,
                        inst_Mf_size,
                    )
                    push!(base_instances2, inst2)
                    inst_idx = length(base_instances2)
                    for k in 1:2
                        b = base_atoms[k]
                        push!(related2_by_base_atom[b], RelatedBaseCluster(inst_idx, k))
                    end
                else
                    inst = BaseClusterInstance(
                        copy(base_atoms),
                        tile_deltas,
                        collect(Int, cbc.ls),
                        collect(Float64, cbc.coefficient),
                        prefactor,
                        inst_dims,
                        inst_strides,
                        inst_coeff_flat,
                        inst_Mf_size,
                    )
                    push!(base_instances, inst)
                    inst_idx = length(base_instances)
                    for k in 1:N_cbc
                        b = base_atoms[k]
                        push!(related_by_base_atom[b], RelatedBaseCluster(inst_idx, k))
                    end
                end
            end
        end
    end

    return LocalEnergyTemplate(
        base_instances, base_instances2,
        related_by_base_atom, related2_by_base_atom,
    )
end

"""
Compute the tile coordinates (ti, tj, tk) of supercell atom `i`.
Inverse of `supercell_atom_index`.
"""
@inline function _tile_coords(i::Int, base_n::Int, repeat::NTuple{3,Int})::NTuple{3,Int}
    offset = (i - 1) ÷ base_n
    n1, n2 = repeat[1], repeat[2]
    tk = offset ÷ (n1 * n2)
    tj = (offset % (n1 * n2)) ÷ n1
    ti = offset % n1
    return (ti, tj, tk)
end

# Full contraction of a BaseClusterInstance against the zlm_cache.
# Used when changed_atom is not found in atoms (fall-through case).
@inline function _tensor_contract_template_cached(
    inst::BaseClusterInstance,
    atoms::AbstractVector{Int},
    zlm_cache::Matrix{Float64},
)::Float64
    N = length(inst.base_atoms)
    tensor_result = 0.0
    Mf_size = inst.Mf_size
    coeff_flat = inst.coeff_flat
    total_spatial = inst.strides[N + 1]

    for mf_idx in 1:Mf_size
        mf_contribution = 0.0
        base_mf = 1 + (mf_idx - 1) * total_spatial
        for combo_id in 0:(total_spatial - 1)
            product = 1.0
            tmp = combo_id
            @inbounds for k in 1:N
                d = inst.dims[k]
                m_idx = tmp % d + 1
                tmp ÷= d
                atom = atoms[k]
                l = inst.ls[k]
                product *= zlm_cache[atom, _zlm_col(l, m_idx)]
            end
            mf_contribution += coeff_flat[base_mf + combo_id] * product
        end
        tensor_result += inst.cbc_coefficient[mf_idx] * mf_contribution
    end
    return tensor_result
end

"""
Delta-energy tensor contraction for a `BaseClusterInstance` with the changed atom
identified by `changed_atom`. Uses preallocated `other_sites_buf` / `cart_idx_buf`.
"""
@inline function _tensor_contract_template_changed!(
    other_sites_buf::AbstractVector{Int},
    cart_idx_buf::AbstractVector{Int},
    inst::BaseClusterInstance,
    atoms::AbstractVector{Int},
    zlm_cache::Matrix{Float64},
    changed_atom::Int,
)::Float64
    N = length(inst.base_atoms)
    sitepos = 0
    @inbounds for k in 1:N
        if atoms[k] == changed_atom
            sitepos = k
            break
        end
    end
    if sitepos == 0
        return _tensor_contract_template_cached(inst, atoms, zlm_cache)
    end

    changed_l = inst.ls[sitepos]
    n_other = 0
    @inbounds for s in 1:N
        if s != sitepos
            n_other += 1
            other_sites_buf[n_other] = s
        end
    end

    strides = inst.strides
    stride_changed = strides[sitepos]
    dims_sitepos = 2 * changed_l + 1
    changed_col_base = changed_l * changed_l

    Mf_size = inst.Mf_size
    coeff_flat = inst.coeff_flat
    tensor_result = 0.0

    if n_other == 0
        @inbounds for mf_idx in 1:Mf_size
            mf_contribution = 0.0
            base_mf = 1 + (mf_idx - 1) * strides[N + 1]
            @simd for mchg_idx in 1:dims_sitepos
                mf_contribution +=
                    coeff_flat[base_mf + (mchg_idx - 1) * stride_changed] *
                    zlm_cache[changed_atom, changed_col_base + mchg_idx]
            end
            tensor_result += inst.cbc_coefficient[mf_idx] * mf_contribution
        end
        return tensor_result
    end

    total_combos = 1
    @inbounds for j in 1:n_other
        total_combos *= inst.dims[other_sites_buf[j]]
    end

    @inbounds for mf_idx in 1:Mf_size
        mf_contribution = 0.0
        base_mf = 1 + (mf_idx - 1) * strides[N + 1]
        for combo_id in 0:(total_combos - 1)
            tmp = combo_id
            @inbounds for pos in 1:n_other
                d = inst.dims[other_sites_buf[pos]]
                r = tmp % d
                tmp = tmp ÷ d
                cart_idx_buf[pos] = r + 1
            end
            product_other = 1.0
            base_without_changed = base_mf
            @inbounds for pos in 1:n_other
                site = other_sites_buf[pos]
                m_idx = cart_idx_buf[pos]
                l = inst.ls[site]
                atom = atoms[site]
                product_other *= zlm_cache[atom, _zlm_col(l, m_idx)]
                base_without_changed += (m_idx - 1) * strides[site]
            end
            inner = 0.0
            @simd for mchg_idx in 1:dims_sitepos
                inner +=
                    coeff_flat[base_without_changed + (mchg_idx - 1) * stride_changed] *
                    zlm_cache[changed_atom, changed_col_base + mchg_idx]
            end
            mf_contribution += product_other * inner
        end
        tensor_result += inst.cbc_coefficient[mf_idx] * mf_contribution
    end

    return tensor_result
end

# N=2-specialized contraction kernels. Avoid the n_other / cart_idx_buf
# bookkeeping (n_other ≡ 1) and benefit from SVector stack allocation.

@inline function _tensor_contract_template2_cached(
    inst::BaseClusterInstance2,
    a1::Int, a2::Int,
    zlm_cache::Matrix{Float64},
)::Float64
    l1 = inst.ls[1]; l2 = inst.ls[2]
    d1 = inst.dims[1]; d2 = inst.dims[2]
    s1 = inst.strides[1]; s2 = inst.strides[2]
    total_spatial = inst.strides[3]
    col1_base = l1 * l1
    col2_base = l2 * l2
    Mf_size = inst.Mf_size
    coeff_flat = inst.coeff_flat
    result = 0.0
    @inbounds for mf_idx in 1:Mf_size
        base_mf = 1 + (mf_idx - 1) * total_spatial
        mf_contribution = 0.0
        for m1 in 1:d1
            z1 = zlm_cache[a1, col1_base + m1]
            base_m1 = base_mf + (m1 - 1) * s1
            inner = 0.0
            @simd for m2 in 1:d2
                inner +=
                    coeff_flat[base_m1 + (m2 - 1) * s2] *
                    zlm_cache[a2, col2_base + m2]
            end
            mf_contribution += z1 * inner
        end
        result += inst.cbc_coefficient[mf_idx] * mf_contribution
    end
    return result
end

@inline function _tensor_contract_template2_changed!(
    inst::BaseClusterInstance2,
    a1::Int, a2::Int,
    zlm_cache::Matrix{Float64},
    changed_atom::Int,
)::Float64
    if a1 == changed_atom
        sitepos = 1
    elseif a2 == changed_atom
        sitepos = 2
    else
        return _tensor_contract_template2_cached(inst, a1, a2, zlm_cache)
    end

    if sitepos == 1
        other_atom = a2
        l_chg = inst.ls[1]; l_oth = inst.ls[2]
        d_chg = inst.dims[1]; d_oth = inst.dims[2]
        s_chg = inst.strides[1]; s_oth = inst.strides[2]
    else
        other_atom = a1
        l_chg = inst.ls[2]; l_oth = inst.ls[1]
        d_chg = inst.dims[2]; d_oth = inst.dims[1]
        s_chg = inst.strides[2]; s_oth = inst.strides[1]
    end
    chg_col_base = l_chg * l_chg
    oth_col_base = l_oth * l_oth
    total_spatial = inst.strides[3]
    Mf_size = inst.Mf_size
    coeff_flat = inst.coeff_flat
    result = 0.0
    @inbounds for mf_idx in 1:Mf_size
        base_mf = 1 + (mf_idx - 1) * total_spatial
        mf_contribution = 0.0
        for m_oth in 1:d_oth
            z_oth = zlm_cache[other_atom, oth_col_base + m_oth]
            base_no_chg = base_mf + (m_oth - 1) * s_oth
            inner = 0.0
            @simd for m_chg in 1:d_chg
                inner +=
                    coeff_flat[base_no_chg + (m_chg - 1) * s_chg] *
                    zlm_cache[changed_atom, chg_col_base + m_chg]
            end
            mf_contribution += z_oth * inner
        end
        result += inst.cbc_coefficient[mf_idx] * mf_contribution
    end
    return result
end

# _template_local_energy! is defined in JPhiMagestyCarlo.jl after JPhiSpinMC,
# so that mc::JPhiSpinMC can be used as the type annotation to avoid boxing of
# Union{Nothing,LocalEnergyTemplate} fields in the hot sweep! path.
