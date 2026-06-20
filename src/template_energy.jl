# Template-based energy kernel: stores only base-cell ClusterInstances and
# reconstructs supercell atom indices on-the-fly during sweep!, avoiding the
# O(repeat_volume) memory blow-up of the full _build_cluster_instances path.

struct BaseClusterInstance
    base_atoms::Vector{Int}             # base-cell atom index for each factor
    tile_deltas::Vector{NTuple{3,Int}}  # relative tile offset of each factor from pivot (factor 1)
    ls::Vector{Int}                     # angular-momentum indices per factor
    cbc_coefficient::Vector{Float64}    # cbc.coefficient copy (Mf dimension; length = Mf_size)
    prefactor::Float64
    dims::Vector{Int}                   # 2*ls[k]+1
    strides::Vector{Int}
    coeff_flat::Vector{Float64}
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
end

struct BaseClusterInstance3
    base_atoms::SVector{3,Int}
    tile_deltas::SVector{3,NTuple{3,Int}}
    ls::SVector{3,Int}
    cbc_coefficient::Vector{Float64}
    prefactor::Float64
    dims::SVector{3,Int}
    strides::SVector{4,Int}                 # N+1 = 4
    coeff_flat::Vector{Float64}
end

struct LocalEnergyTemplate
    base_instances::Vector{BaseClusterInstance}                    # N ≥ 4
    base_instances2::Vector{BaseClusterInstance2}                  # N = 2
    base_instances3::Vector{BaseClusterInstance3}                  # N = 3
    related_by_base_atom::Vector{Vector{RelatedBaseCluster}}       # → base_instances
    related2_by_base_atom::Vector{Vector{RelatedBaseCluster}}      # → base_instances2
    related3_by_base_atom::Vector{Vector{RelatedBaseCluster}}      # → base_instances3
    # Precomputed supercell-atom indices for the N=2 and N=3 hot paths. Built once
    # at template construction; replaces the per-sweep `_tile_coords` + `mod`-heavy
    # `supercell_atom_index` calls in `_template_local_energy!`. For each supercell
    # atom `i` and each `rc` in `related{2,3}_by_base_atom[base_of(i)]`, the table
    # stores the N SAI values for the cluster's sites at positions
    #   sai{2,3}_flat[sai{2,3}_offsets[i] + N*(rc_local_idx-1) + (k-1)]
    # for `k = 1..N`. `sai{2,3}_offsets[i+1] - sai{2,3}_offsets[i]` is N * len(related[b]).
    # N ≥ 4 path keeps the on-the-fly SAI calls; both test problems (bcc_2x2x2,
    # ferh_4x4x4) have zero N ≥ 4 clusters.
    sai2_flat::Vector{Int}
    sai2_offsets::Vector{Int}
    sai3_flat::Vector{Int}
    sai3_offsets::Vector{Int}
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
    base_instances3 = BaseClusterInstance3[]
    related_by_base_atom = [RelatedBaseCluster[] for _ in 1:h.base_n_atoms]
    related2_by_base_atom = [RelatedBaseCluster[] for _ in 1:h.base_n_atoms]
    related3_by_base_atom = [RelatedBaseCluster[] for _ in 1:h.base_n_atoms]

    coeff_flat_cache = Dict{UInt, Vector{Float64}}()

    for (s, group) in enumerate(h.salc_list)
        js = h.jphi[s]
        for cbc in group
            scaling = _cluster_scaling(length(cbc.atoms))
            N_cbc = length(cbc.atoms)
            inst_dims = [2 * l + 1 for l in cbc.ls]
            inst_strides = _compute_instance_strides(cbc.ls)
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
                    )
                    push!(base_instances2, inst2)
                    inst_idx = length(base_instances2)
                    for k in 1:2
                        b = base_atoms[k]
                        push!(related2_by_base_atom[b], RelatedBaseCluster(inst_idx, k))
                    end
                elseif N_cbc == 3
                    inst3 = BaseClusterInstance3(
                        SVector{3,Int}(base_atoms[1], base_atoms[2], base_atoms[3]),
                        SVector{3,NTuple{3,Int}}(tile_deltas[1], tile_deltas[2], tile_deltas[3]),
                        SVector{3,Int}(cbc.ls[1], cbc.ls[2], cbc.ls[3]),
                        collect(Float64, cbc.coefficient),
                        prefactor,
                        SVector{3,Int}(inst_dims[1], inst_dims[2], inst_dims[3]),
                        SVector{4,Int}(inst_strides[1], inst_strides[2], inst_strides[3], inst_strides[4]),
                        inst_coeff_flat,
                    )
                    push!(base_instances3, inst3)
                    inst_idx = length(base_instances3)
                    for k in 1:3
                        b = base_atoms[k]
                        push!(related3_by_base_atom[b], RelatedBaseCluster(inst_idx, k))
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

    sai2_flat, sai2_offsets = _build_sai_table_n(
        related2_by_base_atom, base_instances2, h, 2,
    )
    sai3_flat, sai3_offsets = _build_sai_table_n(
        related3_by_base_atom, base_instances3, h, 3,
    )

    return LocalEnergyTemplate(
        base_instances, base_instances2, base_instances3,
        related_by_base_atom, related2_by_base_atom, related3_by_base_atom,
        sai2_flat, sai2_offsets, sai3_flat, sai3_offsets,
    )
end

# Precompute SAIs for one fixed cluster size N (2 or 3). For each supercell atom `i`,
# for each `rc` in `related_by_base_atom[base_of(i)]`, store the N supercell-atom
# indices of the cluster sites packed in `flat`.
#
# Indexing (all 1-based, Julia convention):
#   slice for atom `i` is `flat[offsets[i] : offsets[i+1] - 1]` (length = N * len(related[b]))
#   within that slice, `rc_idx`'s SAIs are at positions `N*(rc_idx-1) + 1 .. N*rc_idx`
#   the readers in `_template_local_energy!` use `base_off = offsets[i] - 1` and
#   `flat[base_off + N*(rc_idx-1) + k]`, which is equivalent.
# `base_instances_n` provides `base_atoms` and `tile_deltas`.
function _build_sai_table_n(
    related_by_base_atom::Vector{Vector{RelatedBaseCluster}},
    base_instances_n,
    h::SCEHamiltonian,
    N::Int,
)::Tuple{Vector{Int}, Vector{Int}}
    n_atoms = h.n_atoms
    base_n = h.base_n_atoms
    rep = h.repeat
    n1, n2, n3 = rep
    offsets = Vector{Int}(undef, n_atoms + 1)
    offsets[1] = 1
    @inbounds for i in 1:n_atoms
        b = ((i - 1) % base_n) + 1
        offsets[i + 1] = offsets[i] + N * length(related_by_base_atom[b])
    end
    flat = Vector{Int}(undef, offsets[n_atoms + 1] - 1)
    @inbounds for i in 1:n_atoms
        b = ((i - 1) % base_n) + 1
        ti, tj, tk = _tile_coords(i, base_n, rep)
        base_off = offsets[i] - 1
        related = related_by_base_atom[b]
        for rc_idx in 1:length(related)
            rc = related[rc_idx]
            inst = base_instances_n[rc.inst_idx]
            pvd = inst.tile_deltas[rc.pivot_k]
            pv1, pv2, pv3 = pvd[1], pvd[2], pvd[3]
            for k in 1:N
                da = inst.tile_deltas[k]
                flat[base_off + N * (rc_idx - 1) + k] = supercell_atom_index(
                    inst.base_atoms[k],
                    mod(ti + da[1] - pv1, n1),
                    mod(tj + da[2] - pv2, n2),
                    mod(tk + da[3] - pv3, n3),
                    base_n, rep,
                )
            end
        end
    end
    return flat, offsets
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

"""
Delta-energy tensor contraction for a `BaseClusterInstance` with the changed atom
identified by `changed_atom`. Uses preallocated `other_sites_buf` / `cart_idx_buf`.
Invariant: `changed_atom` is one of `atoms[1:N]`, guaranteed by `related_by_base_atom`.
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

    Mf_size = length(inst.cbc_coefficient)
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
# Invariant: changed_atom is one of {a1, a2}, guaranteed by `related2_by_base_atom`.

@inline function _tensor_contract_template2_changed!(
    inst::BaseClusterInstance2,
    a1::Int, a2::Int,
    zlm_cache::Matrix{Float64},
    changed_atom::Int,
)::Float64
    if a1 == changed_atom
        other_atom = a2
        l_chg = inst.ls[1]; l_oth = inst.ls[2]
        d_chg = inst.dims[1]; d_oth = inst.dims[2]
        s_chg = inst.strides[1]; s_oth = inst.strides[2]
    else  # a2 == changed_atom by invariant
        other_atom = a1
        l_chg = inst.ls[2]; l_oth = inst.ls[1]
        d_chg = inst.dims[2]; d_oth = inst.dims[1]
        s_chg = inst.strides[2]; s_oth = inst.strides[1]
    end
    chg_col_base = l_chg * l_chg
    oth_col_base = l_oth * l_oth
    total_spatial = inst.strides[3]
    Mf_size = length(inst.cbc_coefficient)
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

# N=3-specialized contraction kernels.
# Invariant: changed_atom is one of {a1, a2, a3}, guaranteed by `related3_by_base_atom`.

# Inner kernel for "atom at site `chg_pos` is changed". The other two sites
# are passed as (l, d, s, atom) triples. SIMD over the changed-site index
# (innermost), product over the other two.
@inline function _kernel3_chg(
    inst::BaseClusterInstance3,
    l_chg::Int, d_chg::Int, s_chg::Int, changed_atom::Int,
    l_o1::Int, d_o1::Int, s_o1::Int, a_o1::Int,
    l_o2::Int, d_o2::Int, s_o2::Int, a_o2::Int,
    zlm_cache::Matrix{Float64},
)::Float64
    chg_col_base = l_chg * l_chg
    o1_col_base = l_o1 * l_o1
    o2_col_base = l_o2 * l_o2
    total_spatial = inst.strides[4]
    Mf_size = length(inst.cbc_coefficient)
    coeff_flat = inst.coeff_flat
    result = 0.0
    @inbounds for mf_idx in 1:Mf_size
        base_mf = 1 + (mf_idx - 1) * total_spatial
        mf_contribution = 0.0
        for m_o2 in 1:d_o2
            z_o2 = zlm_cache[a_o2, o2_col_base + m_o2]
            base_m2 = base_mf + (m_o2 - 1) * s_o2
            for m_o1 in 1:d_o1
                z_o1 = zlm_cache[a_o1, o1_col_base + m_o1]
                base_m1 = base_m2 + (m_o1 - 1) * s_o1
                inner = 0.0
                @simd for m_chg in 1:d_chg
                    inner +=
                        coeff_flat[base_m1 + (m_chg - 1) * s_chg] *
                        zlm_cache[changed_atom, chg_col_base + m_chg]
                end
                mf_contribution += z_o1 * z_o2 * inner
            end
        end
        result += inst.cbc_coefficient[mf_idx] * mf_contribution
    end
    return result
end

@inline function _tensor_contract_template3_changed!(
    inst::BaseClusterInstance3,
    a1::Int, a2::Int, a3::Int,
    zlm_cache::Matrix{Float64},
    changed_atom::Int,
)::Float64
    if a1 == changed_atom
        return _kernel3_chg(inst,
            inst.ls[1], inst.dims[1], inst.strides[1], changed_atom,
            inst.ls[2], inst.dims[2], inst.strides[2], a2,
            inst.ls[3], inst.dims[3], inst.strides[3], a3,
            zlm_cache)
    elseif a2 == changed_atom
        return _kernel3_chg(inst,
            inst.ls[2], inst.dims[2], inst.strides[2], changed_atom,
            inst.ls[1], inst.dims[1], inst.strides[1], a1,
            inst.ls[3], inst.dims[3], inst.strides[3], a3,
            zlm_cache)
    else  # a3 == changed_atom by invariant
        return _kernel3_chg(inst,
            inst.ls[3], inst.dims[3], inst.strides[3], changed_atom,
            inst.ls[1], inst.dims[1], inst.strides[1], a1,
            inst.ls[2], inst.dims[2], inst.strides[2], a2,
            zlm_cache)
    end
end

# _template_local_energy! is defined in JPhiMagestyCarlo.jl after JPhiSpinMC,
# so that mc::JPhiSpinMC can be used as the type annotation to avoid boxing of
# Union{Nothing,LocalEnergyTemplate} fields in the hot sweep! path.

# =============================================================================
# Phase 2: primitive cell-major template construction (general supercell M)
#
# The folded `BaseClusterInstance{,2,3}` above store base-cell atom indices plus
# tile offsets and reconstruct supercell indices with `supercell_atom_index`
# (tile-major numbering). The Phase-2 un-fold path instead describes each cluster
# in *primitive-cell* coordinates (sublattice + pivot-relative cell offset) and
# places it onto an arbitrary integer supercell matrix M via `SupercellCommon`,
# matching the `:tensor` un-fold reference (`_build_cluster_instances_matrix`).
#
# This block (P2-M1) builds the templates and a cell-major SAI table but is NOT
# yet wired into `_template_local_energy!` (sweep still uses the folded path).
# Its geometry is validated against `_build_cluster_instances_matrix` by a unit
# test. Kernel wiring (N=2/3 fast paths) follows in P2-M2.
# =============================================================================

# One primitive-cell-based cluster template (one per (salc, cbc), un-fold path).
# `site_delta[k]` is the pivot-relative (site 1) integer primitive-cell offset of
# site k (`site_delta[1] == (0,0,0)`); `site_subl[k]` its primitive sublattice.
# `prefactor = js * eff_mult * scaling` with `eff_mult = multiplicity ÷ s_base`,
# un-folding the XML self-overlap (matches `_build_cluster_instances_matrix`).
struct PrimClusterTemplate
    cbc_id::UInt                       # objectid(cbc), for grouping / validation
    pivot_subl::Int                    # site_subl[1]
    site_subl::Vector{Int}             # primitive sublattice of each site
    site_delta::Vector{NTuple{3,Int}}  # pivot-relative primitive-cell offsets
    ls::Vector{Int}
    prefactor::Float64
    dims::Vector{Int}                  # 2*ls[k]+1
    strides::Vector{Int}               # length N+1
    coeff_flat::Vector{Float64}
    cbc_coefficient::Vector{Float64}   # Mf-dimension coefficients (length Mf_size)
end

"""
    _build_prim_cluster_templates(h::SCEHamiltonian)
        -> (templates::Vector{PrimClusterTemplate},
            related_by_subl::Vector{Vector{RelatedBaseCluster}})

Build one `PrimClusterTemplate` per coupled basis of the general supercell-matrix
Hamiltonian `h` (requires `h.prim`/`h.supercell_matrix`). `related_by_subl[subl]`
lists `(template_idx, k)` for every site `k` of every template with
`site_subl[k] == subl`, so a supercell atom on sublattice `subl` finds all
clusters that touch it (at any participating site).

The geometry mirrors `_build_cluster_instances_matrix`: `_cluster_offsets` for
the pivot-relative sublattice offsets and `eff_mult = multiplicity ÷ s_base` for
the un-folded prefactor. Placement onto `M` is done by the SAI table builder.
"""
function _build_prim_cluster_templates(h::SCEHamiltonian)
    prim = h.prim::PrimitiveCell
    n_prim = prim.n_prim
    map_sym = h.map_sym
    n_trans = h.n_trans

    templates = PrimClusterTemplate[]
    related_by_subl = [RelatedBaseCluster[] for _ in 1:n_prim]
    coeff_flat_cache = Dict{UInt, Vector{Float64}}()

    for (s, group) in enumerate(h.salc_list)
        js = h.jphi[s]
        for cbc in group
            N = length(cbc.atoms)
            scaling = _cluster_scaling(N)
            inst_dims = [2 * l + 1 for l in cbc.ls]
            inst_strides = _compute_instance_strides(cbc.ls)
            inst_coeff_flat = get!(coeff_flat_cache, objectid(cbc)) do
                vec(collect(Float64, cbc.coeff_tensor))
            end
            s_base = _cluster_base_stabilizer(cbc.atoms, map_sym, n_trans)
            mod(cbc.multiplicity, s_base) == 0 || throw(ErrorException(
                "multiplicity $(cbc.multiplicity) not divisible by base stabilizer " *
                "$s_base for cluster $(collect(cbc.atoms)); cannot un-fold " *
                "self-overlap for general supercell tiling"))
            eff_mult = cbc.multiplicity ÷ s_base
            pivot_subl, site_subl, site_delta = _cluster_offsets(cbc.atoms, prim)

            push!(templates, PrimClusterTemplate(
                objectid(cbc),
                pivot_subl,
                site_subl,
                site_delta,
                collect(Int, cbc.ls),
                js * eff_mult * scaling,
                inst_dims,
                inst_strides,
                inst_coeff_flat,
                collect(Float64, cbc.coefficient),
            ))
            t_idx = length(templates)
            for k in 1:N
                push!(related_by_subl[site_subl[k]], RelatedBaseCluster(t_idx, k))
            end
        end
    end
    return templates, related_by_subl
end

"""
    _build_sai_table_cellmajor(templates, related_by_subl, h)
        -> (sai_flat::Vector{Int}, sai_offsets::Vector{Int})

Precompute supercell-atom indices (SAIs) for every supercell atom, cell-major.
Supercell atom `i` decomposes as `(cell_id, subl) = ((i-1) ÷ n_prim + 1,
(i-1) % n_prim + 1)`. For each `rc = (template_idx, pivot_k)` in
`related_by_subl[subl]`, the participating site `pivot_k` is placed in `cell_id`
and every site `k'` is wrapped into the supercell:

    abs_off = cells_by_id[cell_id] .+ (site_delta[k'] - site_delta[pivot_k])
    sai_k'  = site_subl[k'] + n_prim * (cell_index[wrap(abs_off)] - 1)

so `sai_{pivot_k} == i`. SAIs for atom `i` are concatenated over `rc` (each `rc`
contributing `N(rc)` entries) in the `related_by_subl[subl]` order, into the
slice `sai_flat[sai_offsets[i] : sai_offsets[i+1]-1]`.
"""
function _build_sai_table_cellmajor(
    templates::Vector{PrimClusterTemplate},
    related_by_subl::Vector{Vector{RelatedBaseCluster}},
    h::SCEHamiltonian,
)::Tuple{Vector{Int}, Vector{Int}}
    prim = h.prim::PrimitiveCell
    n_prim = prim.n_prim
    M = SMatrix{3, 3, Int}(h.supercell_matrix)
    detM = _int_det3(M)
    adjM = _adjugate3(M)
    cell_index, cells_by_id = _enumerate_cells(M, adjM, detM)
    n_atoms = h.n_atoms

    offsets = Vector{Int}(undef, n_atoms + 1)
    offsets[1] = 1
    for i in 1:n_atoms
        subl = ((i - 1) % n_prim) + 1
        tot = 0
        for rc in related_by_subl[subl]
            tot += length(templates[rc.inst_idx].site_subl)
        end
        offsets[i + 1] = offsets[i] + tot
    end

    flat = Vector{Int}(undef, offsets[n_atoms + 1] - 1)
    for i in 1:n_atoms
        cell_id = ((i - 1) ÷ n_prim) + 1
        subl = ((i - 1) % n_prim) + 1
        c0 = cells_by_id[cell_id]
        pos = offsets[i] - 1
        for rc in related_by_subl[subl]
            t = templates[rc.inst_idx]
            pv = t.site_delta[rc.pivot_k]
            N = length(t.site_subl)
            for k in 1:N
                d = t.site_delta[k]
                abs_off = (c0[1] + d[1] - pv[1], c0[2] + d[2] - pv[2], c0[3] + d[3] - pv[3])
                w = _wrap_offset_into_supercell(abs_off, M, adjM, detM)
                pos += 1
                flat[pos] = t.site_subl[k] + n_prim * (cell_index[w] - 1)
            end
        end
    end
    return flat, offsets
end
