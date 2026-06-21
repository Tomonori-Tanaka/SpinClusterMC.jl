# Template-based energy kernel (un-fold, general supercell matrix M): stores one
# primitive-cell cluster template per (salc, cbc) plus a per-atom cell-major
# instance table, and reconstructs supercell atom indices during sweep! without
# enumerating the full O(|det M|) instance list (kept O(n_templates) in memory).

# `related_by_subl[subl]` lists `(template_idx, pivot_k)` for every site `pivot_k`
# of every template whose sublattice is `subl` — i.e. the clusters a supercell
# atom on that sublattice participates in (at any site).
struct RelatedBaseCluster
    inst_idx::Int
    pivot_k::Int  # participating site k of the template (its sublattice == subl)
end

# One primitive-cell-based cluster template (one per (salc, cbc), un-fold path).
# `site_delta[k]` is the pivot-relative (site 1) integer primitive-cell offset of
# site k (`site_delta[1] == (0,0,0)`); `site_subl[k]` its primitive sublattice.
# `prefactor = js * eff_mult * scaling` with `eff_mult = multiplicity ÷ s_base`,
# un-folding the XML self-overlap (matches `_build_cluster_instances`).
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

# Per-atom deduplicated un-fold instance table (cell-major). For each supercell
# atom `i`, `entry_off[i]:entry_off[i+1]-1` indexes the distinct cluster instances
# that touch `i`; entry `e` is template `entry_tmpl[e]` placed onto the supercell
# atoms `sai[sai_off[e]:sai_off[e+1]-1]` (site order matches the template, so the
# changed-atom kernel finds `i` among them). One of those atoms equals `i`.
struct UnfoldSAITable
    entry_off::Vector{Int}    # length n_atoms+1
    entry_tmpl::Vector{Int}   # length n_entries
    sai_off::Vector{Int}      # length n_entries+1
    sai::Vector{Int}          # flattened ordered atom tuples
end

struct LocalEnergyTemplate
    # Un-fold (general supercell matrix M) template. `prim_templates[t]` is one
    # primitive-cell cluster; `unfold` holds the per-atom de-duplicated instance
    # table that `_template_local_energy!` walks. `repeat` is sugar for
    # `M = reshape_base * diag(repeat)`, so this single representation serves both.
    prim_templates::Vector{PrimClusterTemplate}
    unfold::UnfoldSAITable
end

"""
Build a `LocalEnergyTemplate` for the energy-template (`:tensor_template`) kernel
on the un-fold (general supercell matrix M) path: primitive-cell templates
(`_build_prim_cluster_templates`) plus a per-atom cell-major de-duplicated
instance table (`_build_sai_table_cellmajor`). `repeat` is sugar for
`M = reshape_base * diag(repeat)`, so this single builder serves every case
(`h.supercell_matrix` is always set; see `load_sce_hamiltonian`).
"""
function build_local_energy_template(h::SCEHamiltonian)::LocalEnergyTemplate
    templates, related_by_subl = _build_prim_cluster_templates(h)
    unfold = _build_sai_table_cellmajor(templates, related_by_subl, h)
    return LocalEnergyTemplate(templates, unfold)
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

The geometry mirrors `_build_cluster_instances`: `_cluster_offsets` for
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
    _build_sai_table_cellmajor(templates, related_by_subl, h) -> UnfoldSAITable

Precompute, for every supercell atom, the de-duplicated list of un-fold cluster
instances that touch it (cell-major). Supercell atom `i` decomposes as
`(cell_id, subl) = ((i-1) ÷ n_prim + 1, (i-1) % n_prim + 1)`. For each
`rc = (template_idx, pivot_k)` in `related_by_subl[subl]`, the participating site
`pivot_k` is placed in `cell_id` and every site `k'` is wrapped into the
supercell:

    abs_off = cells_by_id[cell_id] .+ (site_delta[k'] - site_delta[pivot_k])
    sai_k'  = site_subl[k'] + n_prim * (cell_index[wrap(abs_off)] - 1)

so `sai_{pivot_k} == i`. The reconstructed ordered tuples are **de-duplicated per
atom**: a placement whose atom `i` occupies several sites (a cluster that
self-overlaps onto `i` in a small supercell) is reconstructed once per occupied
site but is the *same* instance (identical ordered tuple), so it must contribute
once. De-duplicating by the *ordered* tuple collapses these self-overlap repeats
while keeping genuinely distinct cell placements that merely share a *sorted*
atom set (fold-accumulation across cells) as separate entries — exactly matching
the deduped/accumulated instance list of `_build_cluster_instances`, so
the per-atom local energy agrees with the `:tensor` un-fold reference.
"""
function _build_sai_table_cellmajor(
    templates::Vector{PrimClusterTemplate},
    related_by_subl::Vector{Vector{RelatedBaseCluster}},
    h::SCEHamiltonian,
)::UnfoldSAITable
    prim = h.prim::PrimitiveCell
    n_prim = prim.n_prim
    M = SMatrix{3, 3, Int}(h.supercell_matrix)
    detM = _int_det3(M)
    adjM = _adjugate3(M)
    cell_index, cells_by_id = _enumerate_cells(M, adjM, detM)
    n_atoms = h.n_atoms

    entry_off = Vector{Int}(undef, n_atoms + 1)
    entry_off[1] = 1
    entry_tmpl = Int[]
    sai_off = Int[1]
    sai = Int[]

    seen = Set{Vector{Int}}()
    for i in 1:n_atoms
        cell_id = ((i - 1) ÷ n_prim) + 1
        subl = ((i - 1) % n_prim) + 1
        c0 = cells_by_id[cell_id]
        empty!(seen)
        for rc in related_by_subl[subl]
            t = templates[rc.inst_idx]
            pv = t.site_delta[rc.pivot_k]
            N = length(t.site_subl)
            atoms = Vector{Int}(undef, N)
            for k in 1:N
                d = t.site_delta[k]
                abs_off = (c0[1] + d[1] - pv[1], c0[2] + d[2] - pv[2], c0[3] + d[3] - pv[3])
                w = _wrap_offset_into_supercell(abs_off, M, adjM, detM)
                atoms[k] = t.site_subl[k] + n_prim * (cell_index[w] - 1)
            end
            # Dedup by ordered tuple keyed together with the template (different
            # templates can coincide on atoms but are distinct instances).
            key = vcat(rc.inst_idx, atoms)
            key in seen && continue
            push!(seen, key)
            push!(entry_tmpl, rc.inst_idx)
            append!(sai, atoms)
            push!(sai_off, length(sai) + 1)
        end
        entry_off[i + 1] = length(entry_tmpl) + 1
    end
    return UnfoldSAITable(entry_off, entry_tmpl, sai_off, sai)
end

# Full tensor contraction of one un-fold cluster instance (`PrimClusterTemplate`
# `t` placed onto supercell atoms `atoms`), organized with `changed` (the swept
# atom) as the inner SIMD index. Identical math to
# `_tensor_contract_template_changed!` but reads the per-site data from `t` and
# uses the explicit `atoms` tuple. If `changed` occupies several sites of the
# instance (a small-cell self-overlap), the first occurrence is treated as the
# changed site and the others read the same `zlm_cache[changed]` row, so the full
# multilinear contraction is still evaluated correctly. Returns the contraction
# value (caller multiplies by `t.prefactor`).
@inline function _tensor_contract_unfold_changed!(
    other_sites_buf::AbstractVector{Int},
    cart_idx_buf::AbstractVector{Int},
    t::PrimClusterTemplate,
    atoms::AbstractVector{Int},
    zlm_cache::Matrix{Float64},
    changed_atom::Int,
)::Float64
    N = length(t.site_subl)
    sitepos = 0
    @inbounds for k in 1:N
        if atoms[k] == changed_atom
            sitepos = k
            break
        end
    end
    # `changed_atom` is guaranteed to occupy a site of this entry by
    # `_build_sai_table_cellmajor`; guard defensively (matches the reference
    # `_tensor_contract_instance_cached_changed!`) so a broken invariant is a
    # no-op rather than an `@inbounds` out-of-bounds read.
    sitepos == 0 && return 0.0

    changed_l = t.ls[sitepos]
    n_other = 0
    @inbounds for s in 1:N
        if s != sitepos
            n_other += 1
            other_sites_buf[n_other] = s
        end
    end

    strides = t.strides
    stride_changed = strides[sitepos]
    dims_sitepos = 2 * changed_l + 1
    changed_col_base = changed_l * changed_l

    Mf_size = length(t.cbc_coefficient)
    coeff_flat = t.coeff_flat
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
            tensor_result += t.cbc_coefficient[mf_idx] * mf_contribution
        end
        return tensor_result
    end

    total_combos = 1
    @inbounds for j in 1:n_other
        total_combos *= t.dims[other_sites_buf[j]]
    end

    @inbounds for mf_idx in 1:Mf_size
        mf_contribution = 0.0
        base_mf = 1 + (mf_idx - 1) * strides[N + 1]
        for combo_id in 0:(total_combos - 1)
            tmp = combo_id
            @inbounds for pos in 1:n_other
                d = t.dims[other_sites_buf[pos]]
                r = tmp % d
                tmp = tmp ÷ d
                cart_idx_buf[pos] = r + 1
            end
            product_other = 1.0
            base_without_changed = base_mf
            @inbounds for pos in 1:n_other
                site = other_sites_buf[pos]
                m_idx = cart_idx_buf[pos]
                l = t.ls[site]
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
        tensor_result += t.cbc_coefficient[mf_idx] * mf_contribution
    end

    return tensor_result
end
