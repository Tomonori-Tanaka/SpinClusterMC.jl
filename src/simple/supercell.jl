"""
    supercell.jl

General supercell construction for the Simple engine.

The shared geometry (integer linear algebra, primitive-cell recovery, cell
enumeration, cluster-offset / self-overlap helpers) lives in
[`SupercellCommon`](@ref) and is reused by `JPhiMagestyCarlo`. This file keeps
only the Simple-specific pieces: the `ClusterTemplate` type, `build_templates`
(which bundles the physics — `ls`, weights, `J`, un-folded multiplicity), and
`_generate_instances_matrix` (which emits Simple `ClusterInstance`s for a general
supercell matrix `M`).

The diagonal `repeat` API keeps using the unchanged `_generate_instances`
(legacy, bit-exact). This `supercell_matrix` path is the general one.
"""

using StaticArrays: SMatrix
using ..SupercellCommon: PrimitiveCell, _int_det3, _adjugate3, _col_hermite,
                         _wrap_offset_into_supercell,
                         _cluster_base_stabilizer, _cluster_offsets,
                         _enumerate_cells, _supercell_from_repeat
import ..SupercellCommon: extract_primitive

"""
    extract_primitive(sys::SystemData) -> PrimitiveCell

Simple-engine convenience wrapper: recover the primitive cell from a parsed
`SystemData`. Forwards to the shared `SupercellCommon.extract_primitive`.
"""
extract_primitive(sys::SystemData)::PrimitiveCell = extract_primitive(
    sys.lattice, sys.pos_frac, sys.map_sym, sys.n_trans)

# =============================================================================
# Cluster templating
# =============================================================================

"""
    ClusterTemplate

A tile-invariant description of one SALC cluster term, expressed relative to the
primitive cell. Tiling onto a supercell instantiates this template once per
supercell cell (`_generate_instances_matrix`).

Each site `k` is given by its primitive sublattice `site_subl[k]` and an integer
primitive-cell offset `site_delta[k]` measured relative to the pivot (site 1),
so `site_delta[1] == (0, 0, 0)` and `pivot_subl == site_subl[1]`.

# Fields

- `pivot_subl`: sublattice of the pivot site (site 1).
- `site_subl`: sublattice index per site, length `N = length(ls)`.
- `site_delta`: pivot-relative primitive-cell offset per site (`site_delta[1]`
  is `(0, 0, 0)`).
- `ls`, `Lf`, `Lseq`, `weights`, `J`, `multiplicity`: identical meaning to the
  corresponding `ClusterInstance` / `SALCBasisData` fields. `multiplicity` is
  the **un-folded** effective multiplicity (`basis.multiplicity ÷ s_base`).
"""
struct ClusterTemplate
    pivot_subl::Int
    site_subl::Vector{Int}
    site_delta::Vector{NTuple{3, Int}}
    ls::Vector{Int}
    Lf::Int
    Lseq::Vector{Int}
    weights::Vector{Float64}
    J::Float64
    multiplicity::Int
end

"""
    build_templates(salcs, jphi, sys, prim; jphi_threshold = 0.0)
        -> Vector{ClusterTemplate}

Convert the SALC bases of a `jphi.xml` into primitive-cell `ClusterTemplate`s,
one template per `(SALC, basis)`.

Each site's pivot-relative `(sublattice, offset)` comes from
`SupercellCommon._cluster_offsets`. No base-cell translation loop is needed: the
supercell tiling (`_generate_instances_matrix`) regenerates the copies that the
diagonal `_generate_instances` produces by enumerating translations.

**Self-overlap correction.** The XML `multiplicity` folds in the cluster's
self-overlap under the base cell: a half-period ("face") pair whose `±Δ` images
coincide in the base cell carries `multiplicity ≥ 2`. When such a cluster is
tiled onto a larger supercell its `±Δ` images become *distinct* atoms, so the
template stores an **un-folded** multiplicity

    effective_mult = basis.multiplicity ÷ s_base,

where `s_base = _cluster_base_stabilizer(...)`. This un-folds the cluster to its
distinct ±Δ neighbors (clusters are geometric — defined by their relative
vector, not by an atom-index pair). For a ferromagnet / ground state the
per-atom energy equals the base-cell model; for n > 1 non-collinear configs it
differs from the folded diagonal `repeat` path and is the geometrically faithful
one (see `docs/specs/260620-general-supercell`).

`jphi_threshold` skips SALCs with `|J_s| < jphi_threshold` exactly as
`_generate_instances` does (`0.0` keeps every SALC).
"""
function build_templates(
        salcs::AbstractVector{SALCData},
        jphi::AbstractVector{Float64},
        sys::SystemData,
        prim::PrimitiveCell;
        jphi_threshold::Float64 = 0.0
)::Vector{ClusterTemplate}
    map_sym = sys.map_sym
    n_trans = sys.n_trans
    templates = ClusterTemplate[]
    for (s, salc) in enumerate(salcs)
        J = jphi[s]
        # `abs(J) < 0.0` is always false, so threshold=0.0 keeps every SALC.
        abs(J) < jphi_threshold && continue
        for basis in salc.bases
            atoms = basis.atoms
            subl1, site_subl, site_delta = _cluster_offsets(atoms, prim)
            s_base = _cluster_base_stabilizer(atoms, map_sym, n_trans)
            mod(basis.multiplicity, s_base) == 0 || throw(ErrorException(
                "multiplicity $(basis.multiplicity) not divisible by base " *
                "stabilizer $s_base for cluster $(collect(atoms)); cannot " *
                "un-fold self-overlap for general supercell tiling"
            ))
            eff_mult = basis.multiplicity ÷ s_base
            push!(
                templates,
                ClusterTemplate(
                    subl1,
                    site_subl,
                    site_delta,
                    collect(Int, basis.ls),
                    salc.Lf,
                    collect(Int, basis.Lseq),
                    collect(Float64, basis.weights),
                    J,
                    eff_mult
                )
            )
        end
    end
    return templates
end

# =============================================================================
# General supercell tiling
# =============================================================================

"""
    _generate_instances_matrix(templates, prim, M)
        -> (instances::Vector{ClusterInstance}, n_atoms::Int)

Tile the primitive-cell `ClusterTemplate`s onto the supercell defined by the
integer matrix `M` (3×3, in primitive-cell units, `det(M) != 0`).

Each template is instantiated once per supercell cell: the pivot is placed at
the cell's canonical offset, every site's offset is wrapped into the supercell
(`_wrap_offset_into_supercell`), and the partner atom index is looked up. Atoms
use a primitive cell-major numbering
`super_index(cell_id, subl) = subl + n_prim * (cell_id - 1)`, with
`n_atoms = n_prim * |det(M)|`.

Within a template, placements that land on the same sorted atom set (a cluster
that folds onto itself in `M`) are merged into one instance whose multiplicity
is the sum of the coincident placements' `effective_mult`. Combined with the
un-folded `effective_mult` carried by the template (`build_templates`), this
gives the geometrically faithful supercell (clusters placed by relative vector).
For a ferromagnet / ground state the per-atom energy equals the base-cell model;
for n > 1 non-collinear configs it differs from the folded diagonal `repeat`
path (and is the geometrically correct one).
"""
function _generate_instances_matrix(
        templates::AbstractVector{ClusterTemplate},
        prim::PrimitiveCell,
        M::SMatrix{3, 3, Int}
)
    detM = _int_det3(M)
    detM != 0 || throw(ArgumentError("supercell matrix is singular (det = 0)"))
    adjM = _adjugate3(M)
    ncells = abs(detM)
    n_prim = prim.n_prim
    cell_index, cells_by_id = _enumerate_cells(M, adjM, detM)
    n_atoms = n_prim * ncells

    instances = ClusterInstance[]
    for tpl in templates
        N = length(tpl.ls)
        # Merge placements with the same sorted atom set, accumulating the
        # multiplicity (an `M`-level self-fold adds `effective_mult` per
        # coincidence). `order` preserves first-seen ordering for determinism.
        folded = Dict{Vector{Int}, Tuple{Vector{Int}, Int}}()
        order = Vector{Int}[]
        for cid in 1:ncells
            c0 = cells_by_id[cid]
            atoms = Vector{Int}(undef, N)
            for k in 1:N
                d = tpl.site_delta[k]
                ab = (c0[1] + d[1], c0[2] + d[2], c0[3] + d[3])
                w = _wrap_offset_into_supercell(ab, M, adjM, detM)
                cid_k = cell_index[w]
                atoms[k] = tpl.site_subl[k] + n_prim * (cid_k - 1)
            end
            key = sort(atoms)
            if haskey(folded, key)
                prev_atoms, prev_mult = folded[key]
                folded[key] = (prev_atoms, prev_mult + tpl.multiplicity)
            else
                folded[key] = (atoms, tpl.multiplicity)
                push!(order, key)
            end
        end
        for key in order
            atoms, mult = folded[key]
            push!(
                instances,
                ClusterInstance(
                    atoms,
                    tpl.ls,
                    tpl.Lf,
                    tpl.Lseq,
                    tpl.weights,
                    tpl.J,
                    mult
                )
            )
        end
    end
    return instances, n_atoms
end
