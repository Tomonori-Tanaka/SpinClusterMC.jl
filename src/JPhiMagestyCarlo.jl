"""
	JPhiMagestyCarlo

Reads Magesty `jphi.xml` (`SCEBasis`, `AngularMomentumCouplings`, `JPhi`), evaluates SCE energy
in spin directions with the same contract as `Magesty.Optimize.design_matrix_energy_element`, and
provides a thin [`Carlo`](https://github.com/lattice-quantum/Carlo.jl) `AbstractMC` adapter with
single-spin Metropolis updates. Local energy deltas reuse preallocated stride / index buffers (no
per-update `Vector` allocations in the tensor contraction). Optional task parameter `spin_theta_max`
selects a local geodesic spin proposal instead of i.i.d. uniform-on-sphere draws.

Real (tesseral) spherical harmonics use `SpheriCart.SphericalHarmonics` with its default
`:L2` (L2-orthonormal) normalization, which is bit-exact with Magesty's `Zₗₘ_unsafe` for
`l ≤ 3` — see `docs/zlm_convention_vs_sphericart.md`.
"""
module JPhiMagestyCarlo

using Carlo
using HDF5
using MPI
using EzXML
using LinearAlgebra
using Random
using StaticArrays
import Serialization

using Magesty.CoupledBases: CoupledBasis_with_coefficient
using Magesty.XMLIO: read_salcbasis_from_xml
using SpheriCart: SphericalHarmonics, compute, compute!
using ..SupercellCommon: PrimitiveCell, extract_primitive, _int_det3, _adjugate3,
                         _wrap_offset_into_supercell, _enumerate_cells,
                         _cluster_base_stabilizer, _cluster_offsets,
                         _supercell_from_repeat

export SCEHamiltonian,
    load_sce_hamiltonian,
    sce_energy,
    JPhiSpinMC

include("xml_io.jl")

"""
	SCEHamiltonian

Reconstructed from `jphi.xml`: SALC list, **base cell** translation map `map_sym` of size
`base_n_atoms × n_trans`, and `jphi` coefficients. The XML's `ReferenceEnergy` (`j0`)
constant is intentionally not stored: this package is for MC sampling where only ΔE
matters, so the constant offset is irrelevant.

If `repeat != (1,1,1)`, the XML cell is tiled `(n₁,n₂,n₃)` times in fractional stacking; then
`n_atoms = base_n_atoms * n₁ * n₂ * n₃`. Each cluster term is replicated on every tile, and inside
each tile the same translations as in the original `map_sym` apply (natural extension of the Magesty
definition).

Supercell lattice columns are `[n₁a₁ n₂a₂ n₃a₃]` for primitive columns `a1,a2,a3`.
"""
struct SCEHamiltonian
    n_atoms::Int
    base_n_atoms::Int
    repeat::NTuple{3, Int}
    lattice::Matrix{Float64}
    pos_frac::Matrix{Float64}
    salc_list::Vector{Vector{CoupledBasis_with_coefficient}}
    jphi::Vector{Float64}
    map_sym::Matrix{Int}
    n_trans::Int
    # Un-fold supercell (both `:tensor` and `:tensor_template` kernels). After
    # Phase 2 these are ALWAYS set: `load_sce_hamiltonian` converts `repeat` to
    # `M = reshape_base * diag(repeat)`, so every Hamiltonian carries its matrix
    # `M` and recovered primitive cell `prim`. `repeat` records the user-facing
    # tile factors (the `(0, 0, 0)` sentinel marks a directly-specified matrix).
    # (The `Union{Nothing, …}` types are retained for serialization layout; the
    # `nothing` case no longer occurs on the live construction path.)
    supercell_matrix::Union{Nothing, Matrix{Int}}
    prim::Union{Nothing, PrimitiveCell}
end

"""
Concrete translated cluster term used in MC energy evaluation.

# Fields
- `atoms`: Supercell atom indices for this instance (site order matches `cbc`).
- `cbc`: Basis/tensor data for the coupled cluster.
- `prefactor`: Pre-multiplied scalar factor (`jphi * multiplicity * scaling`).
- `dims`: Per-site tensor dimensions (`dims[k] = 2*cbc.ls[k] + 1`).
- `strides`: Flattened tensor strides (length `N+1`, `N = length(atoms)`).
- `coeff_flat`: Column-major flattened copy of `cbc.coeff_tensor` as `Vector{Float64}`.
  Avoids type instability: `cbc.coeff_tensor::AbstractArray` forces dynamic dispatch on
  every indexing call in the hot sweep loop.
- `Mf_size`: Last dimension of `coeff_tensor` (number of Mf components).
"""
struct ClusterInstance
    atoms::Vector{Int}
    cbc::CoupledBasis_with_coefficient
    prefactor::Float64
    # Precomputed from cbc.ls to avoid per-call allocations in the hot sweep path.
    dims::Vector{Int}    # dims[k] = 2*cbc.ls[k]+1
    strides::Vector{Int} # tensor strides, length N+1; strides[k] = prod(dims[1:k-1])
    # Concrete-typed copy of cbc.coeff_tensor (which is AbstractArray, causing boxing).
    coeff_flat::Vector{Float64}
    Mf_size::Int
end

"""
Precompute tensor strides for flattened coefficient-tensor indexing.

# Example
For `ls = [1, 2]`, tensor dimensions are `[3, 5]` and the returned
strides are `[1, 3, 15]` (`N+1` entries).
"""
@inline function _compute_instance_strides(ls::AbstractVector{Int})::Vector{Int}
    N = length(ls)
    s = Vector{Int}(undef, N + 1)
    s[1] = 1
    @inbounds for k in 2:N
        s[k] = s[k - 1] * (2 * ls[k - 1] + 1)
    end
    s[N + 1] = s[N] * (2 * ls[N] + 1)
    return s
end

struct LocalEnergyCache
    instances::Vector{ClusterInstance}
    body_list::Vector{Int}
    by_atom_by_body::Vector{Vector{Vector{Int}}}
    partners_by_atom::Vector{Vector{Int}}
    partners_by_atom_by_body::Vector{Vector{Vector{Int}}}
end

# Process-local cache for SCEHamiltonian and LocalEnergyCache.
# Justified as a performance necessity: on a shared-memory node with 33 MPI ranks,
# simultaneous construction of LocalEnergyCache (12+ minutes, multi-GB peak alloc
# per rank) causes total peak memory to exceed node capacity.  Each MPI rank is a
# separate OS process so this cache is not cross-rank shared memory — it only avoids
# redundant work within the same process (e.g. register_evaluables calling
# load_sce_hamiltonian after JPhiSpinMC is already constructed).
# The supercell descriptor `Union{Nothing,NTuple{9,Int}}` disambiguates the
# diagonal `repeat` path (nothing) from the general `supercell_matrix` path
# (flattened 3×3) so the two never collide in the cache.
const _HAM_CACHE    = Dict{Tuple{String,NTuple{3,Int},Union{Nothing,NTuple{9,Int}},Float64}, SCEHamiltonian}()
const _ECACHE_CACHE = Dict{Tuple{String,NTuple{3,Int},Union{Nothing,NTuple{9,Int}},Float64}, LocalEnergyCache}()

# Caches the derived per-atom instance index structures, which are deterministically
# computed from (ham, cache, active_body_indices) and can be expensive (~70 MiB) to
# rebuild during Carlo PT checkpoint gather on the coordinator rank.
struct DerivedInstanceCache
    active_body_indices::Vector{Int}
    active_instance_indices::Vector{Int}
    related_instances_by_atom::Vector{Vector{Int}}
    max_l::Int
    max_sites::Int
end

const _DERIVED_CACHE = Dict{Tuple{String,NTuple{3,Int},Union{Nothing,NTuple{9,Int}},Float64,Tuple}, DerivedInstanceCache}()

# Canonical supercell descriptor for cache keys: `nothing` for the diagonal
# `repeat` path, the flattened 3×3 matrix for the `supercell_matrix` path.
_scm_key(::Nothing) = nothing
_scm_key(M::AbstractMatrix{<:Integer})::NTuple{9, Int} = Tuple(Int.(vec(M)))

"""
Return `(4π)^(n_sites/2)` normalization used for cluster contributions.
"""
@inline _cluster_scaling(n_sites::Integer)::Float64 = (4 * pi)^(n_sites / 2)

"""
Build supercell lattice and wrapped fractional positions for the general
supercell-matrix path: the primitive cell `prim` tiled by the integer matrix
`M` (primitive-cell units). Atom numbering is primitive cell-major
(`subl + n_prim*(cell_id-1)`), matching `_build_cluster_instances` via the
shared `_enumerate_cells` ordering, so geometry and instance atom indices agree.
"""
function _build_supercell_geometry_matrix(prim::PrimitiveCell, M::SMatrix{3, 3, Int})
    detM = _int_det3(M)
    adjM = _adjugate3(M)
    ncells = abs(detM)
    n_prim = prim.n_prim
    _, cells_by_id = _enumerate_cells(M, adjM, detM)
    lattice_super = prim.lattice * Matrix{Float64}(M)
    pos_super = zeros(3, n_prim * ncells)
    for cid in 1:ncells
        c = cells_by_id[cid]
        for s in 1:n_prim
            ia = s + n_prim * (cid - 1)
            g = prim.pos_frac[:, s] .+ Float64.(collect(c))   # primitive coords
            r = prim.lattice * g                              # Cartesian
            x = lattice_super \ r
            x .-= floor.(x)
            pos_super[:, ia] .= x
        end
    end
    return lattice_super, pos_super
end

"""
    load_sce_hamiltonian(xml_path; repeat=(1,1,1), supercell_matrix=nothing,
                         jphi_threshold=0.0) -> SCEHamiltonian

Supercell selection (mutually exclusive); both take the same un-fold path:
- `repeat = (n1, n2, n3)` (default): sugar for
  `supercell_matrix = reshape_base * diag(n1, n2, n3)`. Because the base (XML)
  cell is itself a supercell of the primitive cell, even `repeat = (1, 1, 1)`
  un-folds into the primitive cells with cell-major numbering; the physical
  energy is unchanged but the atom index → atom map is no longer the historical
  tile-major one.
- `supercell_matrix = M` (3×3 integer, `det(M) != 0`): arbitrary supercell of
  the primitive cell recovered from the XML translation table — non-diagonal /
  non-base-multiple cells, down to a single primitive cell.

Both modes use primitive cell-major atom numbering; clusters are placed by their
relative vector and self-overlapping ("face") pairs are un-folded into their
distinct ±Δ neighbors. `repeat` and the equivalent `supercell_matrix` produce an
identical Hamiltonian. Both the `:tensor` and `:tensor_template` kernels serve
this path (see `JPhiSpinMC`).

`jphi_threshold` (eV, non-negative): SALCs whose `abs(jphi[s]) < jphi_threshold`
are filtered out of `salc_list` / `jphi` before the Hamiltonian is built.
Default `0.0` keeps every SALC (bit-exact match to the unfiltered path).
Use `eps()` or `nextfloat(0.0)` to drop only strictly-zero coefficients.
Throws `ArgumentError` if every SALC would be dropped.
"""
function load_sce_hamiltonian(
    xml_path::AbstractString;
    repeat::NTuple{3, Int} = (1, 1, 1),
    supercell_matrix::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
    jphi_threshold::Real = 0.0,
)::SCEHamiltonian
    supercell_matrix === nothing &&
        (all(r -> r ≥ 1, repeat) ||
         throw(ArgumentError("repeat must be positive integers, got $repeat")))
    thr = Float64(jphi_threshold)
    thr ≥ 0 || throw(ArgumentError("jphi_threshold must be non-negative, got $thr"))
    basis = read_salcbasis_from_xml(xml_path)
    sys = parse_system_xml(xml_path)
    jphi = read_jphi_coefficients(xml_path)
    length(jphi) == length(basis.salc_list) ||
        throw(ArgumentError("number of jphi values ($(length(jphi))) != num_salc ($(length(basis.salc_list)))"))
    sys.n_atoms ≥ maximum(
        maximum(maximum(cbc.atoms) for cbc in grp; init = 0) for grp in basis.salc_list;
        init = 0,
    ) ||
        throw(ArgumentError("atom index in basis exceeds NumberOfAtoms"))

    salc_list = basis.salc_list
    if thr > 0
        # `keep(s) = abs(jphi[s]) ≥ thr`. Short-circuit when thr == 0 so the
        # unfiltered path stays bit-exact (no filter/log/check runs at all).
        n_total = length(jphi)
        keep_mask = abs.(jphi) .≥ thr
        n_kept = count(keep_mask)
        if n_kept == 0
            max_abs = isempty(jphi) ? 0.0 : maximum(abs, jphi)
            throw(ArgumentError(
                "jphi_threshold=$thr eV filters out all $n_total SALCs " *
                "(max |J|=$max_abs eV); Hamiltonian would be empty"))
        end
        n_dropped = n_total - n_kept
        if n_dropped > 0
            max_dropped = maximum(abs(j) for (j, k) in zip(jphi, keep_mask) if !k; init = 0.0)
            @debug "Dropped $n_dropped / $n_total SALCs below jphi_threshold=$thr eV " *
                   "(max dropped |J|=$max_dropped eV)"
            salc_list = salc_list[keep_mask]
            jphi = jphi[keep_mask]
        end
    end

    n0 = sys.n_atoms
    # Unified un-fold path: both `repeat` and `supercell_matrix` are expressed as
    # an integer supercell matrix M (primitive-cell units) and tiled by the shared
    # `SupercellCommon` geometry. `repeat = (n1, n2, n3)` is the sugar
    # `M = reshape_base * diag(n)` (Phase 2); the base cell is itself a supercell
    # of the primitive cell, so even `repeat = (1, 1, 1)` un-folds into `n_trans`
    # primitive cells with cell-major atom numbering. The physical energy is
    # unchanged (folded ≡ un-fold at the base cell); only the atom index → atom
    # map differs from the historical tile-major numbering.
    prim = extract_primitive(sys.lattice, sys.pos_frac, sys.map_sym, sys.n_trans)
    if supercell_matrix === nothing
        M = _supercell_from_repeat(prim.reshape_base, repeat)
        rep_record = repeat
    else
        repeat == (1, 1, 1) ||
            throw(ArgumentError("specify either repeat or supercell_matrix, not both"))
        size(supercell_matrix) == (3, 3) ||
            throw(ArgumentError("supercell_matrix must be 3×3, got $(size(supercell_matrix))"))
        M = SMatrix{3, 3, Int}(supercell_matrix)
        _int_det3(M) != 0 ||
            throw(ArgumentError("supercell_matrix is singular (det = 0)"))
        # `(0, 0, 0)` sentinel marks a directly-specified matrix (no repeat sugar).
        rep_record = (0, 0, 0)
    end
    lat_s, pos_s = _build_supercell_geometry_matrix(prim, M)
    n_super = prim.n_prim * abs(_int_det3(M))
    return SCEHamiltonian(
        n_super,
        n0,
        rep_record,
        lat_s,
        pos_s,
        salc_list,
        jphi,
        sys.map_sym,
        sys.n_trans,
        Matrix{Int}(M),
        prim,
    )
end

"""
    _mpi_build_ham_and_cache(xml_path, rep, thr) -> (SCEHamiltonian, LocalEnergyCache)

MPI-aware constructor: only MPI rank 0 (global) builds from XML; all other ranks
receive the result via a single `MPI_Bcast` of the serialized bytes.

On a shared-memory node with N ranks, the naive approach would run
`load_sce_hamiltonian` + `build_local_energy_cache` N times simultaneously,
causing N× the peak construction memory.  Here only rank 0 runs the expensive
path; the others wait and deserialize from the broadcast buffer.

Result is stored in the process-local `_HAM_CACHE` / `_ECACHE_CACHE` so that
subsequent calls within the same process (e.g. `Carlo.register_evaluables`) skip
both the MPI coordination and the XML parse.
"""
function _mpi_build_ham_and_cache(
    xml_path::String,
    rep::NTuple{3, Int},
    scm::Union{Nothing, AbstractMatrix{<:Integer}},
    thr::Float64,
)::Tuple{SCEHamiltonian, LocalEnergyCache}
    key = (xml_path, rep, _scm_key(scm), thr)
    if haskey(_HAM_CACHE, key) && haskey(_ECACHE_CACHE, key)
        return _HAM_CACHE[key], _ECACHE_CACHE[key]
    end
    # Each MPI rank builds its own copy in parallel.  Broadcasting via
    # Julia Serialization is impractical here: the object graph (millions of
    # ClusterInstance nodes with shared cbc references) takes longer to
    # serialize than to rebuild from XML, and the broadcast buffer itself
    # requires a second copy on every rank.  The process-local cache below
    # only prevents redundant rebuilds *within the same process* (e.g. from
    # Carlo.register_evaluables calling load_sce_hamiltonian a second time).
    # `_HAM_CACHE` may already be populated by the :tensor_template path
    # (which doesn't build the full LocalEnergyCache), so reuse the ham
    # in that case and only build the missing cache.
    ham = get!(_HAM_CACHE, key) do
        scm === nothing ?
        load_sce_hamiltonian(xml_path; repeat = rep, jphi_threshold = thr) :
        load_sce_hamiltonian(xml_path; supercell_matrix = scm, jphi_threshold = thr)
    end
    cache = build_local_energy_cache(ham)
    _ECACHE_CACHE[key] = cache
    return ham, cache
end

"""
Return the `DerivedInstanceCache` for `(xml_path, rep, active_body_indices)`, building
and storing it on the first call and returning the cached result on subsequent calls.

This avoids rebuilding `_build_related_instances_by_atom` (O(n_instances × n_atoms),
~70 MiB for ferh_4x4x4) every time `Serialization.deserialize` reconstructs a
`JPhiSpinMC` during Carlo's parallel-tempering checkpoint gather.
"""
function _get_or_build_derived(
    xml_path::String,
    rep::NTuple{3,Int},
    scm::Union{Nothing, AbstractMatrix{<:Integer}},
    thr::Float64,
    active_body_indices::Vector{Int},
    cache::LocalEnergyCache,
    n_atoms::Int,
)::DerivedInstanceCache
    key = (xml_path, rep, _scm_key(scm), thr, Tuple(active_body_indices))
    haskey(_DERIVED_CACHE, key) && return _DERIVED_CACHE[key]
    derived = DerivedInstanceCache(
        active_body_indices,
        _active_instance_indices(cache, active_body_indices),
        _build_related_instances_by_atom(cache, active_body_indices, n_atoms),
        _max_l_in_instances(cache.instances),
        _max_sites_in_instances(cache.instances),
    )
    _DERIVED_CACHE[key] = derived
    return derived
end

# Reconstruct active_body_indices from the stored enabled_bodies reconstruction key.
# O(n_body_sizes); used in Serialization.deserialize where enabled_bodies is stored
# but active_body_indices is not (to keep the serialized payload small).
function _enabled_bodies_to_active_indices(
    enabled_bodies,
    body_list::Vector{Int},
)::Vector{Int}
    enabled_bodies === nothing && return collect(eachindex(body_list))
    req_set = Set(Int.(enabled_bodies))
    return [i for (i, b) in enumerate(body_list) if b in req_set]
end

"""
Evaluate one cluster tensor contraction for the provided translated atoms.
"""
@inline function _tensor_contract_instance(
    cbc::CoupledBasis_with_coefficient,
    translated_atoms::Vector{Int},
    spin_directions::Union{AbstractMatrix{<:Real},AbstractVector{<:SVector{3,<:Real}}},
)::Float64
    N = length(cbc.atoms)
    sh_values = Vector{Vector{Float64}}(undef, N)
    sph_local = SphericalHarmonics(maximum(cbc.ls))
    for (site_idx, atom) in enumerate(translated_atoms)
        l = cbc.ls[site_idx]
        sh_values[site_idx] = Vector{Float64}(undef, 2 * l + 1)
        u = _spin_at(spin_directions, atom)
        y = compute(sph_local, SVector{3,Float64}(u[1], u[2], u[3]))
        base = l * l
        @inbounds @simd for m_idx in 1:(2 * l + 1)
            sh_values[site_idx][m_idx] = y[base + m_idx]
        end
    end

    tensor_result = 0.0
    Mf_size = size(cbc.coeff_tensor, N + 1)
    dims = [2 * l + 1 for l in cbc.ls]
    for mf_idx in 1:Mf_size
        mf_contribution = 0.0
        for site_idx_tuple in CartesianIndices(Tuple(dims))
            product = 1.0
            for (site_idx, m_idx) in enumerate(site_idx_tuple.I)
                product *= sh_values[site_idx][m_idx]
            end
            tensor_idx = (site_idx_tuple.I..., mf_idx)
            mf_contribution += cbc.coeff_tensor[tensor_idx...] * product
        end
        tensor_result += cbc.coefficient[mf_idx] * mf_contribution
    end
    return tensor_result
end

"""
Enumerate the un-fold cluster instances of `h` and precompute their metadata.

Tiles each coupled basis onto the supercell matrix `M` (via `SupercellCommon`;
`repeat` is sugar for `M = reshape_base * diag(repeat)`): the cluster's
pivot-relative offsets (`_cluster_offsets`) are placed at every supercell cell and
wrapped (`_wrap_offset_into_supercell`), using primitive cell-major atom numbering
(`subl + n_prim*(cell_id-1)`). The XML self-overlap is un-folded
(`effective_mult = cbc.multiplicity ÷ s_base`); placements that fold onto the same
sorted atom set in `M` accumulate their multiplicity. The resulting
`prefactor = jphi * total_mult * scaling` un-folds each cluster onto its distinct
±Δ neighbors (clusters are geometric — defined by their relative vector).

The low-level contraction kernels (`_tensor_contract_instance*`) are unchanged;
only the instance list this feeds differs.
"""
function _build_cluster_instances(h::SCEHamiltonian)::Vector{ClusterInstance}
    prim = h.prim::PrimitiveCell
    M = SMatrix{3, 3, Int}(h.supercell_matrix)
    detM = _int_det3(M)
    adjM = _adjugate3(M)
    ncells = abs(detM)
    n_prim = prim.n_prim
    map_sym = h.map_sym
    n_trans = h.n_trans
    cell_index, cells_by_id = _enumerate_cells(M, adjM, detM)

    instances = ClusterInstance[]
    coeff_flat_cache = Dict{UInt, Vector{Float64}}()
    for (s, group) in enumerate(h.salc_list)
        js = h.jphi[s]
        for cbc in group
            N_cbc = length(cbc.atoms)
            scaling = _cluster_scaling(N_cbc)
            inst_dims = [2 * l + 1 for l in cbc.ls]
            inst_strides = _compute_instance_strides(cbc.ls)
            inst_Mf_size = size(cbc.coeff_tensor, N_cbc + 1)
            inst_coeff_flat = get!(coeff_flat_cache, objectid(cbc)) do
                vec(collect(Float64, cbc.coeff_tensor))
            end
            s_base = _cluster_base_stabilizer(cbc.atoms, map_sym, n_trans)
            mod(cbc.multiplicity, s_base) == 0 || throw(ErrorException(
                "multiplicity $(cbc.multiplicity) not divisible by base stabilizer " *
                "$s_base for cluster $(collect(cbc.atoms)); cannot un-fold " *
                "self-overlap for general supercell tiling"))
            eff_mult = cbc.multiplicity ÷ s_base
            _, site_subl, site_delta = _cluster_offsets(cbc.atoms, prim)
            N = length(site_subl)
            # Tile + accumulate-dedup by sorted atoms, summing eff_mult per fold.
            folded = Dict{Vector{Int}, Tuple{Vector{Int}, Int}}()
            order = Vector{Int}[]
            for cid in 1:ncells
                c0 = cells_by_id[cid]
                atoms = Vector{Int}(undef, N)
                for k in 1:N
                    d = site_delta[k]
                    ab = (c0[1] + d[1], c0[2] + d[2], c0[3] + d[3])
                    w = _wrap_offset_into_supercell(ab, M, adjM, detM)
                    atoms[k] = site_subl[k] + n_prim * (cell_index[w] - 1)
                end
                key = sort(atoms)
                if haskey(folded, key)
                    prev_atoms, prev_mult = folded[key]
                    folded[key] = (prev_atoms, prev_mult + eff_mult)
                else
                    folded[key] = (atoms, eff_mult)
                    push!(order, key)
                end
            end
            for key in order
                atoms, mult = folded[key]
                push!(
                    instances,
                    ClusterInstance(
                        atoms,
                        cbc,
                        js * mult * scaling,
                        inst_dims,
                        inst_strides,
                        inst_coeff_flat,
                        inst_Mf_size,
                    ),
                )
            end
        end
    end
    return instances
end

function build_local_energy_cache(h::SCEHamiltonian)::LocalEnergyCache
    instances = _build_cluster_instances(h)
    body_set = Set{Int}()
    for inst in instances
        push!(body_set, length(inst.atoms))
    end
    body_list = sort!(collect(body_set))
    body_to_idx = Dict(body => i for (i, body) in enumerate(body_list))

    by_atom_by_body = [[Int[] for _ in 1:h.n_atoms] for _ in body_list]
    partners_set_by_atom = [Set{Int}() for _ in 1:h.n_atoms]
    partners_set_by_atom_by_body = [[Set{Int}() for _ in 1:h.n_atoms] for _ in body_list]

    for (inst_idx, inst) in enumerate(instances)
        touched = Set(inst.atoms)
        body = length(inst.atoms)
        bidx = body_to_idx[body]
        for atom in touched
            push!(by_atom_by_body[bidx][atom], inst_idx)
            for other in touched
                other == atom && continue
                push!(partners_set_by_atom[atom], other)
                push!(partners_set_by_atom_by_body[bidx][atom], other)
            end
        end
    end

    partners_by_atom = [sort!(collect(s)) for s in partners_set_by_atom]
    partners_by_atom_by_body = [
        [sort!(collect(s)) for s in partners_set_by_atom_by_body[bidx]] for
        bidx in eachindex(body_list)
    ]
    return LocalEnergyCache(
        instances,
        body_list,
        by_atom_by_body,
        partners_by_atom,
        partners_by_atom_by_body,
    )
end

"""
Accumulate total interaction energy from prebuilt cluster instances.
"""
function _energy_from_instances(
    instances::Vector{ClusterInstance},
    spin_directions::Union{AbstractMatrix{<:Real},AbstractVector{<:SVector{3,<:Real}}},
)::Float64
    E = 0.0
    for inst in instances
        E += inst.prefactor * _tensor_contract_instance(inst.cbc, inst.atoms, spin_directions)
    end
    return E
end

"""
Accumulate total interaction energy from prebuilt cluster instances using a precomputed Zlm cache.
Avoids redundant Ylm evaluations: each atom's spherical harmonics are read once from `zlm_cache`
instead of being recomputed for every instance. Use `_build_zlm_cache` to construct the cache.
"""
function _energy_from_instances_cached(
    instances::Vector{ClusterInstance},
    zlm_cache::Matrix{Float64},
)::Float64
    E = 0.0
    @inbounds for inst in instances
        E += inst.prefactor * _tensor_contract_instance_cached(inst, zlm_cache)
    end
    return E
end

"""
    sce_energy(h, spin_directions) -> Float64

SCE interaction energy (j0 excluded) of the supercell: the sum of every un-fold
cluster instance's tensor contraction (`_build_cluster_instances` weighted by
`jphi * multiplicity * scaling`). j0 is intentionally excluded because this
package is used for MC sampling where only ΔE matters.

`spin_directions` should be `3 × h.n_atoms`: rows 1–3 are `x`, `y`, `z` of the
spin direction; columns are supercell atoms (`a` → column `a`).

This is a reference/validation entry point — it rebuilds the instance list on
each call (the MC hot path uses the prebuilt `LocalEnergyCache` /
`LocalEnergyTemplate`), so the per-call rebuild is acceptable.
"""
function sce_energy(
    h::SCEHamiltonian,
    spin_directions::Union{AbstractMatrix{<:Real},AbstractVector{<:SVector{3,<:Real}}},
)::Float64
    return _energy_from_instances(_build_cluster_instances(h), spin_directions)
end

include("template_energy.jl")

# --- Carlo.AbstractMC ---

"""
    JPhiSpinMC <: Carlo.AbstractMC

Metropolis Monte Carlo sampler for a spin Hamiltonian expressed as a
Symmetry-adapted Cluster Expansion (SCE).  Implements the `Carlo.AbstractMC`
interface and is intended to be driven by the Carlo.jl scheduler.

# Quick start

```julia
using Carlo, Carlo.JobTools
using SpinClusterMC.JPhiMagestyCarlo

tm = JobTools.TaskMaker()
tm.sweeps        = 2000
tm.thermalization = 500
tm.binsize       = 10
tm.seed          = 42
tm.xml_path      = "path/to/jphi.xml"
tm.T             = 0.5          # temperature in eV

# Optional: start from a ferromagnetic initial configuration
tm.initial_spins = let s = zeros(3, 16); s[3,:] .= 1.0; s end

JobTools.task(tm)
job = JobTools.JobInfo("output_dir", JPhiSpinMC; tasks = JobTools.make_tasks(tm),
                       checkpoint_time = "1:00", run_time = "60:00")
Carlo.start(Carlo.SingleScheduler, job)
```

# Accepted `params` keys

## Required
| Key | Type | Description |
|:----|:-----|:------------|
| `:xml_path` | `String` | Path to the Magesty XML file that defines the SCE Hamiltonian. |
| `:T` | `Real` | Temperature in eV. |
| `:thermalization` | `Int` | Number of thermalization sweeps before measurements begin (Carlo convention). |
| `:binsize` | `Int` | Measurement bin length (Carlo convention). |

## Geometry
| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `:repeat` or `:supercell` | 3-vector of `Int` | `(1,1,1)` | Diagonal tiling of the base (XML) cell. Total atom count becomes `base_n_atoms × n₁ × n₂ × n₃`. |
| `:supercell_matrix` | `Matrix{Int}` (3×3) | — | General integer supercell of the primitive cell (non-diagonal / non-base-multiple). Mutually exclusive with `:repeat`/`:supercell`. Both kernels support it via the shared un-fold geometry: `:tensor_template` (default) and `:tensor` give the same energy. Atoms use primitive cell-major numbering, so `:initial_spins` base-cell tiling is unavailable on this path (use random init). |

## Initial spin configuration
| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `:initial_spins` | `Matrix{<:Real}` of size `(3, n_atoms)` | (random) | Full supercell spin configuration, in the supercell's own primitive cell-major atom order. If provided, each column is assigned to the corresponding atom (renormalized to a unit vector); otherwise all spins are drawn uniformly at random on the unit sphere. Base-cell tiling (a `(3, base_n_atoms)` pattern) is not supported on the un-fold path — use `:random` or a full config. |

## Spin proposal
| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `:spin_theta_max` | `Float64 > 0` | `nothing` | If set, each Metropolis proposal is drawn geodesically within a cone of half-angle `θ_max` (radians) around the current spin. This typically yields higher acceptance at low temperatures. If absent, proposals are drawn uniformly on the sphere. |

## Numerical stability
| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `:renorm_every` | `Int ≥ 0` | `1000` | Renormalize all spins every this many sweeps to prevent floating-point drift. Set to `0` to disable. |

## Body-size selection
| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `:enabled_bodies` | collection of `Int` | (all) | Restrict the active cluster interactions to the listed body sizes (e.g., `[2]` for pair interactions only). Raises `ArgumentError` if a listed size is not present in the XML or if the selection is empty. |

## SALC pruning
| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `:jphi_threshold` | `Real ≥ 0` (eV) | `0.0` | Drop SALCs with `abs(J_s) < jphi_threshold`. Default `0.0` keeps every SALC (bit-exact match to the unfiltered Hamiltonian). Use `eps()` to drop only strictly-zero coefficients. Raises `ArgumentError` if all SALCs would be dropped. |

## Energy kernel
| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `:energy_kernel` | `Symbol` | `:tensor_template` | `:tensor_template` (default) stores one `PrimClusterTemplate` per (salc, cbc) plus a per-atom cell-major `UnfoldSAITable`, and reconstructs supercell atom indices on-the-fly during `sweep!` — O(n_templates) memory regardless of supercell size. A single generic contraction kernel (`_tensor_contract_unfold_changed!`) handles all body sizes. `:tensor` enumerates every cluster instance up front — uses more memory and is mainly kept as a validation reference. The two kernels agree to within floating-point summation order. See [`build_local_energy_template`](@ref). |

## Carlo scheduler
| Key | Type | Description |
|:----|:-----|:------------|
| `:seed` | `Int` | RNG seed. |

# Measured observables

Every sweep records the following observables in the Carlo accumulator:

| Name | Formula |
|:-----|:--------|
| `:Energy` | Total energy per atom (eV). |
| `:Energy2` | Squared energy per atom (eV²). |
| `:Magnetization` | Vector-magnetization magnitude `|⟨S⟩|`. |
| `:AbsMagnetization` | Same as `:Magnetization`. |
| `:Magnetization2` | `|⟨S⟩|²`. |
| `:Magnetization4` | `|⟨S⟩|⁴`. |

Derived quantities registered via `Carlo.register_evaluables`:

| Name | Formula |
|:-----|:--------|
| `:SpecificHeat` | `N (⟨E²⟩ − ⟨E⟩²) / T²` |
| `:BinderRatio` | `⟨m²⟩² / ⟨m⁴⟩` |
| `:Susceptibility` | `N ⟨m²⟩ / T` |
"""
mutable struct JPhiSpinMC{S<:SphericalHarmonics} <: AbstractMC
    T::Float64
    ham::SCEHamiltonian
    spins::Vector{SVector{3,Float64}}
    energy::Float64
    local_cache::LocalEnergyCache
    active_body_indices::Vector{Int}
    active_instance_indices::Vector{Int}
    related_instances_by_atom::Vector{Vector{Int}}
    max_l::Int
    zlm_cache::Matrix{Float64}
    # Preallocated buffer to save/restore one atom's ZLM row on Metropolis rejection,
    # avoiding recomputation of all (max_l+1)² spherical harmonics for rejected moves.
    zlm_row_buf::Vector{Float64}
    # Real spherical-harmonics evaluator from SpheriCart.jl. Bit-exact replacement for
    # Magesty's `Zₗₘ_unsafe` (see docs/zlm_convention_vs_sphericart.md). `JPhiSpinMC` is
    # parameterized on `S<:SphericalHarmonics` so the concrete `(L+1)²` size is visible
    # at every `compute(sph, u)` call site; otherwise the returned `SVector` would heap-
    # allocate every Metropolis proposal (~5 KB/sweep at 128 atoms).
    sph::S
    # Reused in Metropolis sweeps to avoid per-instance allocations in delta energy.
    # (strides and dims are now precomputed in ClusterInstance)
    contract_other_sites::Vector{Int}
    contract_cart_idx::Vector{Int}
    # `nothing`: uniform random unit spin (legacy). `θ>0`: geodesic proposal with angle
    # uniform in `[-θ, θ]` around current spin (often higher acceptance at low T).
    spin_theta_max::Union{Nothing,Float64}
    # Renormalize all spins (and rebuild zlm cache) every this many sweeps. 0 = disabled.
    renorm_every::Int
    sweep_count::Int
    # Reconstruction keys stored for lightweight MPI serialization (Carlo PT gather).
    # ham and local_cache are deterministically derived from these, so they are
    # excluded from the serialized representation to avoid OOM on the root rank.
    xml_path::String
    repeat::NTuple{3,Int}
    # `nothing` for the diagonal `repeat` path; the general supercell matrix
    # (primitive-cell units, un-fold; both kernels) otherwise.
    supercell_matrix::Union{Nothing,Matrix{Int}}
    jphi_threshold::Float64
    enabled_bodies::Union{Nothing,Vector{Int}}
    # `:tensor_template` (default) uses primitive-cell un-fold templates with
    # on-the-fly supercell index reconstruction (`_template_local_energy!` over a
    # per-atom `UnfoldSAITable`). `:tensor` uses the fully-enumerated SALC tensor-
    # contraction kernel via `_energy_from_instances_cached` /
    # `_tensor_contract_instance_cached_changed!`.
    energy_kernel::Symbol
    # Populated only when `energy_kernel === :tensor_template`.
    local_template::Union{Nothing,LocalEnergyTemplate}
    atoms_buf::Vector{Int}  # reused buffer in sweep! to avoid per-instance allocation
end

# Compute max_l, max_sites, and body_list directly from the SALC list, without
# enumerating all translated instances.  Used by :tensor_template to avoid the
# O(n_atoms × n_base_instances) full cache build.
function _salc_max_l_max_sites_bodies(
    h::SCEHamiltonian,
    enabled_bodies::Union{Nothing, Vector{Int}},
)::Tuple{Int, Int, Vector{Int}}
    body_set = Set{Int}()
    max_l = 0
    max_sites = 0
    for group in h.salc_list
        for cbc in group
            N = length(cbc.atoms)
            enabled_bodies === nothing || N in enabled_bodies || continue
            push!(body_set, N)
            max_sites < N && (max_sites = N)
            for l in cbc.ls
                max_l < l && (max_l = l)
            end
        end
    end
    return max_l, max_sites, sort!(collect(body_set))
end

function JPhiSpinMC(params::AbstractDict)
    xml = params[:xml_path]
    rep = _parse_repeat_param(params)
    thr = _parse_jphi_threshold_param(params)
    T = Float64(params[:T])
    enabled_bodies = if haskey(params, :enabled_bodies)
        Int.(collect(params[:enabled_bodies]))
    else
        nothing
    end
    energy_kernel = if haskey(params, :energy_kernel)
        sym = Symbol(params[:energy_kernel])
        sym in (:tensor, :tensor_template) ||
            throw(ArgumentError("energy_kernel must be :tensor or :tensor_template, got $sym"))
        sym
    else
        :tensor_template
    end
    scm = _parse_supercell_matrix_param(params)
    if scm !== nothing
        rep == (1, 1, 1) || throw(ArgumentError(
            "specify either repeat/supercell or supercell_matrix, not both"))
        # Phase 2: both :tensor and :tensor_template support the general
        # supercell matrix via the shared un-fold geometry (SupercellCommon).
    end
    spin_theta_max = if haskey(params, :spin_theta_max)
        θ = Float64(params[:spin_theta_max])
        θ > 0.0 || throw(ArgumentError("spin_theta_max must be positive, got $θ"))
        θ
    else
        nothing
    end
    renorm_every = if haskey(params, :renorm_every)
        k = Int(params[:renorm_every])
        k ≥ 0 || throw(ArgumentError("renorm_every must be non-negative, got $k"))
        k
    else
        1000
    end

    if energy_kernel === :tensor_template
        # Skip the O(n_atoms × n_base) full cache build. Build only the ham and
        # derive max_l / max_sites / body_list directly from the SALC list.
        ham = get!(_HAM_CACHE, (xml, rep, _scm_key(scm), thr)) do
            scm === nothing ?
                load_sce_hamiltonian(xml; repeat = rep, jphi_threshold = thr) :
                load_sce_hamiltonian(xml; supercell_matrix = scm, jphi_threshold = thr)
        end
        max_l, max_sites, body_list = _salc_max_l_max_sites_bodies(ham, enabled_bodies)
        active_body_indices = collect(eachindex(body_list))
        n = ham.n_atoms
        # Stub cache: empty instances; partners_* are unused by :tensor_template sweep!.
        stub_cache = LocalEnergyCache(
            ClusterInstance[],
            body_list,
            [[Int[] for _ in 1:n] for _ in body_list],
            [Int[] for _ in 1:n],
            [[Int[] for _ in 1:n] for _ in body_list],
        )
        zlm_cache       = _alloc_zlm_cache(n, max_l)
        zlm_row_buf     = Vector{Float64}(undef, (max_l + 1)^2)
        sph             = SphericalHarmonics(max_l)
        other_sites_buf = Vector{Int}(undef, max(max_sites, 1))
        cart_idx_buf    = Vector{Int}(undef, max(max_sites, 1))
        # Build the template eagerly so mc is usable without Carlo.init!
        # (e.g. tests that set spins/energy manually and call sweep! directly).
        local_template = build_local_energy_template(ham)
        atoms_buf = Vector{Int}(undef, max(max_sites, 1))
        return JPhiSpinMC(
            T, ham, [zero(SVector{3,Float64}) for _ in 1:n], 0.0,
            stub_cache,
            active_body_indices, Int[], [Int[] for _ in 1:n],
            max_l, zlm_cache, zlm_row_buf, sph,
            other_sites_buf, cart_idx_buf,
            spin_theta_max, renorm_every, 0,
            xml, rep, scm, thr, enabled_bodies, energy_kernel,
            local_template, atoms_buf,
        )
    end

    # :tensor path: build the full LocalEnergyCache.
    ham, cache = _mpi_build_ham_and_cache(xml, rep, scm, thr)
    active_body_indices = _parse_enabled_body_indices(params, cache.body_list)
    derived = _get_or_build_derived(xml, rep, scm, thr, active_body_indices, cache, ham.n_atoms)
    zlm_cache = _alloc_zlm_cache(ham.n_atoms, derived.max_l)
    zlm_row_buf = Vector{Float64}(undef, (derived.max_l + 1)^2)
    sph = SphericalHarmonics(derived.max_l)
    other_sites_work = Vector{Int}(undef, derived.max_sites)
    cart_idx_work = Vector{Int}(undef, derived.max_sites)
    return JPhiSpinMC(
        T,
        ham,
        [zero(SVector{3,Float64}) for _ in 1:ham.n_atoms],
        0.0,
        cache,
        derived.active_body_indices,
        derived.active_instance_indices,
        derived.related_instances_by_atom,
        derived.max_l,
        zlm_cache,
        zlm_row_buf,
        sph,
        other_sites_work,
        cart_idx_work,
        spin_theta_max,
        renorm_every,
        0,
        xml,
        rep,
        scm,
        thr,
        enabled_bodies,
        energy_kernel,
        nothing,   # local_template: unused for :tensor
        Int[],     # atoms_buf: unused for :tensor
    )
end

include("spin_utils.jl")

# Defined here (after JPhiSpinMC) so mc::JPhiSpinMC type annotation is available,
# preventing boxing of Union{Nothing,LocalEnergyTemplate} in the hot sweep! path.
@inline function _template_local_energy!(mc::JPhiSpinMC, i::Int)::Float64
    tpl = mc.local_template::LocalEnergyTemplate

    # Un-fold path (general supercell matrix M; `repeat` is sugar for it): walk the
    # per-atom cell-major de-duplicated instance table. Each entry is the full
    # contraction of a distinct cluster instance touching atom `i`. Body sizes N=2
    # and N=3 (the overwhelming majority of clusters) use the unrolled
    # `_contract_n{2,3}_unfold_changed` fast paths; N≥4 falls back to the generic
    # `_tensor_contract_unfold_changed!`. All three contract identically.
    unfold = tpl.unfold
    templates = tpl.prim_templates
    sai = unfold.sai
    e = 0.0
    @inbounds for ent in unfold.entry_off[i]:(unfold.entry_off[i + 1] - 1)
        t = templates[unfold.entry_tmpl[ent]]
        lo = unfold.sai_off[ent]
        hi = unfold.sai_off[ent + 1] - 1
        n_sites = hi - lo + 1
        atoms = view(sai, lo:hi)
        if n_sites == 2
            e += t.prefactor * _contract_n2_unfold_changed(t, atoms, mc.zlm_cache, i)
        elseif n_sites == 3
            e += t.prefactor * _contract_n3_unfold_changed(t, atoms, mc.zlm_cache, i)
        else
            e += t.prefactor * _tensor_contract_unfold_changed!(
                mc.contract_other_sites,
                mc.contract_cart_idx,
                t,
                atoms,
                mc.zlm_cache,
                i,
            )
        end
    end
    return e
end

"""
Rebuild the full per-atom `Z_lm` cache from current MC spins.
"""
function _rebuild_zlm_cache!(mc::JPhiSpinMC)
    compute!(mc.zlm_cache, mc.sph, mc.spins)
    return nothing
end

"""
Resolve active body-size indices from `params[:enabled_bodies]` selection.
"""
function _parse_enabled_body_indices(
    params::AbstractDict,
    body_list::Vector{Int},
)::Vector{Int}
    if !haskey(params, :enabled_bodies)
        return collect(eachindex(body_list))
    end
    req = Int.(collect(params[:enabled_bodies]))
    req_set = Set(req)
    active = Int[]
    for (bidx, body) in enumerate(body_list)
        body in req_set && push!(active, bidx)
    end
    missing = setdiff(req, body_list)
    isempty(missing) ||
        throw(
            ArgumentError(
                "enabled_bodies contains unknown body sizes $(sort(missing)); available=$(body_list)",
            ),
        )
    isempty(active) && throw(ArgumentError("enabled_bodies selects no active bodies"))
    return active
end

"""
Collect unique instance indices that belong to active body-size groups.
"""
function _active_instance_indices(
    cache::LocalEnergyCache,
    active_body_indices::Vector{Int},
)::Vector{Int}
    marks = falses(length(cache.instances))
    for bidx in active_body_indices
        for by_atom in cache.by_atom_by_body[bidx]
            for inst_idx in by_atom
                marks[inst_idx] = true
            end
        end
    end
    return findall(marks)
end

"""
    _build_related_instances_by_atom(cache, active_body_indices, n_atoms) -> Vector{Vector{Int}}

Build per-atom lists of active cluster instance indices that include each atom.
Used in `sweep!` to identify which instances must be recontracted when a single spin changes.

# Arguments
- `cache::LocalEnergyCache`: Prebuilt cache containing all cluster instances and their
  per-atom, per-body-size index lists (`cache.by_atom_by_body`).
- `active_body_indices::Vector{Int}`: Indices into `cache.body_list` selecting which
  cluster body sizes are enabled (e.g. only 2-body or only 2- and 3-body terms).
- `n_atoms::Int`: Total number of atoms in the supercell.

# Returns
`by_atom` where `by_atom[i]` is the sorted list of instance indices whose atom set
contains atom `i` and whose body size is active. Duplicate instance indices that appear
in multiple body-size lists are deduplicated via a stamp array (O(1) per entry).
"""
function _build_related_instances_by_atom(
    cache::LocalEnergyCache,
    active_body_indices::Vector{Int},
    n_atoms::Int,
)::Vector{Vector{Int}}
    by_atom = [Int[] for _ in 1:n_atoms]
    marks = zeros(Int, length(cache.instances))
    stamp = 0
    for atom in 1:n_atoms
        stamp += 1
        for bidx in active_body_indices
            for inst_idx in cache.by_atom_by_body[bidx][atom]
                if marks[inst_idx] != stamp
                    marks[inst_idx] = stamp
                    push!(by_atom[atom], inst_idx)
                end
            end
        end
    end
    return by_atom
end

"""
Parse `:repeat` / `:supercell` parameters, defaulting to `(1,1,1)`.
"""
function _parse_jphi_threshold_param(params::AbstractDict)::Float64
    thr = Float64(get(params, :jphi_threshold, 0.0))
    thr ≥ 0 || throw(
        ArgumentError("JPhiSpinMC: params[:jphi_threshold] must be non-negative, got $thr")
    )
    return thr
end

function _parse_repeat_param(params::AbstractDict)::NTuple{3, Int}
    if haskey(params, :repeat)
        r = params[:repeat]
        length(r) == 3 || throw(ArgumentError(":repeat must be length-3, got $r"))
        return (Int(r[1]), Int(r[2]), Int(r[3]))
    end
    if haskey(params, :supercell)
        r = params[:supercell]
        length(r) == 3 || throw(ArgumentError(":supercell must be length-3, got $r"))
        return (Int(r[1]), Int(r[2]), Int(r[3]))
    end
    return (1, 1, 1)
end

# Read params[:supercell_matrix] (3×3 integer matrix) if present, else nothing.
function _parse_supercell_matrix_param(params::AbstractDict)::Union{Nothing, Matrix{Int}}
    haskey(params, :supercell_matrix) || return nothing
    M = params[:supercell_matrix]
    M isa AbstractMatrix || throw(ArgumentError(
        ":supercell_matrix must be a 3×3 integer matrix; got $(typeof(M))"))
    size(M) == (3, 3) ||
        throw(ArgumentError(":supercell_matrix must be 3×3; got $(size(M))"))
    all(x -> isinteger(x), M) ||
        throw(ArgumentError(":supercell_matrix must have integer entries; got $M"))
    return Matrix{Int}(M)
end

"""
Evaluate one instance contraction using precomputed per-atom `Z_lm` cache.
"""
@inline function _tensor_contract_instance_cached(
    inst::ClusterInstance,
    zlm_cache::Matrix{Float64},
)::Float64
    cbc = inst.cbc
    N = length(inst.atoms)
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
                atom = inst.atoms[k]
                l = cbc.ls[k]
                product *= zlm_cache[atom, _zlm_col(l, m_idx)]
            end
            mf_contribution += coeff_flat[base_mf + combo_id] * product
        end
        tensor_result += cbc.coefficient[mf_idx] * mf_contribution
    end
    return tensor_result
end

"""
Delta energy tensor contraction for one changed site. Uses preallocated buffers
(`other_sites_buf`, `cart_idx_buf`) of length at least `N` for `N = length(inst.atoms)`.
Strides and dims are read from `inst.strides` / `inst.dims` (precomputed at build time).
"""
@inline function _tensor_contract_instance_cached_changed!(
    other_sites_buf::AbstractVector{Int},
    cart_idx_buf::AbstractVector{Int},
    inst::ClusterInstance,
    zlm_cache::Matrix{Float64},
    changed_atom::Int,
)::Float64
    cbc = inst.cbc
    N = length(inst.atoms)
    sitepos = 0
    @inbounds for k in 1:N
        if inst.atoms[k] == changed_atom
            sitepos = k
            break
        end
    end
    if sitepos == 0
        return _tensor_contract_instance_cached(inst, zlm_cache)
    end

    changed_l = cbc.ls[sitepos]
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
            tensor_result += cbc.coefficient[mf_idx] * mf_contribution
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
                l = cbc.ls[site]
                atom = inst.atoms[site]
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
        tensor_result += cbc.coefficient[mf_idx] * mf_contribution
    end

    return tensor_result
end

"""
    Carlo.init!(mc::JPhiSpinMC, ctx::MCContext, params::AbstractDict)

Initialize the spin configuration and internal caches before the first sweep.

**Spin initialization** (controlled by `params[:initial_spins]`):

- **With `:initial_spins`** — expects a full `3 × n_atoms` matrix whose columns
  are the spin vectors for every supercell atom, in the supercell's own
  (primitive cell-major) atom order. Each column is renormalized to a unit
  vector; a zero-norm column raises `ArgumentError`. Base-cell tiling (a
  `3 × base_n_atoms` pattern replicated over the cell) is no longer supported:
  the un-fold numbering is primitive cell-major, so a base-cell pattern can no
  longer be tiled meaningfully (Phase 2). Use `:random` or a full config.

- **Without `:initial_spins`** — all spins are drawn independently and uniformly
  on the unit sphere using the seeded RNG in `ctx`.

After setting the spin configuration, the spherical-harmonic cache and the
stored energy are rebuilt consistently.
"""
function Carlo.init!(mc::JPhiSpinMC, ctx::MCContext, params::AbstractDict)
    n = mc.ham.n_atoms
    if haskey(params, :initial_spins)
        s0 = params[:initial_spins]
        s0 isa AbstractMatrix && size(s0) == (3, n) || throw(ArgumentError(
            "initial_spins must be a full 3 × n_atoms ($n) matrix in primitive " *
            "cell-major order; base-cell tiling is unavailable on the un-fold " *
            "path (use :random or a full config). Got $(summary(s0))"))
        @inbounds for i in 1:n
            x, y, z = Float64(s0[1, i]), Float64(s0[2, i]), Float64(s0[3, i])
            nrm = sqrt(x * x + y * y + z * z)
            nrm > 0 || throw(ArgumentError("initial_spins column $i has zero norm"))
            mc.spins[i] = SVector(x / nrm, y / nrm, z / nrm)
        end
    else
        for i in 1:n
            sx, sy, sz = _rand_unit_spin(ctx.rng)
            mc.spins[i] = SVector(sx, sy, sz)
        end
    end
    _rebuild_zlm_cache!(mc)
    # j0 excluded: only ΔE matters for MC sampling.
    if mc.energy_kernel === :tensor_template
        # local_template was already built eagerly in the constructor; recompute the
        # initial energy here in case the test harness or restart path mutated mc.spins
        # after construction.
        mc.energy = sce_energy(mc.ham, mc.spins)
    else
        mc.energy = _energy_from_instances_cached(
            mc.local_cache.instances[mc.active_instance_indices],
            mc.zlm_cache,
        )
    end
    return nothing
end

function Carlo.sweep!(mc::JPhiSpinMC, ctx::MCContext)
    n = mc.ham.n_atoms
    use_template = mc.energy_kernel === :tensor_template
    @inbounds for _ in 1:n
        i = rand(ctx.rng, 1:n)

        E_old_local = if use_template
            _template_local_energy!(mc, i)
        else
            e = 0.0
            for inst_idx in mc.related_instances_by_atom[i]
                inst = mc.local_cache.instances[inst_idx]
                e += inst.prefactor * _tensor_contract_instance_cached_changed!(
                    mc.contract_other_sites,
                    mc.contract_cart_idx,
                    inst,
                    mc.zlm_cache,
                    i,
                )
            end
            e
        end

        s_old = mc.spins[i]
        theta = mc.spin_theta_max
        sx_new, sy_new, sz_new = if theta isa Float64
            _propose_spin_geodesic(ctx.rng, s_old[1], s_old[2], s_old[3], theta)
        else
            _rand_unit_spin(ctx.rng)
        end
        s_new = SVector(sx_new, sy_new, sz_new)
        zlm_row_buf = mc.zlm_row_buf
        ncols = (mc.max_l + 1)^2
        @inbounds for j in 1:ncols
            zlm_row_buf[j] = mc.zlm_cache[i, j]
        end
        mc.spins[i] = s_new
        _update_atom_zlm_cache!(mc.zlm_cache, i, s_new, mc.sph)

        E_new_local = if use_template
            _template_local_energy!(mc, i)
        else
            e = 0.0
            for inst_idx in mc.related_instances_by_atom[i]
                inst = mc.local_cache.instances[inst_idx]
                e += inst.prefactor * _tensor_contract_instance_cached_changed!(
                    mc.contract_other_sites,
                    mc.contract_cart_idx,
                    inst,
                    mc.zlm_cache,
                    i,
                )
            end
            e
        end

        dE = E_new_local - E_old_local
        if dE <= 0.0 || rand(ctx.rng) < exp(-dE / mc.T)
            mc.energy += dE
        else
            mc.spins[i] = s_old
            @inbounds for j in 1:ncols
                mc.zlm_cache[i, j] = zlm_row_buf[j]
            end
        end
    end
    mc.sweep_count += 1
    if mc.renorm_every > 0 && mc.sweep_count % mc.renorm_every == 0
        @inbounds for i in 1:n
            s = mc.spins[i]
            mc.spins[i] = s / hypot(s[1], s[2], s[3])
        end
        _rebuild_zlm_cache!(mc)
    end
    return nothing
end

function Carlo.measure!(mc::JPhiSpinMC, ctx::MCContext)
    n = mc.ham.n_atoms
    m = sum(mc.spins) / n
    mx, my, mz = m[1], m[2], m[3]
    mag2 = mx^2 + my^2 + mz^2
    mag = sqrt(mag2)
    measure!(ctx, :Energy, mc.energy / n)
    measure!(ctx, :Energy2, (mc.energy / n)^2)
    measure!(ctx, :Magnetization, mag)
    measure!(ctx, :AbsMagnetization, mag)
    measure!(ctx, :Magnetization2, mag2)
    measure!(ctx, :Magnetization4, mag2^2)
    return nothing
end

function Carlo.measure!(mc::JPhiSpinMC, ctx::MCContext, comm::MPI.Comm)
    # In parallel run mode, only rank 0 is allowed to record measurements.
    if MPI.Comm_rank(comm) == 0
        Carlo.measure!(mc, ctx)
    end
    return nothing
end

function Carlo.register_evaluables(::Type{<:JPhiSpinMC}, eval::AbstractEvaluator, params::AbstractDict)
    T = Float64(params[:T])
    rep = _parse_repeat_param(params)
    scm = _parse_supercell_matrix_param(params)
    thr = _parse_jphi_threshold_param(params)
    key = (params[:xml_path], rep, _scm_key(scm), thr)
    # Use the process-local cache if available (populated by JPhiSpinMC constructor),
    # avoiding a redundant full XML parse + cluster enumeration on every rank.
    n = if haskey(_HAM_CACHE, key)
        _HAM_CACHE[key].n_atoms
    elseif scm === nothing
        load_sce_hamiltonian(params[:xml_path]; repeat = rep, jphi_threshold = thr).n_atoms
    else
        load_sce_hamiltonian(params[:xml_path]; supercell_matrix = scm, jphi_threshold = thr).n_atoms
    end
    evaluate!(eval, :SpecificHeat, (:Energy2, :Energy)) do e2, e
        return n * (e2 - e^2) / T^2
    end
    evaluate!(eval, :BinderRatio, (:Magnetization2, :Magnetization4)) do mag2, mag4
        return mag2 * mag2 / mag4
    end
    evaluate!(eval, :Susceptibility, (:Magnetization2,)) do mag2
        return n * mag2 / T
    end
    return nothing
end

# --- Lightweight MPI serialization for Carlo parallel-tempering gather ---
#
# Carlo's PT checkpoint gathers the full MC object from all ranks to rank 0 via
# MPI.gather / Julia Serialization.  Without this override, JPhiSpinMC serializes
# ham::SCEHamiltonian and local_cache::LocalEnergyCache (both O(GB) for large SCE
# bases), causing rank 0 to hold 32+ copies and run OOM on a 256 GiB node.
#
# Only the truly mutable simulation state (T, spins, energy) plus the reconstruction
# keys (xml_path, repeat, enabled_bodies) are written; everything else is rebuilt
# deterministically on deserialization.

# Serialize layout is position-dependent: the deserialize half reads fields in
# the exact order written here. Adding / removing / reordering a field requires
# updating BOTH halves in the same commit. The format is valid only within a
# single SpinClusterMC version — there is no on-disk version tag.
function Serialization.serialize(s::Serialization.AbstractSerializer, mc::JPhiSpinMC)
    # `typeof(mc)` is a concrete `JPhiSpinMC{S}` (DataType) — `JPhiSpinMC` itself
    # is now a UnionAll due to the `{S<:SphericalHarmonics}` parameter, and
    # `serialize_type` only dispatches on DataType. The deserialize half is keyed
    # on `::Type{<:JPhiSpinMC}` so it matches whatever concrete subtype we write.
    Serialization.serialize_type(s, typeof(mc), false)
    Serialization.serialize(s, mc.T)
    Serialization.serialize(s, _spins_to_matrix(mc.spins))
    Serialization.serialize(s, mc.energy)
    Serialization.serialize(s, mc.xml_path)
    Serialization.serialize(s, mc.repeat)
    Serialization.serialize(s, mc.supercell_matrix)
    Serialization.serialize(s, mc.jphi_threshold)
    Serialization.serialize(s, mc.spin_theta_max)
    Serialization.serialize(s, mc.renorm_every)
    Serialization.serialize(s, mc.sweep_count)
    Serialization.serialize(s, mc.enabled_bodies)
    Serialization.serialize(s, mc.energy_kernel)
end

# See the serialize comment above: the order of `deserialize` calls must match
# the writer exactly.
function Serialization.deserialize(s::Serialization.AbstractSerializer, ::Type{<:JPhiSpinMC})
    T            = Serialization.deserialize(s)::Float64
    spins_mat    = Serialization.deserialize(s)::Matrix{Float64}
    spins        = _matrix_to_spins(spins_mat)
    energy       = Serialization.deserialize(s)::Float64
    xml_path     = Serialization.deserialize(s)::String
    repeat       = Serialization.deserialize(s)::NTuple{3,Int}
    supercell_matrix = Serialization.deserialize(s)::Union{Nothing,Matrix{Int}}
    jphi_threshold = Serialization.deserialize(s)::Float64
    spin_theta_max = Serialization.deserialize(s)
    renorm_every = Serialization.deserialize(s)::Int
    sweep_count  = Serialization.deserialize(s)::Int
    enabled_bodies = Serialization.deserialize(s)
    energy_kernel = Serialization.deserialize(s)::Symbol

    # Use the process-local cache (populated at startup via _mpi_build_ham_and_cache).
    # On rank 0, this is called 32 times during Carlo's PT checkpoint gather;
    # the cache ensures ham/local_cache are NOT rebuilt 32 times.
    ham, cache = _mpi_build_ham_and_cache(xml_path, repeat, supercell_matrix, jphi_threshold)

    active_body_indices = _enabled_bodies_to_active_indices(enabled_bodies, cache.body_list)
    derived = _get_or_build_derived(
        xml_path, repeat, supercell_matrix, jphi_threshold, active_body_indices, cache, ham.n_atoms)
    zlm_cache = _alloc_zlm_cache(ham.n_atoms, derived.max_l)
    zlm_row_buf = Vector{Float64}(undef, (derived.max_l + 1)^2)
    sph = SphericalHarmonics(derived.max_l)
    local_template, atoms_buf = if energy_kernel === :tensor_template
        tpl = build_local_energy_template(ham)
        (tpl, Vector{Int}(undef, max(derived.max_sites, 1)))
    else
        (nothing, Int[])
    end

    mc = JPhiSpinMC(
        T, ham, spins, energy, cache,
        derived.active_body_indices, derived.active_instance_indices, derived.related_instances_by_atom,
        derived.max_l, zlm_cache, zlm_row_buf, sph,
        Vector{Int}(undef, derived.max_sites), Vector{Int}(undef, derived.max_sites),
        spin_theta_max, renorm_every, sweep_count,
        xml_path, repeat, supercell_matrix, jphi_threshold, enabled_bodies,
        energy_kernel, local_template, atoms_buf,
    )
    _rebuild_zlm_cache!(mc)
    return mc
end

function Carlo.write_checkpoint(mc::JPhiSpinMC, out::HDF5.Group)
    out["spins"] = _spins_to_matrix(mc.spins)
    out["energy"] = mc.energy
    return nothing
end

function Carlo.write_checkpoint(
    mc::JPhiSpinMC,
    out::Union{HDF5.Group,Nothing},
    comm::MPI.Comm,
)
    all_spins = MPI.gather(_spins_to_matrix(mc.spins), comm)
    all_energies = MPI.Gather(mc.energy, comm)

    if MPI.Comm_rank(comm) == 0
        out_grp = out::HDF5.Group
        out_grp["spins"] = cat((all_spins::Vector)...; dims = 3)
        out_grp["energy"] = all_energies
    end
    return nothing
end

function Carlo.read_checkpoint!(mc::JPhiSpinMC, in::HDF5.Group)
    spins_mat = read(in, "spins")::Matrix{Float64}
    @inbounds for i in eachindex(mc.spins)
        mc.spins[i] = SVector{3,Float64}(spins_mat[1, i], spins_mat[2, i], spins_mat[3, i])
    end
    mc.energy = read(in, "energy")
    _rebuild_zlm_cache!(mc)
    return nothing
end

function Carlo.read_checkpoint!(
    mc::JPhiSpinMC,
    in::Union{HDF5.Group,Nothing},
    comm::MPI.Comm,
)
    if MPI.Comm_rank(comm) == 0
        in_grp = in::HDF5.Group
        spins_all = read(in_grp, "spins")
        energies = vec(read(in_grp, "energy"))
        spins_per_rank = [copy(s) for s in eachslice(spins_all; dims = 3)]
    else
        spins_per_rank = nothing
        energies = nothing
    end

    spins_mat = MPI.scatter(spins_per_rank, comm)::Matrix{Float64}
    @inbounds for i in eachindex(mc.spins)
        mc.spins[i] = SVector{3,Float64}(spins_mat[1, i], spins_mat[2, i], spins_mat[3, i])
    end
    mc.energy = MPI.Scatter(energies, Float64, comm)
    _rebuild_zlm_cache!(mc)
    return nothing
end

function Carlo.parallel_tempering_log_weight_ratio(mc::JPhiSpinMC, parameter::Symbol, new_value)
    parameter == :T || error("parallel tempering not implemented for $parameter")
    -(1 / Float64(new_value) - 1 / mc.T) * mc.energy
end

function Carlo.parallel_tempering_change_parameter!(mc::JPhiSpinMC, parameter::Symbol, new_value)
    parameter == :T || error("parallel tempering not implemented for $parameter")
    mc.T = Float64(new_value)
end

# Compatibility patch helper for Carlo.jl checkpoints:
# In some Carlo versions, `ParallelMeasurements` checkpoints with an empty queue
# may not contain the "names" group. The upstream reader assumes it always exists
# and throws `KeyError: key "names" not found` on resume.
function _read_parallel_measurements_checkpoint(in::HDF5.Group)
    if !haskey(in, "names")
        return Carlo.ParallelMeasurements()
    end

    saved_values = read(in, "names")
    if isempty(saved_values)
        return Carlo.ParallelMeasurements()
    end

    queue = Vector{Tuple{Symbol,Any}}(
        undef,
        maximum(x -> maximum(x["order"]), values(saved_values)),
    )

    collapse_scalar(x) = x
    collapse_scalar(x::AbstractArray{<:Any,0}) = x[]

    for (name, vals) in saved_values
        for (i, v) in zip(vals["order"], eachslice(vals["values"]; dims = ndims(vals["values"])))
            queue[i] = (Symbol(name), collapse_scalar(v))
        end
    end

    return Carlo.ParallelMeasurements(queue)
end

function __init__()
    # Patch at runtime (not during precompile) to avoid precompile-time method-overwrite errors.
    # Use $ to interpolate the function object directly — the name `JPhiMagestyCarlo` is not
    # in Carlo's namespace and would cause UndefVarError when the method is called.
    fn = _read_parallel_measurements_checkpoint
    @eval Carlo begin
        function read_checkpoint(::Type{ParallelMeasurements}, in::HDF5.Group)
            return $fn(in)
        end
    end
end

end # module
