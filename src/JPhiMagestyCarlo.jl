"""
	JPhiMagestyCarlo

Reads Magesty `jphi.xml` (`SCEBasisSet`, `AngularMomentumCouplings`, `JPhi`), evaluates SCE energy
in spin directions with the same contract as `Magesty.Optimize.design_matrix_energy_element`, and
provides a thin [`Carlo`](https://github.com/lattice-quantum/Carlo.jl) `AbstractMC` adapter with
single-spin Metropolis updates. Local energy deltas reuse preallocated stride / index buffers (no
per-update `Vector` allocations in the tensor contraction). Optional task parameter `spin_theta_max`
selects a local geodesic spin proposal instead of i.i.d. uniform-on-sphere draws.

Real (tesseral) spherical harmonics use `SpheriCart.SphericalHarmonics` with its default
`:L2` (L2-orthonormal) normalisation, which is bit-exact with Magesty's `Zₗₘ_unsafe` for
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

using Magesty.Basis: CoupledBasis_with_coefficient
using Magesty.XMLIO: read_basisset_from_xml
using SpheriCart: SphericalHarmonics, compute, compute!

export SCEHamiltonian,
    load_sce_hamiltonian,
    sce_energy,
    coupled_cluster_energy,
    supercell_atom_index,
    interaction_partners,
    interaction_partners_by_body,
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
const _HAM_CACHE    = Dict{Tuple{String,NTuple{3,Int}}, SCEHamiltonian}()
const _ECACHE_CACHE = Dict{Tuple{String,NTuple{3,Int}}, LocalEnergyCache}()

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

const _DERIVED_CACHE = Dict{Tuple{String,NTuple{3,Int},Tuple}, DerivedInstanceCache}()

"""
Return `(4π)^(n_sites/2)` normalization used for cluster contributions.
"""
@inline _cluster_scaling(n_sites::Integer)::Float64 = (4 * pi)^(n_sites / 2)

@inline function supercell_atom_index(
    base_atom::Int,
    ti::Integer,
    tj::Integer,
    tk::Integer,
    base_n::Int,
    repeat::NTuple{3, Int},
)::Int
    n1, n2, n3 = repeat
    1 ≤ base_atom ≤ base_n || throw(ArgumentError("base_atom=$base_atom not in 1:$base_n"))
    0 ≤ ti < n1 && 0 ≤ tj < n2 && 0 ≤ tk < n3 ||
        throw(ArgumentError("tile ($ti,$tj,$tk) out of range for repeat=$repeat"))
    return base_atom + base_n * (ti + n1 * tj + n1 * n2 * tk)
end

"""
Build supercell lattice and wrapped fractional positions from base-cell data.
"""
function _build_supercell_geometry(
    lattice::Matrix{Float64},
    pos_base_frac::Matrix{Float64},
    base_n::Int,
    repeat::NTuple{3, Int},
)
    n1, n2, n3 = repeat
    a1 = @view lattice[:, 1]
    a2 = @view lattice[:, 2]
    a3 = @view lattice[:, 3]
    lattice_super = hcat(n1 .* a1, n2 .* a2, n3 .* a3)
    n_tot = base_n * n1 * n2 * n3
    pos_super = zeros(3, n_tot)
    for tk in 0:(n3 - 1)
        for tj in 0:(n2 - 1)
            for ti in 0:(n1 - 1)
                for b in 1:base_n
                    ia = supercell_atom_index(b, ti, tj, tk, base_n, repeat)
                    r = lattice * pos_base_frac[:, b] + ti * a1 + tj * a2 + tk * a3
                    x = lattice_super \ r
                    x .-= floor.(x)
                    pos_super[:, ia] .= x
                end
            end
        end
    end
    return lattice_super, pos_super
end

function load_sce_hamiltonian(
    xml_path::AbstractString;
    repeat::NTuple{3, Int} = (1, 1, 1),
)::SCEHamiltonian
    all(r -> r ≥ 1, repeat) || throw(ArgumentError("repeat must be positive integers, got $repeat"))
    basis = read_basisset_from_xml(xml_path)
    sys = parse_system_xml(xml_path)
    jphi = read_jphi_coefficients(xml_path)
    length(jphi) == length(basis.salc_list) ||
        throw(ArgumentError("number of jphi values ($(length(jphi))) != num_salc ($(length(basis.salc_list)))"))
    sys.n_atoms ≥ maximum(
        maximum(maximum(cbc.atoms) for cbc in grp; init = 0) for grp in basis.salc_list;
        init = 0,
    ) ||
        throw(ArgumentError("atom index in basis exceeds NumberOfAtoms"))
    n0 = sys.n_atoms
    lat_s, pos_s = _build_supercell_geometry(sys.lattice, sys.pos_frac, n0, repeat)
    n_super = n0 * repeat[1] * repeat[2] * repeat[3]
    return SCEHamiltonian(
        n_super,
        n0,
        repeat,
        lat_s,
        pos_s,
        basis.salc_list,
        jphi,
        sys.map_sym,
        sys.n_trans,
    )
end

"""
    _mpi_build_ham_and_cache(xml_path, rep) -> (SCEHamiltonian, LocalEnergyCache)

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
)::Tuple{SCEHamiltonian, LocalEnergyCache}
    key = (xml_path, rep)
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
        load_sce_hamiltonian(xml_path; repeat = rep)
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
    active_body_indices::Vector{Int},
    cache::LocalEnergyCache,
    n_atoms::Int,
)::DerivedInstanceCache
    key = (xml_path, rep, Tuple(active_body_indices))
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
	coupled_cluster_energy(cbc, spin_directions, map_sym; repeat, base_n_atoms) -> Float64

Same contract as `Magesty.Optimize.design_matrix_energy_element`. For `repeat=(1,1,1)`, columns of
`map_sym` are XML `trans=1..n_trans` and `map_sym[atom,t]` is the image within the base cell.

With a supercell, for each tile `(ti,tj,tk)` and each translation `t`, base images are mapped to
supercell atoms via `supercell_atom_index(map_sym[a,t], ti,tj,tk, base_n_atoms, repeat)` before the
same tensor contraction. Deduplication always uses **sorted supercell atom indices** together with `ls`.
"""
function coupled_cluster_energy(
    cbc::CoupledBasis_with_coefficient,
    spin_directions::Union{AbstractMatrix{<:Real},AbstractVector{<:SVector{3,<:Real}}},
    map_sym::AbstractMatrix{Int};
    repeat::NTuple{3, Int} = (1, 1, 1),
    base_n_atoms::Int = size(map_sym, 1),
    pos_frac::Union{Nothing, AbstractMatrix{Float64}} = nothing,
)::Float64
    n_trans = size(map_sym, 2)
    n1, n2, n3 = repeat
    n_expect = base_n_atoms * n1 * n2 * n3
    _n_spins(spin_directions) == n_expect ||
        throw(ArgumentError("spin count $(_n_spins(spin_directions)) != supercell atoms $n_expect"))
    result = 0.0
    N = length(cbc.atoms)
    scaling = _cluster_scaling(N)
    searched_pairs = Set{Tuple{Vector{Int}, Vector{Int}}}()

    for tk in 0:(n3 - 1)
        for tj in 0:(n2 - 1)
            for ti in 0:(n1 - 1)
                for t in 1:n_trans
                    translated_base = Int[map_sym[atom, t] for atom in cbc.atoms]
                    if pos_frac !== nothing
                        # pos_frac is expected to be BASE-CELL fractional positions (3×base_n).
                        p_ref = pos_frac[:, translated_base[1]]
                        translated_atoms = Vector{Int}(undef, length(translated_base))
                        for (k, ba) in enumerate(translated_base)
                            p = pos_frac[:, ba]
                            w1 = round(Int, p[1] - p_ref[1])
                            w2 = round(Int, p[2] - p_ref[2])
                            w3 = round(Int, p[3] - p_ref[3])
                            translated_atoms[k] = supercell_atom_index(
                                ba, mod(ti + w1, n1), mod(tj + w2, n2), mod(tk + w3, n3),
                                base_n_atoms, repeat,
                            )
                        end
                    else
                        translated_atoms = Int[
                            supercell_atom_index(ba, ti, tj, tk, base_n_atoms, repeat) for
                            ba in translated_base
                        ]
                    end
                    atoms_sorted = sort(translated_atoms)
                    pair = (atoms_sorted, cbc.ls)
                    pair in searched_pairs && continue
                    push!(searched_pairs, pair)

                    sh_values = Vector{Vector{Float64}}(undef, N)
                    sph_local = SphericalHarmonics(maximum(cbc.ls))
                    for (site_idx, atom) in enumerate(translated_atoms)
                        l = cbc.ls[site_idx]
                        sh_values[site_idx] = Vector{Float64}(undef, 2 * l + 1)
                        u = _spin_at(spin_directions, atom)
                        y = compute(sph_local,
                                    SVector{3,Float64}(u[1], u[2], u[3]))
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

                    result += tensor_result * cbc.multiplicity * scaling
                end
            end
        end
    end

    return result
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

# Iterate over every unique translated cluster instance for `cbc` in the supercell
# defined by `h`, calling `f(translated_atoms)` for each. Deduplication (by sorted
# atom-index tuple) is handled here so callers don't repeat it.
function _foreach_translated_instance(f, h::SCEHamiltonian, cbc)
    n1, n2, n3 = h.repeat
    n1f, n2f, n3f = Float64(n1), Float64(n2), Float64(n3)
    N = length(cbc.atoms)
    seen = Set{Tuple{Vector{Int}, Vector{Int}}}()
    for tk in 0:(n3 - 1), tj in 0:(n2 - 1), ti in 0:(n1 - 1)
        for t in 1:h.n_trans
            translated_base = Int[h.map_sym[a, t] for a in cbc.atoms]
            p_ref = h.pos_frac[:, translated_base[1]]
            f_ref = (p_ref[1] * n1f, p_ref[2] * n2f, p_ref[3] * n3f)
            translated_atoms = Vector{Int}(undef, N)
            for (k, ba) in enumerate(translated_base)
                p = h.pos_frac[:, ba]
                w1 = round(Int, p[1] * n1f - f_ref[1])
                w2 = round(Int, p[2] * n2f - f_ref[2])
                w3 = round(Int, p[3] * n3f - f_ref[3])
                translated_atoms[k] = supercell_atom_index(
                    ba,
                    mod(ti + w1, n1),
                    mod(tj + w2, n2),
                    mod(tk + w3, n3),
                    h.base_n_atoms,
                    h.repeat,
                )
            end
            atoms_sorted = sort(translated_atoms)
            pair = (atoms_sorted, cbc.ls)
            pair in seen && continue
            push!(seen, pair)
            f(translated_atoms)
        end
    end
end

"""
Enumerate unique translated cluster instances and precompute metadata.
"""
function _build_cluster_instances(h::SCEHamiltonian)::Vector{ClusterInstance}
    instances = ClusterInstance[]
    # Shared coeff_flat per unique cbc object: multiple ClusterInstances that are
    # geometric translations of the same cbc would otherwise each get a separate
    # Vector allocation, multiplying memory by the number of translations.
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
            _foreach_translated_instance(h, cbc) do translated_atoms
                push!(
                    instances,
                    ClusterInstance(
                        translated_atoms,
                        cbc,
                        js * cbc.multiplicity * scaling,
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

@inline interaction_partners(cache::LocalEnergyCache, atom::Int)::Vector{Int} =
    cache.partners_by_atom[atom]

function interaction_partners_by_body(
    cache::LocalEnergyCache,
    atom::Int,
)::Dict{Int, Vector{Int}}
    out = Dict{Int, Vector{Int}}()
    for (bidx, body) in enumerate(cache.body_list)
        out[body] = cache.partners_by_atom_by_body[bidx][atom]
    end
    return out
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

SCE interaction energy (j0 excluded) for each SALC index `s`: the weighted sum of
`coupled_cluster_energy` over every coupled cluster in `h.salc_list[s]`, with weights `h.jphi[s]`.
j0 is intentionally excluded because this package is used for MC sampling where only ΔE matters.

`spin_directions` should be `3 × h.n_atoms`: rows 1–3 are `x`, `y`, `z` of the spin direction; columns are
supercell atoms (`a` → column `a`). Only the column count is checked here; each column is passed to
`SpheriCart.compute` as a 3-vector inside `coupled_cluster_energy`. Shape matches `h.map_sym`, `h.repeat`, and
`h.base_n_atoms` as in that routine.
"""
function sce_energy(
    h::SCEHamiltonian,
    spin_directions::Union{AbstractMatrix{<:Real},AbstractVector{<:SVector{3,<:Real}}},
)::Float64
    E = 0.0
    n1, n2, n3 = h.repeat
    # Recover base-cell fractional positions from supercell positions (tile-(0,0,0) block).
    # h.pos_frac[:,ba] for ba in 1:base_n_atoms = base_frac / (n1,n2,n3).
    base_pos = h.pos_frac[:, 1:h.base_n_atoms] .* [Float64(n1); Float64(n2); Float64(n3)]
    for (s, group) in enumerate(h.salc_list)
        js = h.jphi[s]
        for cbc in group
            E += js * coupled_cluster_energy(
                cbc,
                spin_directions,
                h.map_sym;
                repeat = h.repeat,
                base_n_atoms = h.base_n_atoms,
                pos_frac = base_pos,
            )
        end
    end
    return E
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
| `:repeat` or `:supercell` | 3-vector of `Int` | `(1,1,1)` | Tiling of the primitive cell read from the XML. Total atom count becomes `base_n_atoms × n₁ × n₂ × n₃`. |

## Initial spin configuration
| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `:initial_spins` | `Matrix{<:Real}` of size `(3, base_n_atoms)` | (random) | Spin configuration for the **base cell** (`repeat = (1,1,1)`). If provided, this configuration is tiled periodically over the full supercell by `Carlo.init!`; otherwise all spins are drawn uniformly at random on the unit sphere. Each column is renormalized to a unit vector automatically. See [`_tile_base_spins!`](@ref) for the tiling convention. |

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

## Energy kernel
| Key | Type | Default | Description |
|:----|:-----|:--------|:------------|
| `:energy_kernel` | `Symbol` | `:tensor_template` | `:tensor_template` (default) stores only base-cell cluster instances and reconstructs supercell atom indices on-the-fly during `sweep!`, giving O(n_base_instances) memory regardless of supercell size. N=2 / N=3 instances use SVector-specialized contraction kernels. `:tensor` enumerates every translated instance up front — uses more memory and is mainly kept as a validation reference. The two kernels agree to within floating-point summation order. See [`build_local_energy_template`](@ref). |

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
    enabled_bodies::Union{Nothing,Vector{Int}}
    # `:tensor_template` (default) uses base-cell cluster templates with on-the-fly
    # supercell index reconstruction (`_template_local_energy!`); SVector-specialized
    # for N=2 / N=3 clusters. `:tensor` uses the fully-enumerated SALC tensor-contraction
    # kernel via `_energy_from_instances_cached` / `_tensor_contract_instance_cached_changed!`.
    energy_kernel::Symbol
    # Populated only when `energy_kernel === :tensor_template`.
    local_template::Union{Nothing,LocalEnergyTemplate}
    atoms_buf::Vector{Int}  # reused buffer in sweep! to avoid per-instance allocation
end

@inline interaction_partners(mc::JPhiSpinMC, atom::Int)::Vector{Int} =
    interaction_partners(mc.local_cache, atom)

@inline interaction_partners_by_body(mc::JPhiSpinMC, atom::Int)::Dict{Int, Vector{Int}} =
    interaction_partners_by_body(mc.local_cache, atom)

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
        ham = get!(_HAM_CACHE, (xml, rep)) do
            load_sce_hamiltonian(xml; repeat = rep)
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
            xml, rep, enabled_bodies, energy_kernel,
            local_template, atoms_buf,
        )
    end

    # :tensor path: build the full LocalEnergyCache.
    ham, cache = _mpi_build_ham_and_cache(xml, rep)
    active_body_indices = _parse_enabled_body_indices(params, cache.body_list)
    derived = _get_or_build_derived(xml, rep, active_body_indices, cache, ham.n_atoms)
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
    rep = mc.ham.repeat
    base_n = mc.ham.base_n_atoms
    b = ((i - 1) % base_n) + 1
    e = 0.0

    # N=2 fast path: SAIs precomputed in tpl.sai2_flat (no mod / no _tile_coords).
    related2 = tpl.related2_by_base_atom[b]
    sai2 = tpl.sai2_flat
    sai2_off = tpl.sai2_offsets[i] - 1
    @inbounds for rc_idx in 1:length(related2)
        rc = related2[rc_idx]
        inst2 = tpl.base_instances2[rc.inst_idx]
        slot = sai2_off + 2 * (rc_idx - 1)
        a1 = sai2[slot + 1]
        a2 = sai2[slot + 2]
        e += inst2.prefactor * _tensor_contract_template2_changed!(
            inst2, a1, a2, mc.zlm_cache, i,
        )
    end

    # N=3 fast path: SAIs precomputed in tpl.sai3_flat.
    related3 = tpl.related3_by_base_atom[b]
    sai3 = tpl.sai3_flat
    sai3_off = tpl.sai3_offsets[i] - 1
    @inbounds for rc_idx in 1:length(related3)
        rc = related3[rc_idx]
        inst3 = tpl.base_instances3[rc.inst_idx]
        slot = sai3_off + 3 * (rc_idx - 1)
        a1 = sai3[slot + 1]
        a2 = sai3[slot + 2]
        a3 = sai3[slot + 3]
        e += inst3.prefactor * _tensor_contract_template3_changed!(
            inst3, a1, a2, a3, mc.zlm_cache, i,
        )
    end

    # N≥4 general path: keeps on-the-fly SAI computation (no precomputed table).
    # Both current test problems have zero N≥4 instances, so this branch is unused
    # there; the local `_tile_coords` call below is only paid when N≥4 clusters
    # actually exist.
    related_other = tpl.related_by_base_atom[b]
    if !isempty(related_other)
        n1, n2, n3 = rep
        ti, tj, tk = _tile_coords(i, base_n, rep)
        for rc in related_other
            inst = tpl.base_instances[rc.inst_idx]
            N = length(inst.base_atoms)
            pv1, pv2, pv3 = inst.tile_deltas[rc.pivot_k]
            @inbounds for k in 1:N
                d1, d2, d3 = inst.tile_deltas[k]
                mc.atoms_buf[k] = supercell_atom_index(
                    inst.base_atoms[k],
                    mod(ti + d1 - pv1, n1),
                    mod(tj + d2 - pv2, n2),
                    mod(tk + d3 - pv3, n3),
                    base_n,
                    rep,
                )
            end
            e += inst.prefactor * _tensor_contract_template_changed!(
                mc.contract_other_sites,
                mc.contract_cart_idx,
                inst,
                mc.atoms_buf,
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

- **With `:initial_spins`** — expects a `3 × base_n_atoms` matrix whose columns
  are spin vectors for the base cell (`repeat = (1,1,1)` in the XML).
  The configuration is tiled periodically over the full supercell via
  [`_tile_base_spins!`](@ref): supercell atom `ia` is assigned the spin of base
  atom `((ia-1) % base_n_atoms) + 1`.  Each column is renormalized to a unit
  vector; a zero-norm column raises `ArgumentError`.

  ```julia
  # Ferromagnetic +z start for a 16-atom base cell
  s0 = zeros(3, 16); s0[3, :] .= 1.0
  params[:initial_spins] = s0
  ```

- **Without `:initial_spins`** — all spins are drawn independently and uniformly
  on the unit sphere using the seeded RNG in `ctx`.

After setting the spin configuration, the spherical-harmonic cache and the
stored energy are rebuilt consistently.
"""
function Carlo.init!(mc::JPhiSpinMC, ctx::MCContext, params::AbstractDict)
    n = mc.ham.n_atoms
    if haskey(params, :initial_spins)
        _tile_base_spins!(mc.spins, params[:initial_spins], mc.ham.base_n_atoms)
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
    key = (params[:xml_path], _parse_repeat_param(params))
    # Use the process-local cache if available (populated by JPhiSpinMC constructor),
    # avoiding a redundant full XML parse + cluster enumeration on every rank.
    n = if haskey(_HAM_CACHE, key)
        _HAM_CACHE[key].n_atoms
    else
        load_sce_hamiltonian(key[1]; repeat = key[2]).n_atoms
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

function Serialization.serialize(s::Serialization.AbstractSerializer, mc::JPhiSpinMC)
    Serialization.serialize_type(s, JPhiSpinMC, false)
    Serialization.serialize(s, mc.T)
    Serialization.serialize(s, _spins_to_matrix(mc.spins))
    Serialization.serialize(s, mc.energy)
    Serialization.serialize(s, mc.xml_path)
    Serialization.serialize(s, mc.repeat)
    Serialization.serialize(s, mc.spin_theta_max)
    Serialization.serialize(s, mc.renorm_every)
    Serialization.serialize(s, mc.sweep_count)
    Serialization.serialize(s, mc.enabled_bodies)
    Serialization.serialize(s, mc.energy_kernel)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, ::Type{<:JPhiSpinMC})
    T            = Serialization.deserialize(s)::Float64
    spins_mat    = Serialization.deserialize(s)::Matrix{Float64}
    spins        = _matrix_to_spins(spins_mat)
    energy       = Serialization.deserialize(s)::Float64
    xml_path     = Serialization.deserialize(s)::String
    repeat       = Serialization.deserialize(s)::NTuple{3,Int}
    spin_theta_max = Serialization.deserialize(s)
    renorm_every = Serialization.deserialize(s)::Int
    sweep_count  = Serialization.deserialize(s)::Int
    enabled_bodies = Serialization.deserialize(s)
    energy_kernel = Serialization.deserialize(s)::Symbol

    # Use the process-local cache (populated at startup via _mpi_build_ham_and_cache).
    # On rank 0, this is called 32 times during Carlo's PT checkpoint gather;
    # the cache ensures ham/local_cache are NOT rebuilt 32 times.
    ham, cache = _mpi_build_ham_and_cache(xml_path, repeat)

    active_body_indices = _enabled_bodies_to_active_indices(enabled_bodies, cache.body_list)
    derived = _get_or_build_derived(xml_path, repeat, active_body_indices, cache, ham.n_atoms)
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
        xml_path, repeat, enabled_bodies,
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
