"""
	JPhiMagestyCarlo

Reads Magesty `jphi.xml` (`SCEBasisSet`, `AngularMomentumCouplings`, `JPhi`), evaluates SCE energy
in spin directions with the same contract as `Magesty.Optimize.design_matrix_energy_element`, and
provides a thin [`Carlo`](https://github.com/lattice-quantum/Carlo.jl) `AbstractMC` adapter with
single-spin Metropolis updates. Local energy deltas reuse preallocated stride / index buffers (no
per-update `Vector` allocations in the tensor contraction). Optional task parameter `spin_theta_max`
selects a local geodesic spin proposal instead of i.i.d. uniform-on-sphere draws.

Real (tesseral) spherical harmonics use `Magesty.MySphericalHarmonics.Zₗₘ`.
"""
module JPhiMagestyCarlo

using Carlo
using HDF5
using MPI
using EzXML
using LinearAlgebra
using Random
import Serialization

using Magesty.Basis: CoupledBasis_with_coefficient
using Magesty.MySphericalHarmonics: Zₗₘ_unsafe
using Magesty.XMLIO: read_basisset_from_xml

export SCEHamiltonian,
    load_sce_hamiltonian,
    sce_energy,
    coupled_cluster_energy,
    supercell_atom_index,
    interaction_partners,
    interaction_partners_by_body,
    JPhiSpinMC

# --- XML: lattice, positions, translation maps (infer missing rows from geometry if XML lists prim atoms only) ---

struct SystemXMLInfo
    n_atoms::Int
    lattice::Matrix{Float64}
    periodicity::NTuple{3, Int}
    pos_frac::Matrix{Float64}
    n_trans::Int
    map_sym::Matrix{Int}
end

"""
Parse a whitespace-separated 3-vector string into `Float64` values.
"""
function _parse_vec3(s::AbstractString)
    p = parse.(Float64, split(s))
    length(p) == 3 || throw(ArgumentError("expected 3 floats, got $(repr(s))"))
    return p
end

"""
Wrap fractional coordinates into the minimum-image range around zero.
"""
function _min_image_frac(v::AbstractVector{<:Real})
    w = collect(Float64, v)
    @inbounds for i in eachindex(w)
        w[i] -= round(w[i])
    end
    return w
end

"""
Compute minimum-image distance between two fractional coordinates.
"""
function _frac_periodic_dist(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    return norm(_min_image_frac(a .- b))
end

"""
Fill missing entries of translation `t` in `map_sym` by periodic nearest-image matching.

# Arguments
- `map_sym::Matrix{Int}`: Translation map table (`atom × trans`) updated in place.
- `pos_frac::AbstractMatrix{Float64}`: Fractional atomic positions (`3 × n_atoms`).
- `t::Int`: Translation-column index to complete.
- `n_atoms::Int`: Number of atoms to process in the base cell.
"""
function _infer_atom_map_from_atom1!(
    map_sym::Matrix{Int},
    pos_frac::AbstractMatrix{Float64},
    t::Int,
    n_atoms::Int,
)
    j1 = map_sym[1, t]
    j1 == 0 && error("translation $t: missing map for atom 1, cannot infer other atoms")
    δ = _min_image_frac(pos_frac[:, j1] - pos_frac[:, 1])
    for j in 1:n_atoms
        map_sym[j, t] != 0 && continue
        target = pos_frac[:, j] + δ
        target .-= floor.(target)
        best_k = 0
        best_d = Inf
        for k in 1:n_atoms
            d = _frac_periodic_dist(pos_frac[:, k], target)
            if d < best_d
                best_d = d
                best_k = k
            end
        end
        best_d < 1e-5 || error(
            "translation $t: could not infer image of atom $j (best periodic dist=$best_d)",
        )
        map_sym[j, t] = best_k
    end
    return nothing
end

function parse_system_xml(xml_path::AbstractString)::SystemXMLInfo
    doc = readxml(xml_path)
    system_node = findfirst("//System", doc)
    isnothing(system_node) && throw(ArgumentError("no //System in $xml_path"))

    n_atoms = parse(Int, nodecontent(findfirst("NumberOfAtoms", system_node)))

    lat_node = findfirst("LatticeVector", system_node)
    a1 = _parse_vec3(nodecontent(findfirst("a1", lat_node)))
    a2 = _parse_vec3(nodecontent(findfirst("a2", lat_node)))
    a3 = _parse_vec3(nodecontent(findfirst("a3", lat_node)))
    lattice = hcat(a1, a2, a3)

    per_el = findfirst("Periodicity", system_node)
    per_ints = parse.(Int, split(nodecontent(per_el)))
    length(per_ints) == 3 || throw(ArgumentError("Periodicity must have 3 integers"))
    per = (per_ints[1], per_ints[2], per_ints[3])

    pos_frac = zeros(3, n_atoms)
    pos_block = findfirst("Positions", system_node)
    for p in findall("pos", pos_block)
        ia = parse(Int, p["atom_index"])
        pos_frac[:, ia] .= _parse_vec3(nodecontent(p))
    end

    sym_node = findfirst("Symmetry", system_node)
    n_trans = parse(Int, nodecontent(findfirst("NumberOfTranslations", sym_node)))
    trans_block = findfirst("Translations", sym_node)
    map_sym = zeros(Int, n_atoms, n_trans)
    for m in findall("map", trans_block)
        t = parse(Int, m["trans"])
        a = parse(Int, m["atom"])
        dest = parse(Int, nodecontent(m))
        (1 ≤ t ≤ n_trans && 1 ≤ a ≤ n_atoms) || throw(ArgumentError("invalid map trans=$t atom=$a"))
        map_sym[a, t] = dest
    end

    for t in 1:n_trans
        if any(iszero, map_sym[:, t])
            _infer_atom_map_from_atom1!(map_sym, pos_frac, t, n_atoms)
        end
    end

    return SystemXMLInfo(n_atoms, lattice, per, pos_frac, n_trans, map_sym)
end

function read_jphi_coefficients(xml_path::AbstractString)::Tuple{Float64, Vector{Float64}}
    doc = readxml(xml_path)
    jnode = findfirst("//JPhi", doc)
    isnothing(jnode) && throw(ArgumentError("no //JPhi in $xml_path"))
    j0 = parse(Float64, nodecontent(findfirst("ReferenceEnergy", jnode)))
    pairs = Tuple{Int, Float64}[]
    for el in findall("jphi", jnode)
        push!(pairs, (parse(Int, el["salc_index"]), parse(Float64, nodecontent(el))))
    end
    sort!(pairs)
    for (i, (si, _)) in enumerate(pairs)
        si == i || throw(ArgumentError("jphi salc_index must be 1..n without gaps; got index $si at position $i"))
    end
    return j0, last.(pairs)
end

"""
	SCEHamiltonian

Reconstructed from `jphi.xml`: SALC list, **base cell** translation map `map_sym` of size
`base_n_atoms × n_trans`, and `j0` / `jphi` coefficients.

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
    j0::Float64
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
    j0, jphi = read_jphi_coefficients(xml_path)
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
        j0,
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
    if haskey(_HAM_CACHE, key)
        return _HAM_CACHE[key], _ECACHE_CACHE[key]
    end
    # Each MPI rank builds its own copy in parallel.  Broadcasting via
    # Julia Serialization is impractical here: the object graph (millions of
    # ClusterInstance nodes with shared cbc references) takes longer to
    # serialize than to rebuild from XML, and the broadcast buffer itself
    # requires a second copy on every rank.  The process-local cache below
    # only prevents redundant rebuilds *within the same process* (e.g. from
    # Carlo.register_evaluables calling load_sce_hamiltonian a second time).
    ham   = load_sce_hamiltonian(xml_path; repeat = rep)
    cache = build_local_energy_cache(ham)
    _HAM_CACHE[key]    = ham
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
    spin_directions::AbstractMatrix{<:Real},
    map_sym::AbstractMatrix{Int};
    repeat::NTuple{3, Int} = (1, 1, 1),
    base_n_atoms::Int = size(map_sym, 1),
    pos_frac::Union{Nothing, AbstractMatrix{Float64}} = nothing,
)::Float64
    n_trans = size(map_sym, 2)
    n1, n2, n3 = repeat
    n_expect = base_n_atoms * n1 * n2 * n3
    size(spin_directions, 2) == n_expect ||
        throw(ArgumentError("spin columns $(size(spin_directions,2)) != supercell atoms $n_expect"))
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
                    for (site_idx, atom) in enumerate(translated_atoms)
                        l = cbc.ls[site_idx]
                        sh_values[site_idx] = Vector{Float64}(undef, 2 * l + 1)
                        for m_idx in 1:(2 * l + 1)
                            m = m_idx - l - 1
                            u = @view spin_directions[:, atom]
                            sh_values[site_idx][m_idx] = Zₗₘ_unsafe(l, m, u)
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
    spin_directions::AbstractMatrix{<:Real},
)::Float64
    N = length(cbc.atoms)
    sh_values = Vector{Vector{Float64}}(undef, N)
    for (site_idx, atom) in enumerate(translated_atoms)
        l = cbc.ls[site_idx]
        sh_values[site_idx] = Vector{Float64}(undef, 2 * l + 1)
        for m_idx in 1:(2 * l + 1)
            m = m_idx - l - 1
            u = @view spin_directions[:, atom]
            sh_values[site_idx][m_idx] = Zₗₘ_unsafe(l, m, u)
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
Enumerate unique translated cluster instances and precompute metadata.
"""
function _build_cluster_instances(h::SCEHamiltonian)::Vector{ClusterInstance}
    instances = ClusterInstance[]
    # Shared coeff_flat per unique cbc object: multiple ClusterInstances that are
    # geometric translations of the same cbc would otherwise each get a separate
    # Vector allocation, multiplying memory by the number of translations.
    coeff_flat_cache = Dict{UInt, Vector{Float64}}()
    n1, n2, n3 = h.repeat
    n_trans = h.n_trans

    for (s, group) in enumerate(h.salc_list)
        js = h.jphi[s]
        for cbc in group
            scaling = _cluster_scaling(length(cbc.atoms))
            searched_pairs = Set{Tuple{Vector{Int}, Vector{Int}}}()
            for tk in 0:(n3 - 1)
                for tj in 0:(n2 - 1)
                    for ti in 0:(n1 - 1)
                        for t in 1:n_trans
                            translated_base = Int[h.map_sym[atom, t] for atom in cbc.atoms]
                            # Each atom may belong to a different tile: compute tile offset
                            # from the minimum-image wrapping vector relative to atom[1].
                            # h.pos_frac[:,ba] stores supercell fractional positions (i.e.
                            # base-cell frac / repeat). Multiply back by repeat to get base-cell
                            # fractional coords before rounding to the wrapping integer.
                            n1f, n2f, n3f = Float64(n1), Float64(n2), Float64(n3)
                            p_ref = h.pos_frac[:, translated_base[1]]
                            f_ref = (p_ref[1] * n1f, p_ref[2] * n2f, p_ref[3] * n3f)
                            translated_atoms = Vector{Int}(undef, length(translated_base))
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
                            pair in searched_pairs && continue
                            push!(searched_pairs, pair)
                            inst_dims = [2 * l + 1 for l in cbc.ls]
                            inst_strides = _compute_instance_strides(cbc.ls)
                            N_cbc = length(cbc.atoms)
                            inst_Mf_size = size(cbc.coeff_tensor, N_cbc + 1)
                            inst_coeff_flat = get!(coeff_flat_cache, objectid(cbc)) do
                                vec(collect(Float64, cbc.coeff_tensor))
                            end
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
    spin_directions::AbstractMatrix{<:Real},
)::Float64
    E = 0.0
    for inst in instances
        E += inst.prefactor * _tensor_contract_instance(inst.cbc, inst.atoms, spin_directions)
    end
    return E
end

"""
    sce_energy(h, spin_directions) -> Float64

Total SCE Hamiltonian energy: constant offset `h.j0` plus, for each SALC index `s`, the weighted sum of
`coupled_cluster_energy` over every coupled cluster in `h.salc_list[s]`, with weights `h.jphi[s]`.

`spin_directions` should be `3 × h.n_atoms`: rows 1–3 are `x`, `y`, `z` of the spin direction; columns are
supercell atoms (`a` → column `a`). Only the column count is checked here; each column is passed to
`Zₗₘ_unsafe` as a 3-vector inside `coupled_cluster_energy`. Shape matches `h.map_sym`, `h.repeat`, and
`h.base_n_atoms` as in that routine.
"""
function sce_energy(h::SCEHamiltonian, spin_directions::AbstractMatrix{<:Real})::Float64
    E = h.j0
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
mutable struct JPhiSpinMC <: AbstractMC
    T::Float64
    ham::SCEHamiltonian
    spins::Matrix{Float64}
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
end

@inline interaction_partners(mc::JPhiSpinMC, atom::Int)::Vector{Int} =
    interaction_partners(mc.local_cache, atom)

@inline interaction_partners_by_body(mc::JPhiSpinMC, atom::Int)::Dict{Int, Vector{Int}} =
    interaction_partners_by_body(mc.local_cache, atom)

function JPhiSpinMC(params::AbstractDict)
    xml = params[:xml_path]
    rep = _parse_repeat_param(params)
    # MPI-aware: only rank 0 builds from XML; all ranks share via Bcast.
    ham, cache = _mpi_build_ham_and_cache(xml, rep)
    T = Float64(params[:T])
    active_body_indices = _parse_enabled_body_indices(params, cache.body_list)
    derived = _get_or_build_derived(xml, rep, active_body_indices, cache, ham.n_atoms)
    zlm_cache = _alloc_zlm_cache(ham.n_atoms, derived.max_l)
    zlm_row_buf = Vector{Float64}(undef, (derived.max_l + 1)^2)
    other_sites_work = Vector{Int}(undef, derived.max_sites)
    cart_idx_work = Vector{Int}(undef, derived.max_sites)
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
    enabled_bodies = if haskey(params, :enabled_bodies)
        Int.(collect(params[:enabled_bodies]))
    else
        nothing
    end
    return JPhiSpinMC(
        T,
        ham,
        zeros(3, ham.n_atoms),
        0.0,
        cache,
        derived.active_body_indices,
        derived.active_instance_indices,
        derived.related_instances_by_atom,
        derived.max_l,
        zlm_cache,
        zlm_row_buf,
        other_sites_work,
        cart_idx_work,
        spin_theta_max,
        renorm_every,
        0,
        xml,
        rep,
        enabled_bodies,
    )
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
Rebuild the full per-atom `Z_lm` cache from current MC spins.
"""
function _rebuild_zlm_cache!(mc::JPhiSpinMC)
    @inbounds for atom in 1:mc.ham.n_atoms
        _update_atom_zlm_cache!(mc.zlm_cache, atom, @view(mc.spins[:, atom]), mc.max_l)
    end
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
            mc.spins[1, i] = sx
            mc.spins[2, i] = sy
            mc.spins[3, i] = sz
        end
    end
    _rebuild_zlm_cache!(mc)
    mc.energy = mc.ham.j0 + _energy_from_instances(
        mc.local_cache.instances[mc.active_instance_indices],
        mc.spins,
    )
    return nothing
end

function Carlo.sweep!(mc::JPhiSpinMC, ctx::MCContext)
    n = mc.ham.n_atoms
    @inbounds for _ in 1:n
        i = rand(ctx.rng, 1:n)
        related_instances = mc.related_instances_by_atom[i]

        E_old_local = 0.0
        for inst_idx in related_instances
            inst = mc.local_cache.instances[inst_idx]
            E_old_local += inst.prefactor * _tensor_contract_instance_cached_changed!(
                mc.contract_other_sites,
                mc.contract_cart_idx,
                inst,
                mc.zlm_cache,
                i,
            )
        end

        sold = @view mc.spins[:, i]
        sx_old, sy_old, sz_old = sold[1], sold[2], sold[3]
        sx_new, sy_new, sz_new = if mc.spin_theta_max === nothing
            _rand_unit_spin(ctx.rng)
        else
            _propose_spin_geodesic(ctx.rng, sx_old, sy_old, sz_old, mc.spin_theta_max)
        end
        zlm_row_buf = mc.zlm_row_buf
        ncols = (mc.max_l + 1)^2
        @inbounds for j in 1:ncols
            zlm_row_buf[j] = mc.zlm_cache[i, j]
        end
        sold[1], sold[2], sold[3] = sx_new, sy_new, sz_new
        _update_atom_zlm_cache!(mc.zlm_cache, i, sold, mc.max_l)

        E_new_local = 0.0
        for inst_idx in related_instances
            inst = mc.local_cache.instances[inst_idx]
            E_new_local += inst.prefactor * _tensor_contract_instance_cached_changed!(
                mc.contract_other_sites,
                mc.contract_cart_idx,
                inst,
                mc.zlm_cache,
                i,
            )
        end

        dE = E_new_local - E_old_local
        if dE <= 0.0 || rand(ctx.rng) < exp(-dE / mc.T)
            mc.energy += dE
        else
            sold[1], sold[2], sold[3] = sx_old, sy_old, sz_old
            @inbounds for j in 1:ncols
                mc.zlm_cache[i, j] = zlm_row_buf[j]
            end
        end
    end
    mc.sweep_count += 1
    if mc.renorm_every > 0 && mc.sweep_count % mc.renorm_every == 0
        @inbounds for i in 1:n
            s = @view mc.spins[:, i]
            inv_nrm = 1.0 / hypot(s[1], s[2], s[3])
            s[1] *= inv_nrm
            s[2] *= inv_nrm
            s[3] *= inv_nrm
        end
        _rebuild_zlm_cache!(mc)
    end
    return nothing
end

function Carlo.measure!(mc::JPhiSpinMC, ctx::MCContext)
    n = mc.ham.n_atoms
    mx = sum(@view mc.spins[1, :]) / n
    my = sum(@view mc.spins[2, :]) / n
    mz = sum(@view mc.spins[3, :]) / n
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

function Carlo.register_evaluables(::Type{JPhiSpinMC}, eval::AbstractEvaluator, params::AbstractDict)
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
    Serialization.serialize(s, mc.spins)
    Serialization.serialize(s, mc.energy)
    Serialization.serialize(s, mc.xml_path)
    Serialization.serialize(s, mc.repeat)
    Serialization.serialize(s, mc.spin_theta_max)
    Serialization.serialize(s, mc.renorm_every)
    Serialization.serialize(s, mc.sweep_count)
    Serialization.serialize(s, mc.enabled_bodies)
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, ::Type{JPhiSpinMC})
    T            = Serialization.deserialize(s)::Float64
    spins        = Serialization.deserialize(s)::Matrix{Float64}
    energy       = Serialization.deserialize(s)::Float64
    xml_path     = Serialization.deserialize(s)::String
    repeat       = Serialization.deserialize(s)::NTuple{3,Int}
    spin_theta_max = Serialization.deserialize(s)
    renorm_every = Serialization.deserialize(s)::Int
    sweep_count  = Serialization.deserialize(s)::Int
    enabled_bodies = Serialization.deserialize(s)

    # Use the process-local cache (populated at startup via _mpi_build_ham_and_cache).
    # On rank 0, this is called 32 times during Carlo's PT checkpoint gather;
    # the cache ensures ham/local_cache are NOT rebuilt 32 times.
    ham, cache = _mpi_build_ham_and_cache(xml_path, repeat)

    active_body_indices = _enabled_bodies_to_active_indices(enabled_bodies, cache.body_list)
    derived = _get_or_build_derived(xml_path, repeat, active_body_indices, cache, ham.n_atoms)
    zlm_cache = _alloc_zlm_cache(ham.n_atoms, derived.max_l)
    zlm_row_buf = Vector{Float64}(undef, (derived.max_l + 1)^2)

    mc = JPhiSpinMC(
        T, ham, spins, energy, cache,
        derived.active_body_indices, derived.active_instance_indices, derived.related_instances_by_atom,
        derived.max_l, zlm_cache, zlm_row_buf,
        Vector{Int}(undef, derived.max_sites), Vector{Int}(undef, derived.max_sites),
        spin_theta_max, renorm_every, sweep_count,
        xml_path, repeat, enabled_bodies,
    )
    _rebuild_zlm_cache!(mc)
    return mc
end

function Carlo.write_checkpoint(mc::JPhiSpinMC, out::HDF5.Group)
    out["spins"] = mc.spins
    out["energy"] = mc.energy
    return nothing
end

function Carlo.write_checkpoint(
    mc::JPhiSpinMC,
    out::Union{HDF5.Group,Nothing},
    comm::MPI.Comm,
)
    all_spins = MPI.gather(mc.spins, comm)
    all_energies = MPI.Gather(mc.energy, comm)

    if MPI.Comm_rank(comm) == 0
        out["spins"] = cat(all_spins...; dims = 3)
        out["energy"] = all_energies
    end
    return nothing
end

function Carlo.read_checkpoint!(mc::JPhiSpinMC, in::HDF5.Group)
    mc.spins .= read(in, "spins")
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
        spins_all = read(in, "spins")
        energies = vec(read(in, "energy"))
        spins_per_rank = [copy(s) for s in eachslice(spins_all; dims = 3)]
    else
        spins_per_rank = nothing
        energies = nothing
    end

    mc.spins .= MPI.scatter(spins_per_rank, comm)
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
