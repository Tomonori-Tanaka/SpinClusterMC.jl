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

using Magesty.Basis: CoupledBasis_with_coefficient
using Magesty.MySphericalHarmonics: Zₗₘ
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

function _parse_vec3(s::AbstractString)
    p = parse.(Float64, split(s))
    length(p) == 3 || throw(ArgumentError("expected 3 floats, got $(repr(s))"))
    return p
end

function _min_image_frac(v::AbstractVector{<:Real})
    w = collect(Float64, v)
    @inbounds for i in eachindex(w)
        w[i] -= round(w[i])
    end
    return w
end

function _frac_periodic_dist(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    return norm(_min_image_frac(a .- b))
end

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

struct ClusterInstance
    atoms::Vector{Int}
    cbc::CoupledBasis_with_coefficient
    prefactor::Float64
    # Precomputed from cbc.ls to avoid per-call allocations in the hot sweep path.
    dims::Vector{Int}    # dims[k] = 2*cbc.ls[k]+1
    strides::Vector{Int} # tensor strides, length N+1; strides[k] = prod(dims[1:k-1])
end

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
                    translated_atoms = Int[
                        supercell_atom_index(ba, ti, tj, tk, base_n_atoms, repeat) for
                        ba in translated_base
                    ]
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
                            sh_values[site_idx][m_idx] = Zₗₘ(l, m, u)
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

function coupled_cluster_energy(
    cbc::CoupledBasis_with_coefficient,
    spin_directions::AbstractMatrix{<:Real},
    map_sym::AbstractMatrix{Int},
)::Float64
    return coupled_cluster_energy(
        cbc,
        spin_directions,
        map_sym;
        repeat = (1, 1, 1),
        base_n_atoms = size(map_sym, 1),
    )
end

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
            sh_values[site_idx][m_idx] = Zₗₘ(l, m, u)
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

function _build_cluster_instances(h::SCEHamiltonian)::Vector{ClusterInstance}
    instances = ClusterInstance[]
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
                            translated_atoms = Int[
                                supercell_atom_index(
                                    ba,
                                    ti,
                                    tj,
                                    tk,
                                    h.base_n_atoms,
                                    h.repeat,
                                ) for ba in translated_base
                            ]
                            atoms_sorted = sort(translated_atoms)
                            pair = (atoms_sorted, cbc.ls)
                            pair in searched_pairs && continue
                            push!(searched_pairs, pair)
                            inst_dims = [2 * l + 1 for l in cbc.ls]
                            inst_strides = _compute_instance_strides(cbc.ls)
                            push!(
                                instances,
                                ClusterInstance(
                                    translated_atoms,
                                    cbc,
                                    js * cbc.multiplicity * scaling,
                                    inst_dims,
                                    inst_strides,
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

function sce_energy(h::SCEHamiltonian, spin_directions::AbstractMatrix{<:Real})::Float64
    size(spin_directions, 2) == h.n_atoms ||
        throw(ArgumentError("spin matrix has $(size(spin_directions,2)) columns, expected $(h.n_atoms)"))
    E = h.j0
    for (s, group) in enumerate(h.salc_list)
        js = h.jphi[s]
        for cbc in group
            E += js * coupled_cluster_energy(
                cbc,
                spin_directions,
                h.map_sym;
                repeat = h.repeat,
                base_n_atoms = h.base_n_atoms,
            )
        end
    end
    return E
end

# --- Carlo.AbstractMC ---

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
end

@inline interaction_partners(mc::JPhiSpinMC, atom::Int)::Vector{Int} =
    interaction_partners(mc.local_cache, atom)

@inline interaction_partners_by_body(mc::JPhiSpinMC, atom::Int)::Dict{Int, Vector{Int}} =
    interaction_partners_by_body(mc.local_cache, atom)

function JPhiSpinMC(params::AbstractDict)
    xml = params[:xml_path]
    rep = _parse_repeat_param(params)
    ham = load_sce_hamiltonian(xml; repeat = rep)
    T = Float64(params[:T])
    cache = build_local_energy_cache(ham)
    active_body_indices = _parse_enabled_body_indices(params, cache.body_list)
    active_instance_indices = _active_instance_indices(cache, active_body_indices)
    related_instances_by_atom = _build_related_instances_by_atom(cache, active_body_indices, ham.n_atoms)
    max_l = _max_l_in_instances(cache.instances)
    zlm_cache = _alloc_zlm_cache(ham.n_atoms, max_l)
    max_sites = _max_sites_in_instances(cache.instances)
    other_sites_work = Vector{Int}(undef, max_sites)
    cart_idx_work = Vector{Int}(undef, max_sites)
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
    return JPhiSpinMC(
        T,
        ham,
        zeros(3, ham.n_atoms),
        0.0,
        cache,
        active_body_indices,
        active_instance_indices,
        related_instances_by_atom,
        max_l,
        zlm_cache,
        other_sites_work,
        cart_idx_work,
        spin_theta_max,
        renorm_every,
        0,
    )
end

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

function _max_l_in_instances(instances::Vector{ClusterInstance})::Int
    m = 0
    for inst in instances
        for l in inst.cbc.ls
            m = max(m, l)
        end
    end
    return m
end

@inline _zlm_col(l::Int, m_idx::Int)::Int = l * l + m_idx

function _alloc_zlm_cache(n_atoms::Int, max_l::Int)::Matrix{Float64}
    # sum_{l=0}^{L} (2l+1) = (L+1)^2
    return zeros(Float64, n_atoms, (max_l + 1)^2)
end

function _update_atom_zlm_cache!(
    zlm_cache::Matrix{Float64},
    atom::Int,
    u::AbstractVector{<:Real},
    max_l::Int,
)
    @inbounds for l in 0:max_l
        @simd for m_idx in 1:(2 * l + 1)
            m = m_idx - l - 1
            zlm_cache[atom, _zlm_col(l, m_idx)] = Zₗₘ(l, m, u)
        end
    end
    return nothing
end

function _rebuild_zlm_cache!(mc::JPhiSpinMC)
    @inbounds for atom in 1:mc.ham.n_atoms
        _update_atom_zlm_cache!(mc.zlm_cache, atom, @view(mc.spins[:, atom]), mc.max_l)
    end
    return nothing
end

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

@inline function _tensor_contract_instance_cached(
    inst::ClusterInstance,
    zlm_cache::Matrix{Float64},
)::Float64
    cbc = inst.cbc
    N = length(inst.atoms)
    tensor_result = 0.0
    Mf_size = size(cbc.coeff_tensor, N + 1)

    for mf_idx in 1:Mf_size
        mf_contribution = 0.0
        for site_idx_tuple in CartesianIndices(Tuple(inst.dims))
            product = 1.0
            @inbounds for (site_idx, m_idx) in enumerate(site_idx_tuple.I)
                atom = inst.atoms[site_idx]
                l = cbc.ls[site_idx]
                product *= zlm_cache[atom, _zlm_col(l, m_idx)]
            end
            tensor_idx = (site_idx_tuple.I..., mf_idx)
            mf_contribution += cbc.coeff_tensor[tensor_idx...] * product
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

    Mf_size = size(cbc.coeff_tensor, N + 1)
    coeff_tensor = cbc.coeff_tensor
    tensor_result = 0.0

    if n_other == 0
        @inbounds for mf_idx in 1:Mf_size
            mf_contribution = 0.0
            base_mf = 1 + (mf_idx - 1) * strides[N + 1]
            @simd for mchg_idx in 1:dims_sitepos
                mf_contribution +=
                    coeff_tensor[base_mf + (mchg_idx - 1) * stride_changed] *
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
                    coeff_tensor[base_without_changed + (mchg_idx - 1) * stride_changed] *
                    zlm_cache[changed_atom, changed_col_base + mchg_idx]
            end
            mf_contribution += product_other * inner
        end
        tensor_result += cbc.coefficient[mf_idx] * mf_contribution
    end

    return tensor_result
end

function Carlo.init!(mc::JPhiSpinMC, ctx::MCContext, params::AbstractDict)
    n = mc.ham.n_atoms
    for i in 1:n
        sx, sy, sz = _rand_unit_spin(ctx.rng)
        mc.spins[1, i] = sx
        mc.spins[2, i] = sy
        mc.spins[3, i] = sz
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
            _update_atom_zlm_cache!(mc.zlm_cache, i, sold, mc.max_l)
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
    measure!(ctx, :Energy, mc.energy / n)
    measure!(ctx, :Energy2, (mc.energy / n)^2)
    measure!(ctx, :Magnetization2, mag2)
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
    n = load_sce_hamiltonian(params[:xml_path]; repeat = _parse_repeat_param(params)).n_atoms
    evaluate!(eval, :SpecificHeat, (:Energy2, :Energy)) do e2, e
        return n * (e2 - e^2) / T^2
    end
    return nothing
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

end # module
