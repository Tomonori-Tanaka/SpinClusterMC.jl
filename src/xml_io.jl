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

    natoms_node = findfirst("NumberOfAtoms", system_node)
    isnothing(natoms_node) && throw(ArgumentError("no NumberOfAtoms in $xml_path"))
    n_atoms = parse(Int, nodecontent(natoms_node))

    lat_node = findfirst("LatticeVector", system_node)
    isnothing(lat_node) && throw(ArgumentError("no LatticeVector in $xml_path"))
    a1_node = findfirst("a1", lat_node); isnothing(a1_node) && throw(ArgumentError("no a1 in $xml_path"))
    a2_node = findfirst("a2", lat_node); isnothing(a2_node) && throw(ArgumentError("no a2 in $xml_path"))
    a3_node = findfirst("a3", lat_node); isnothing(a3_node) && throw(ArgumentError("no a3 in $xml_path"))
    a1 = _parse_vec3(nodecontent(a1_node))
    a2 = _parse_vec3(nodecontent(a2_node))
    a3 = _parse_vec3(nodecontent(a3_node))
    lattice = hcat(a1, a2, a3)

    per_el = findfirst("Periodicity", system_node)
    isnothing(per_el) && throw(ArgumentError("no Periodicity in $xml_path"))
    per_ints = parse.(Int, split(nodecontent(per_el)))
    length(per_ints) == 3 || throw(ArgumentError("Periodicity must have 3 integers"))
    per = (per_ints[1], per_ints[2], per_ints[3])

    pos_frac = zeros(3, n_atoms)
    pos_block = findfirst("Positions", system_node)
    isnothing(pos_block) && throw(ArgumentError("no Positions in $xml_path"))
    for p in findall("pos", pos_block)
        ia = parse(Int, p["atom_index"])
        pos_frac[:, ia] .= _parse_vec3(nodecontent(p))
    end

    sym_node = findfirst("Symmetry", system_node)
    isnothing(sym_node) && throw(ArgumentError("no Symmetry in $xml_path"))
    ntrans_node = findfirst("NumberOfTranslations", sym_node)
    isnothing(ntrans_node) && throw(ArgumentError("no NumberOfTranslations in $xml_path"))
    n_trans = parse(Int, nodecontent(ntrans_node))
    trans_block = findfirst("Translations", sym_node)
    isnothing(trans_block) && throw(ArgumentError("no Translations in $xml_path"))
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

function read_jphi_coefficients(xml_path::AbstractString)::Vector{Float64}
    doc = readxml(xml_path)
    jnode = findfirst("//JPhi", doc)
    isnothing(jnode) && throw(ArgumentError("no //JPhi in $xml_path"))
    pairs = Tuple{Int, Float64}[]
    for el in findall("jphi", jnode)
        push!(pairs, (parse(Int, el["salc_index"]), parse(Float64, nodecontent(el))))
    end
    sort!(pairs)
    for (i, (si, _)) in enumerate(pairs)
        si == i || throw(ArgumentError("jphi salc_index must be 1..n without gaps; got index $si at position $i"))
    end
    return last.(pairs)
end
