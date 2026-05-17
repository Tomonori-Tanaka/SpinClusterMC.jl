"""
Independent parser for Magesty `jphi.xml` used by the reference Simple
implementation. Produces plain data structs (no coupling to `JPhiMagestyCarlo`)
that the M3 type layer turns into `SpinClusterHamiltonian` / `ClusterInstance`.

The XML grammar matches the existing fixtures under `test/<lattice>/jphi.xml`.
See `docs/specs/260512-simple-impl/design.md` for the schema. Field semantics follow Magesty's `CoupledBasis_with_coefficient`
(`Magesty/src/CoupledBases.jl`).
"""

using EzXML: readxml, findfirst, findall, nodecontent
using LinearAlgebra: norm

"""
    SALCBasisData

One `<basis>` element inside a `<SALC>`. Each basis represents a single
symmetry-related N-site cluster term, where N equals the parent SALC's `body`
attribute. The SALC groups symmetry-equivalent clusters that share a single
user-facing coupling constant J_s (`<JPhi>`), and `multiplicity` records how
many sites the symmetry operation generates from this basis.

# Fields

- `multiplicity`: Number of symmetry-equivalent clusters represented by this
  `<basis>` (`<basis multiplicity="...">`). Acts as a prefactor m_b in the
  energy sum E = Σ_s J_s · (4π)^(N/2) · Σ_b m_b · ⟨tensor contraction⟩.
- `atoms`: Base-cell atom indices, one per site (length N). Order matches
  `ls` and the leading dimensions of the SALC tensor. These are the *seeds*
  for translation: each `(atoms, weights)` pair is replicated on every
  translation column of `map_sym` (see `SystemData`).
- `ls`: Orbital angular momentum per site (length N). Defines the per-site
  multiplet dimension `2*ls[i]+1` and which spherical harmonic Z_{ls[i]}^{m}
  is evaluated at site i. The physical "rank" of the spin operator at each
  site, in SCE terms.
- `Lseq`: Intermediate-coupling labels along the left-coupling tree
  `(((l1 ⊗ l2 → L12) ⊗ l3 → L123) ⊗ … → Lf)`. Length is `max(0, N-2)` —
  L12 through L_{1..N-1}, with the final coupling to Lf fixed by the parent
  SALC. Empty for N=2. Magesty's `enumerate_paths_left_all(ls)` produces
  the same enumeration. Selects which Clebsch-Gordan tensor to use among
  the multiple valid paths when N ≥ 3.
- `weights`: The (2*Lf+1) numeric entries of the `<basis>` text content,
  ordered by Mf = -Lf, -Lf+1, …, +Lf. These are the symmetry-adapted
  weights w_{s,b}[Mf] that combine with the CG tensor T_real[m₁..m_N; Mf]
  to give the cluster's tesseral SALC. Distinct from the SALC's user-facing
  J_s: J_s is one scalar per SALC, weights[mf] is a (2Lf+1)-vector per basis.

# Invariants (enforced by `_parse_basis_node`)

- `length(atoms) == length(ls) == body`
- `length(Lseq) == max(0, body - 2)`
- `length(weights) == 2*Lf + 1`
"""
struct SALCBasisData
    multiplicity::Int
    atoms::Vector{Int}
    ls::Vector{Int}
    Lseq::Vector{Int}
    weights::Vector{Float64}
end

"""
    SALCData

One `<SALC>` element: a symmetry-adapted linear combination of N-site clusters,
all sharing the same user-facing coupling J_s = `jphi[index]`. SALCs are the
unit of physical coupling in the spin-cluster expansion.

# Fields

- `body`: N-body order of the cluster (= number of sites in each basis).
  `body=2` is pair (e.g., Heisenberg-like), `body=3` is triplet, etc.
  Equals the `<SALC body="...">` attribute.
- `Lf`: Final total angular momentum of the SALC (`<SALC Lf="...">`).
  Lf=0 is isotropic (scalar coupling, single weight). Lf≥1 carries
  directional content: anisotropy, magnetostriction, etc. The number of
  weights per basis is `2*Lf + 1`.
- `bases`: All `<basis>` elements under this SALC, ordered by their `index`
  attribute. Each basis is a distinct symmetry-equivalent cluster; the SALC
  energy contribution is the (multiplicity-weighted) sum over these bases.
"""
struct SALCData
    body::Int
    Lf::Int
    bases::Vector{SALCBasisData}
end

"""
    SystemData

Crystal/system block of `jphi.xml`: lattice geometry plus the discrete
translation table used to replicate cluster terms across the cell.

# Fields

- `n_atoms`: Number of atoms in the base (XML) cell.
- `lattice`: 3×3 matrix whose columns are the lattice vectors `[a1 a2 a3]` (Å).
  Cartesian position of atom i is `lattice * pos_frac[:, i]`.
- `periodicity`: Periodicity flags `(p1, p2, p3)` per lattice direction
  (1 = periodic, 0 = open). Read but not consumed by the Simple energy code,
  which assumes full periodicity; kept for fidelity to the input.
- `pos_frac`: 3×n_atoms fractional positions (columns are atoms, rows are
  fractional coordinates along `lattice`).
- `n_trans`: Number of translations in the symmetry table (= `n_atoms` for
  standard primitive-symmetry crystals — one translation per base atom).
- `map_sym`: `n_atoms × n_trans` translation table. `map_sym[a, t]` is the
  destination atom index when base-cell atom `a` is translated by the `t`-th
  lattice translation. The XML stores only the `atom="1"` rows; the rest are
  inferred by minimum-image matching from atom 1's displacement.
"""
struct SystemData
    n_atoms::Int
    lattice::Matrix{Float64}
    periodicity::NTuple{3, Int}
    pos_frac::Matrix{Float64}
    n_trans::Int
    map_sym::Matrix{Int}
end

"""
    JPhiXMLData

Aggregated parse result of one `jphi.xml`.

# Fields

- `system`: System geometry and translation table (see `SystemData`).
- `salcs`: SALC list in `index` order (`salcs[s]` carries the bases for SALC s).
- `jphi`: User-facing coupling constants in eV, one per SALC (`jphi[s] = J_s`).
  The XML's `<ReferenceEnergy>` (`j0`) is deliberately *not* read — this package
  is for MC sampling where only ΔE matters, so the constant offset is irrelevant
  (see `CLAUDE.md` § "j0").
"""
struct JPhiXMLData
    system::SystemData
    salcs::Vector{SALCData}
    jphi::Vector{Float64}
end

function _parse_int_list(s::AbstractString)::Vector{Int}
    return parse.(Int, split(s))
end

function _parse_float_list(s::AbstractString)::Vector{Float64}
    return parse.(Float64, split(s))
end

function _parse_vec3(s::AbstractString)::Vector{Float64}
    v = _parse_float_list(s)
    length(v) == 3 || throw(ArgumentError("expected 3 floats, got $(repr(s))"))
    return v
end

function _min_image_frac(v::AbstractVector{<:Real})::Vector{Float64}
    w = collect(Float64, v)
    @inbounds for i in eachindex(w)
        w[i] -= round(w[i])
    end
    return w
end

function _frac_periodic_dist(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})::Float64
    return norm(_min_image_frac(a .- b))
end

# Fill missing entries of translation column `t` of `map_sym` by minimum-image
# matching from atom 1's displacement. The XML typically lists only
# `<map atom="1">` rows; other atoms' destinations are inferred under the
# assumption that all atoms translate rigidly by the same lattice vector.
function _infer_atom_map_from_atom1!(
        map_sym::Matrix{Int},
        pos_frac::AbstractMatrix{Float64},
        t::Int,
        n_atoms::Int
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

"""
    parse_system_xml(xml_path) -> SystemData

Parse the `<System>` block: number of atoms, lattice, periodicity, fractional
positions, and the translation map. Missing `<map>` entries for non-leading
atoms are inferred from atom 1's translation by periodic nearest-image match.
"""
function parse_system_xml(xml_path::AbstractString)::SystemData
    doc = readxml(xml_path)
    system_node = findfirst("//System", doc)
    isnothing(system_node) && throw(ArgumentError("no //System in $xml_path"))

    natoms_node = findfirst("NumberOfAtoms", system_node)
    isnothing(natoms_node) && throw(ArgumentError("no NumberOfAtoms in $xml_path"))
    n_atoms = parse(Int, nodecontent(natoms_node))

    lat_node = findfirst("LatticeVector", system_node)
    isnothing(lat_node) && throw(ArgumentError("no LatticeVector in $xml_path"))
    a1_node = findfirst("a1", lat_node)
    a2_node = findfirst("a2", lat_node)
    a3_node = findfirst("a3", lat_node)
    (isnothing(a1_node) || isnothing(a2_node) || isnothing(a3_node)) &&
        throw(ArgumentError("LatticeVector missing a1/a2/a3 in $xml_path"))
    a1 = _parse_vec3(nodecontent(a1_node))
    a2 = _parse_vec3(nodecontent(a2_node))
    a3 = _parse_vec3(nodecontent(a3_node))
    lattice = hcat(a1, a2, a3)

    per_node = findfirst("Periodicity", system_node)
    isnothing(per_node) && throw(ArgumentError("no Periodicity in $xml_path"))
    per_ints = _parse_int_list(nodecontent(per_node))
    length(per_ints) == 3 || throw(ArgumentError("Periodicity must have 3 integers"))
    periodicity = (per_ints[1], per_ints[2], per_ints[3])

    pos_block = findfirst("Positions", system_node)
    isnothing(pos_block) && throw(ArgumentError("no Positions in $xml_path"))
    pos_frac = zeros(3, n_atoms)
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
        (1 ≤ t ≤ n_trans && 1 ≤ a ≤ n_atoms) ||
            throw(ArgumentError("invalid map trans=$t atom=$a"))
        1 ≤ dest ≤ n_atoms || throw(
            ArgumentError("map trans=$t atom=$a dest=$dest out of range 1:$n_atoms"),
        )
        map_sym[a, t] = dest
    end
    for t in 1:n_trans
        if any(iszero, @view map_sym[:, t])
            _infer_atom_map_from_atom1!(map_sym, pos_frac, t, n_atoms)
        end
    end

    return SystemData(n_atoms, lattice, periodicity, pos_frac, n_trans, map_sym)
end

function _parse_basis_node(basis_node, body::Int, Lf::Int)::SALCBasisData
    multiplicity = parse(Int, basis_node["multiplicity"])
    atoms = _parse_int_list(basis_node["atoms"])
    ls = _parse_int_list(basis_node["ls"])
    lseq_attr = haskey(basis_node, "Lseq") ? basis_node["Lseq"] : ""
    Lseq = isempty(strip(lseq_attr)) ? Int[] : _parse_int_list(lseq_attr)
    weights = _parse_float_list(nodecontent(basis_node))

    length(atoms) == body ||
        throw(ArgumentError("basis atoms length $(length(atoms)) != body $body"))
    length(ls) == body ||
        throw(ArgumentError("basis ls length $(length(ls)) != body $body"))
    expected_lseq = max(0, body - 2)
    length(Lseq) == expected_lseq || throw(
        ArgumentError(
        "basis Lseq length $(length(Lseq)) != expected $expected_lseq (body=$body)",
    ),
    )
    expected_weights = 2 * Lf + 1
    length(weights) == expected_weights || throw(
        ArgumentError(
        "basis weights length $(length(weights)) != expected $expected_weights (Lf=$Lf)",
    ),
    )

    return SALCBasisData(multiplicity, atoms, ls, Lseq, weights)
end

"""
    parse_salc_list(xml_path) -> Vector{SALCData}

Parse the `<SCEBasis>` block: one `SALCData` per `<SALC>` element, with all
`<basis>` children attached. SALCs are returned in the order their `index`
attribute defines (validated as contiguous `1..num_salc`).
"""
function parse_salc_list(xml_path::AbstractString)::Vector{SALCData}
    doc = readxml(xml_path)
    basisset_node = findfirst("//SCEBasis", doc)
    isnothing(basisset_node) && throw(ArgumentError("no //SCEBasis in $xml_path"))
    num_salc_attr = haskey(basisset_node, "num_salc") ?
                    parse(Int, basisset_node["num_salc"]) : -1

    salc_nodes = findall("SALC", basisset_node)
    num_salc_attr == -1 || length(salc_nodes) == num_salc_attr ||
        throw(ArgumentError("SCEBasis num_salc=$num_salc_attr but found $(length(salc_nodes)) SALC entries"))

    indexed = Tuple{Int, SALCData}[]
    for sn in salc_nodes
        idx = parse(Int, sn["index"])
        body = parse(Int, sn["body"])
        Lf = parse(Int, sn["Lf"])
        num_basis_attr = haskey(sn, "num_basis") ? parse(Int, sn["num_basis"]) : -1
        basis_nodes = findall("basis", sn)
        num_basis_attr == -1 || length(basis_nodes) == num_basis_attr ||
            throw(
                ArgumentError(
                "SALC index=$idx num_basis=$num_basis_attr but found $(length(basis_nodes)) basis entries",
            ),
            )
        bases = [_parse_basis_node(bn, body, Lf) for bn in basis_nodes]
        push!(indexed, (idx, SALCData(body, Lf, bases)))
    end
    sort!(indexed; by = first)
    for (k, (idx, _)) in enumerate(indexed)
        idx == k ||
            throw(ArgumentError("SALC index must be 1..n without gaps; got $idx at position $k"))
    end
    return [s for (_, s) in indexed]
end

"""
    parse_jphi_coefficients(xml_path) -> Vector{Float64}

Parse the `<JPhi>` block as a dense vector indexed by SALC index. The
`<ReferenceEnergy>` (j0) entry is deliberately ignored — see `CLAUDE.md`.
"""
function parse_jphi_coefficients(xml_path::AbstractString)::Vector{Float64}
    doc = readxml(xml_path)
    jnode = findfirst("//JPhi", doc)
    isnothing(jnode) && throw(ArgumentError("no //JPhi in $xml_path"))
    pairs = Tuple{Int, Float64}[]
    for el in findall("jphi", jnode)
        push!(pairs, (parse(Int, el["salc_index"]), parse(Float64, nodecontent(el))))
    end
    sort!(pairs; by = first)
    for (i, (si, _)) in enumerate(pairs)
        si == i || throw(
            ArgumentError("jphi salc_index must be 1..n without gaps; got index $si at position $i"),
        )
    end
    return last.(pairs)
end

"""
    parse_jphi_xml(xml_path) -> JPhiXMLData

Read the full `jphi.xml` (system, SALCs, jphi coefficients) and cross-check that
the SALC count matches the jphi count and that no `<basis atoms>` references an
atom index outside `1:n_atoms`.
"""
function parse_jphi_xml(xml_path::AbstractString)::JPhiXMLData
    system = parse_system_xml(xml_path)
    salcs = parse_salc_list(xml_path)
    jphi = parse_jphi_coefficients(xml_path)
    length(salcs) == length(jphi) || throw(
        ArgumentError("SALC count $(length(salcs)) != jphi count $(length(jphi))"),
    )
    for (s, salc) in enumerate(salcs)
        for (b, basis) in enumerate(salc.bases)
            for a in basis.atoms
                1 ≤ a ≤ system.n_atoms || throw(
                    ArgumentError(
                    "SALC $s basis $b atom $a out of range 1:$(system.n_atoms)",
                ),
                )
            end
        end
    end
    return JPhiXMLData(system, salcs, jphi)
end
