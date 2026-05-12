"""
Core types for the Simple SCE implementation: `ClusterInstance`, `CGTable`,
`SpinClusterHamiltonian`. Together they describe every quantity the energy
code (`energy.jl`) needs at evaluation time.

The separation mirrors `docs/specs/260512-simple-impl/design.md`:

- `ClusterInstance` carries the per-cluster data that varies between symmetry
  copies (atom indices, weights, coupling J).
- `CGTable` carries the per-`(ls, Lf, Lseq)` tesseral Clebsch-Gordan tensor
  that is shared across every cluster with the same coupling pattern; built
  once when the Hamiltonian is loaded.
- `SpinClusterHamiltonian` is the bundle the MC engine holds.
"""

"""
    ClusterInstance

One translated, symmetry-equivalent realization of an N-site cluster term.
The energy contribution of this instance is

    E_inst = J * (4π)^(N/2) * multiplicity *
             Σ_Mf salc_weights[Mf] *
                  Σ_{m1..mN} T_real[m1, …, mN; Mf] *
                            Π_i Z_{ls[i]}^{m_i}(S_{atoms[i]})

where `T_real` is fetched from `CGTable` by `(ls, Lf, Lseq)`.

# Fields

- `atoms`: Supercell atom indices for the N sites of this instance, in the
  same order as `ls`. For `repeat == (1, 1, 1)` these equal base-cell indices.
- `ls`: Orbital angular momentum per site, length `N = body`. Mirrors
  `SALCBasisData.ls`.
- `Lf`: Final total angular momentum of the parent SALC. 0 = isotropic.
- `Lseq`: Intermediate-coupling path, length `max(0, N - 2)`. Mirrors
  `SALCBasisData.Lseq`.
- `salc_weights`: Per-Mf weights from the parent `<basis>` text content,
  length `2 * Lf + 1`. The symmetry-adapted weights `w_{s,b}[Mf]`.
- `J`: User-facing coupling constant for the parent SALC, in eV
  (`<JPhi> jphi[s]`).
- `multiplicity`: Symmetry-multiplicity prefactor `m_b` from
  `<basis multiplicity="...">`.

# Invariants (length checks)

The inner constructor enforces the structural invariants
`length(atoms) == length(ls)`, `length(Lseq) == max(0, N - 2)`, and
`length(salc_weights) == 2 * Lf + 1`. Semantic preconditions on inputs
(`Lf ≥ 0`, `ls[i] ≥ 0`, `multiplicity ≥ 1`) are already enforced upstream by
`parse_jphi_xml` and are not re-checked here on the construction-hot path.
"""
struct ClusterInstance
    atoms::Vector{Int}
    ls::Vector{Int}
    Lf::Int
    Lseq::Vector{Int}
    salc_weights::Vector{Float64}
    J::Float64
    multiplicity::Int

    function ClusterInstance(
            atoms::AbstractVector{<:Integer},
            ls::AbstractVector{<:Integer},
            Lf::Integer,
            Lseq::AbstractVector{<:Integer},
            salc_weights::AbstractVector{<:Real},
            J::Real,
            multiplicity::Integer
    )
        N = length(ls)
        length(atoms) == N || throw(
            ArgumentError("ClusterInstance: length(atoms)=$(length(atoms)) != length(ls)=$N")
        )
        expected_lseq = max(0, N - 2)
        length(Lseq) == expected_lseq || throw(
            ArgumentError(
            "ClusterInstance: length(Lseq)=$(length(Lseq)) != expected $expected_lseq (N=$N)"
        )
        )
        expected_weights = 2 * Lf + 1
        length(salc_weights) == expected_weights || throw(
            ArgumentError(
            "ClusterInstance: length(salc_weights)=$(length(salc_weights)) != 2*Lf+1=$expected_weights"
        )
        )
        return new(
            collect(Int, atoms),
            collect(Int, ls),
            Int(Lf),
            collect(Int, Lseq),
            collect(Float64, salc_weights),
            Float64(J),
            Int(multiplicity)
        )
    end
end

"""
    CGTable

Read-only lookup of tesseral (real) Clebsch-Gordan tensors, keyed by
`(ls, Lf, Lseq)`. One entry per coupling pattern present in the Hamiltonian;
shared across every `ClusterInstance` that has the same `(ls, Lf, Lseq)`.

# Key

`Tuple{Vector{Int}, Int, Vector{Int}}` — `(ls, Lf, Lseq)` where
`length(Lseq) == max(0, length(ls) - 2)` (left-coupling tree, intermediate
L's only; the final coupling to Lf is part of the key itself).

# Value

`Array{Float64, N+1}` for `N = length(ls)`, with shape
`(2*ls[1]+1, …, 2*ls[N]+1, 2*Lf+1)`. The trailing axis enumerates
`Mf = -Lf, …, +Lf`; the leading axes enumerate `m_i = -ls[i], …, +ls[i]`.

Entries are the *tesseral* CG tensors `T_real`, produced by
`Magesty.AngularMomentumCoupling.build_all_real_bases` (Racah CG composed
with the tesseral phase compensation that converts complex `Y_l^m` to real
`Z_l^m`). Mathematically the same quantity that the optimized side stores
inside `cbc.coeff_tensor`.
"""
struct CGTable
    entries::Dict{Tuple{Vector{Int}, Int, Vector{Int}}, Array{Float64}}

    function CGTable(entries::AbstractDict)
        d = Dict{Tuple{Vector{Int}, Int, Vector{Int}}, Array{Float64}}()
        for (k, v) in entries
            length(k) == 3 ||
                throw(ArgumentError("CGTable key must be (ls, Lf, Lseq); got $k"))
            ls_raw, Lf_raw, Lseq_raw = k
            ls = collect(Int, ls_raw)
            Lf = Int(Lf_raw)
            Lseq = collect(Int, Lseq_raw)
            N = length(ls)
            N ≥ 1 || throw(ArgumentError("CGTable: ls must be non-empty"))
            expected_lseq = max(0, N - 2)
            length(Lseq) == expected_lseq || throw(
                ArgumentError(
                "CGTable: length(Lseq)=$(length(Lseq)) != expected $expected_lseq for ls=$ls"
            )
            )
            tensor = Array{Float64}(v)
            ndims(tensor) == N + 1 || throw(
                ArgumentError(
                "CGTable tensor for ls=$ls has ndims $(ndims(tensor)); expected $(N+1)"
            )
            )
            expected_shape = ntuple(i -> i ≤ N ? 2 * ls[i] + 1 : 2 * Lf + 1, N + 1)
            size(tensor) == expected_shape || throw(
                ArgumentError(
                "CGTable tensor shape $(size(tensor)) != expected $expected_shape for (ls=$ls, Lf=$Lf)"
            )
            )
            d[(ls, Lf, Lseq)] = tensor
        end
        return new(d)
    end
end

Base.length(t::CGTable) = length(t.entries)
Base.haskey(t::CGTable, key) = haskey(t.entries, key)
Base.getindex(t::CGTable, key) = t.entries[key]
Base.keys(t::CGTable) = keys(t.entries)

"""
    SpinClusterHamiltonian

Full description of the SCE Hamiltonian held by the Monte Carlo engine. Built
from a Magesty `jphi.xml` and a tile factor `repeat`.

# Fields

- `n_atoms`: Total atom count of the supercell (`base_n_atoms * prod(repeat)`).
- `base_n_atoms`: Atom count of the XML (base) cell.
- `repeat`: Tile factors along the three lattice directions `(n1, n2, n3)`.
- `lattice`: 3×3 supercell lattice matrix; columns are `n_k * a_k`.
- `pos_frac`: 3×n_atoms fractional positions in the supercell.
- `instances`: All translated cluster terms. Every base-cell `<basis>` is
  replicated once per translation column of `map_sym` and once per tile, with
  atom indices rewritten as supercell indices.
- `cg_table`: Tesseral CG tensors keyed by `(ls, Lf, Lseq)`, looked up at
  energy-evaluation time.

# Construction

`SpinClusterHamiltonian(xml_path; repeat=(1,1,1))` runs the parser, the
supercell geometry build, the instance generation, and the CG-table build in
that order.
"""
struct SpinClusterHamiltonian
    n_atoms::Int
    base_n_atoms::Int
    repeat::NTuple{3, Int}
    lattice::Matrix{Float64}
    pos_frac::Matrix{Float64}
    instances::Vector{ClusterInstance}
    cg_table::CGTable
end

@inline function _supercell_atom_index(
        base_atom::Int,
        ti::Integer,
        tj::Integer,
        tk::Integer,
        base_n::Int,
        repeat::NTuple{3, Int}
)::Int
    n1, n2, n3 = repeat
    1 ≤ base_atom ≤ base_n ||
        throw(ArgumentError("base_atom=$base_atom not in 1:$base_n"))
    (0 ≤ ti < n1 && 0 ≤ tj < n2 && 0 ≤ tk < n3) ||
        throw(ArgumentError("tile ($ti,$tj,$tk) out of range for repeat=$repeat"))
    return base_atom + base_n * (ti + n1 * tj + n1 * n2 * tk)
end

# Build supercell lattice and wrapped fractional positions by tiling the base
# cell. Independent of the optimized-side `_build_supercell_geometry` so the
# Simple file stays self-contained.
function _build_supercell_geometry(
        lattice::AbstractMatrix{<:Real},
        pos_base_frac::AbstractMatrix{<:Real},
        base_n::Int,
        repeat::NTuple{3, Int}
)
    n1, n2, n3 = repeat
    a1 = lattice[:, 1]
    a2 = lattice[:, 2]
    a3 = lattice[:, 3]
    lattice_super = hcat(n1 .* a1, n2 .* a2, n3 .* a3)
    n_tot = base_n * n1 * n2 * n3
    pos_super = zeros(3, n_tot)
    for tk in 0:(n3 - 1), tj in 0:(n2 - 1), ti in 0:(n1 - 1), b in 1:base_n
        ia = _supercell_atom_index(b, ti, tj, tk, base_n, repeat)
        r = lattice * pos_base_frac[:, b] .+ ti .* a1 .+ tj .* a2 .+ tk .* a3
        x = lattice_super \ r
        x .-= floor.(x)
        pos_super[:, ia] .= x
    end
    return lattice_super, pos_super
end

# Replicate each <basis> across every translation column of map_sym, then
# across every tile of the supercell. Returns a flat ClusterInstance list.
function _generate_instances(
        salcs::AbstractVector{SALCData},
        jphi::AbstractVector{Float64},
        map_sym::AbstractMatrix{Int},
        base_n::Int,
        repeat::NTuple{3, Int}
)::Vector{ClusterInstance}
    n1, n2, n3 = repeat
    n_trans = size(map_sym, 2)
    instances = ClusterInstance[]
    for (s, salc) in enumerate(salcs)
        J = jphi[s]
        for basis in salc.bases
            for t in 1:n_trans
                base_atoms = [map_sym[a, t] for a in basis.atoms]
                for tk in 0:(n3 - 1), tj in 0:(n2 - 1), ti in 0:(n1 - 1)
                    super_atoms = [_supercell_atom_index(a, ti, tj, tk, base_n, repeat)
                                   for a in base_atoms]
                    push!(
                        instances,
                        ClusterInstance(
                            super_atoms,
                            basis.ls,
                            salc.Lf,
                            basis.Lseq,
                            basis.weights,
                            J,
                            basis.multiplicity
                        )
                    )
                end
            end
        end
    end
    return instances
end

"""
    SpinClusterHamiltonian(xml_path; repeat=(1, 1, 1)) -> SpinClusterHamiltonian

Load a Magesty `jphi.xml` and build the full Hamiltonian: parse the XML,
construct the supercell geometry, generate one `ClusterInstance` per
`(SALC basis, translation, tile)` triple, and build the tesseral CG table.
"""
function SpinClusterHamiltonian(
        xml_path::AbstractString;
        repeat::NTuple{3, Int} = (1, 1, 1)
)::SpinClusterHamiltonian
    all(>(0), repeat) || throw(
        ArgumentError("repeat factors must be positive integers, got $repeat")
    )
    data = parse_jphi_xml(xml_path)
    base_n = data.system.n_atoms
    n1, n2, n3 = repeat
    n_super = base_n * n1 * n2 * n3
    lattice_super,
    pos_super = _build_supercell_geometry(
        data.system.lattice, data.system.pos_frac, base_n, repeat
    )
    instances = _generate_instances(
        data.salcs, data.jphi, data.system.map_sym, base_n, repeat
    )
    cg_table = build_cg_table(data.salcs)
    return SpinClusterHamiltonian(
        n_super,
        base_n,
        repeat,
        lattice_super,
        pos_super,
        instances,
        cg_table
    )
end
