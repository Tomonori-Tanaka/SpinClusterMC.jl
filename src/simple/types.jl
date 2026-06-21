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

The central data object of the Simple submodule. Carries everything the energy
code (`total_energy`, `local_energy`, `delta_local_energy`, `gradient`) needs
to evaluate the SCE Hamiltonian on a given spin configuration.

Built from a Magesty `jphi.xml` (which defines the *base cell* and its SALCs)
and an optional tile factor `repeat` (which builds a *supercell* by stacking
the base cell). See `docs/terminology.md` for the base / primitive / supercell
distinction.

Geometry note
-------------

This type deliberately does *not* carry the lattice vectors or the
supercell-fractional positions. The energy code does not read them — every
cluster term refers to atoms by their integer supercell index, and the
spherical harmonics are evaluated on the *spin direction*, never on a real-
space position. Geometry is only consumed during construction, in
`_generate_instances`, where the base-cell `pos_frac` from the parser is used
to compute inter-tile wraps. If a downstream extension (a position-dependent
external term, structure-factor observable, visualization) needs lattice or
positions, it should reload them from the XML or carry them itself.

# Sizes

- `n_atoms::Int` — total atom count of the supercell:
  `base_n_atoms · n_1 · n_2 · n_3` for the `repeat` path, or
  `n_prim · |det(M)|` for the `supercell_matrix = M` path.
- `base_n_atoms::Int` — atom count of the base cell (`<NumberOfAtoms>` in
  the XML).
- `repeat::NTuple{3,Int}` — tile factors `(n_1, n_2, n_3)` along the three
  base-lattice directions. `(1, 1, 1)` means "no tiling — supercell equals
  base cell". All entries must be `≥ 1` for the `repeat` path. **`(0, 0, 0)`
  is a sentinel meaning the Hamiltonian was built via `supercell_matrix`** (a
  general matrix has no diagonal tile factors).

# Hamiltonian content

- `instances::Vector{ClusterInstance}` — every translated cluster term that
  contributes to the energy. The XML's `<basis>` elements are replicated
  across every translation of `map_sym` and every tile, with atom indices
  rewritten to `1..n_atoms`. Per-basis deduplication drops translations that
  produce the same physical cluster (same sorted atoms + same `ls`);
  multiplicity stays on the surviving instance. See `_generate_instances`.
- `cg_table::CGTable` — read-only lookup of tesseral Clebsch-Gordan tensors
  keyed by `(ls, Lf, Lseq)`. The energy code fetches `T_real` from here
  rather than rebuilding it per cluster.

# Caches derived from `instances` (constructor-filled; treat as read-only)

- `max_l::Int` — largest single-site angular momentum across all instances,
  `max(inst.ls[k])`. Sets the SpheriCart calculator dimension to
  `(max_l + 1)^2`.
- `atom_to_instance_indices::Vector{Vector{Int}}` *(length n_atoms)* —
  `atom_to_instance_indices[i]` is the list of indices into `instances`
  for which `i ∈ instances[idx].atoms`. `local_energy`,
  `delta_local_energy`, and `gradient` use it to scan only the clusters
  that touch site `i` instead of the whole list.

# Construction

`SpinClusterHamiltonian(xml_path::AbstractString; repeat=(1,1,1))` runs, in
order: `parse_jphi_xml` → `_generate_instances` → `build_cg_table` →
cache derivation (`max_l`, `atom_to_instance_indices`).
"""
struct SpinClusterHamiltonian
    n_atoms::Int
    base_n_atoms::Int
    repeat::NTuple{3, Int}
    instances::Vector{ClusterInstance}
    cg_table::CGTable
    max_l::Int
    atom_to_instance_indices::Vector{Vector{Int}}
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

# Replicate each <basis> across every translation column of map_sym and every
# tile of the supercell, then deduplicate so each physical cluster appears
# exactly once. Multiplicity stays on the kept instance.
#
# Matches `JPhiMagestyCarlo._foreach_translated_instance` (per-cbc):
# - For each tile `(ti, tj, tk)` and each translation `t`, map the base atoms
#   to their supercell image, with an integer wrap so basis bonds that cross
#   a base-cell boundary land in the adjacent tile.
# - Drop any translated atom set whose (sorted-atoms, ls) signature has
#   already been seen for this basis. Different bases may legitimately
#   produce the same physical cluster, so the Set is per-basis, not global.
function _generate_instances(
        salcs::AbstractVector{SALCData},
        jphi::AbstractVector{Float64},
        map_sym::AbstractMatrix{Int},
        base_pos_frac::AbstractMatrix{Float64},
        base_n::Int,
        repeat::NTuple{3, Int};
        jphi_threshold::Float64 = 0.0
)::Vector{ClusterInstance}
    n1, n2, n3 = repeat
    n_trans = size(map_sym, 2)
    instances = ClusterInstance[]
    for (s, salc) in enumerate(salcs)
        J = jphi[s]
        # `abs(J) < 0.0` is always false, so threshold=0.0 keeps every SALC
        # bit-exactly (no early `continue` fires).
        abs(J) < jphi_threshold && continue
        for basis in salc.bases
            ls_v = collect(Int, basis.ls)
            seen = Set{Tuple{Vector{Int}, Vector{Int}}}()
            for tk in 0:(n3 - 1), tj in 0:(n2 - 1), ti in 0:(n1 - 1)
                for t in 1:n_trans
                    translated_base = [map_sym[a, t] for a in basis.atoms]
                    p_ref = @view base_pos_frac[:, translated_base[1]]
                    super_atoms = Vector{Int}(undef, length(translated_base))
                    for (k, ba) in enumerate(translated_base)
                        p = @view base_pos_frac[:, ba]
                        w1 = round(Int, p[1] - p_ref[1])
                        w2 = round(Int, p[2] - p_ref[2])
                        w3 = round(Int, p[3] - p_ref[3])
                        super_atoms[k] = _supercell_atom_index(
                            ba,
                            mod(ti + w1, n1),
                            mod(tj + w2, n2),
                            mod(tk + w3, n3),
                            base_n,
                            repeat
                        )
                    end
                    key = (sort(super_atoms), ls_v)
                    key in seen && continue
                    push!(seen, key)
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
    SpinClusterHamiltonian(xml_path; repeat=(1, 1, 1), supercell_matrix=nothing,
                           jphi_threshold=0.0) -> SpinClusterHamiltonian

Load a Magesty `jphi.xml` and build the full Hamiltonian: parse the XML,
generate the cluster instances for the requested supercell, and build the
tesseral CG table.

Two supercell modes (mutually exclusive):

- `repeat = (n_1, n_2, n_3)` (default): an integer **diagonal multiple of the
  base (XML) cell**. Uses the original `_generate_instances` path, so behavior
  and atom numbering are unchanged.
- `supercell_matrix = M` (3×3 integer matrix, `det(M) ≠ 0`): an **arbitrary
  supercell of the primitive cell** recovered from the XML's translation table.
  Handles non-diagonal and non-base-multiple cells (down to a single primitive
  cell). Atoms use a primitive cell-major numbering; clusters are placed by their
  relative vector and self-overlapping "face" pairs are un-folded into distinct
  ±Δ neighbors (see `build_templates`). For a ferromagnet / ground state the
  per-atom energy equals the base-cell model; for n > 1 non-collinear configs it
  differs from the folded diagonal `repeat` path (and is geometrically correct).

# Arguments

- `repeat`: Supercell tile factors `(n_1, n_2, n_3)`; all entries must be `≥ 1`.
- `supercell_matrix`: 3×3 integer matrix in primitive-cell units, or `nothing`.
  Cannot be combined with a non-default `repeat`.
- `jphi_threshold`: Drop SALCs with `|J_s| < jphi_threshold` (eV) before
  building cluster instances. Use this to skip near-zero couplings produced
  by sparse-modeled `jphi.xml`. Must be non-negative. Default `0.0` keeps
  every SALC, including those with `J_s = 0.0` exactly (`abs(0) ≥ 0`); pass
  `eps()` or `nextfloat(0.0)` to drop strict zeros. Throws `ArgumentError`
  if every SALC is filtered out.
"""
function SpinClusterHamiltonian(
        xml_path::AbstractString;
        repeat::NTuple{3, Int} = (1, 1, 1),
        supercell_matrix::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
        jphi_threshold::Real = 0.0
)::SpinClusterHamiltonian
    thr = Float64(jphi_threshold)
    thr ≥ 0 ||
        throw(ArgumentError("jphi_threshold must be non-negative, got $thr"))
    data = parse_jphi_xml(xml_path)
    base_n = data.system.n_atoms

    # Unified un-fold path: both `repeat` and `supercell_matrix` reduce to an
    # integer supercell matrix M (primitive-cell units). `repeat = (n1, n2, n3)`
    # is the sugar `M = reshape_base * diag(n)` (Phase 2). The base cell is itself
    # a supercell of the primitive cell, so even `repeat = (1, 1, 1)` un-folds
    # into `n_trans` primitive cells with cell-major atom numbering; the physical
    # energy is unchanged (folded ≡ un-fold at the base cell), only the atom index
    # → atom map differs from the historical tile-major numbering.
    prim = extract_primitive(data.system)
    if supercell_matrix === nothing
        all(>(0), repeat) || throw(
            ArgumentError("repeat factors must be positive integers, got $repeat")
        )
        M = _supercell_from_repeat(prim.reshape_base, repeat)
        repeat_field = repeat
    else
        repeat == (1, 1, 1) || throw(ArgumentError(
            "specify either repeat or supercell_matrix, not both"
        ))
        size(supercell_matrix) == (3, 3) || throw(ArgumentError(
            "supercell_matrix must be 3×3, got $(size(supercell_matrix))"
        ))
        M = SMatrix{3, 3, Int}(supercell_matrix)
        _int_det3(M) != 0 ||
            throw(ArgumentError("supercell_matrix is singular (det = 0)"))
        # `(0, 0, 0)` sentinel marks a directly-specified matrix (no repeat sugar).
        repeat_field = (0, 0, 0)
    end
    templates = build_templates(
        data.salcs, data.jphi, data.system, prim; jphi_threshold = thr
    )
    isempty(templates) && throw(ArgumentError(
        "jphi_threshold=$thr eV filters out all SALCs; Hamiltonian " *
        "would be empty"
    ))
    instances, n_atoms = _generate_instances_matrix(templates, prim, M)

    # cg_table is built from the unfiltered SALC list. Dropped SALCs produce
    # no instances, so their (ls, Lf, Lseq) keys are never looked up at
    # evaluation time — the extra entries are harmless and cheap.
    cg_table = build_cg_table(data.salcs)
    max_l = _max_l_in_instances(instances)
    atom_to_instance_indices = _build_atom_to_instance_indices(instances, n_atoms)
    return SpinClusterHamiltonian(
        n_atoms,
        base_n,
        repeat_field,
        instances,
        cg_table,
        max_l,
        atom_to_instance_indices
    )
end

# Largest `ls[i]` seen across all cluster instances. Used to size the
# SpheriCart spherical-harmonics calculator: it must hold up to (max_l+1)^2
# tesseral basis functions per atom.
function _max_l_in_instances(instances::AbstractVector{ClusterInstance})::Int
    m = 0
    for inst in instances
        for l in inst.ls
            l > m && (m = l)
        end
    end
    return m
end

# For each atom `i` in 1..n_atoms, list of indices into `instances` for which
# `i ∈ instances[idx].atoms`. Built once at construction; queried per
# `local_energy` / `delta_local_energy` call so those scan only clusters that
# touch the site of interest.
function _build_atom_to_instance_indices(
        instances::AbstractVector{ClusterInstance}, n_atoms::Int
)::Vector{Vector{Int}}
    mapping = [Int[] for _ in 1:n_atoms]
    for (idx, inst) in enumerate(instances)
        for a in inst.atoms
            push!(mapping[a], idx)
        end
    end
    return mapping
end
