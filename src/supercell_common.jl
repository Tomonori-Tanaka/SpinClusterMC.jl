"""
    SupercellCommon

Shared supercell geometry used by both engines (`Simple` and
`JPhiMagestyCarlo`) to tile cluster interactions onto an arbitrary integer
supercell matrix `M` (3×3, `det(M) != 0`) expressed in units of the *primitive*
cell, mirroring how Sunny.jl reshapes a primitive cell onto a general supercell.

Provides:

  - integer linear algebra (`_int_det3`, `_adjugate3`, `_col_hermite`,
    `_wrap_offset_into_supercell`) — pure integer arithmetic, exact for any
    sign of `det(M)`;
  - primitive-cell recovery from a base cell's translation table
    (`PrimitiveCell`, `extract_primitive`);
  - tiling primitives (`_cluster_base_stabilizer`, `_cluster_offsets`,
    `_enumerate_cells`, `_supercell_from_repeat`).

The engine-specific parts (which concrete cluster-instance type to emit, the
energy kernels) live in each engine; this module is geometry only.
"""
module SupercellCommon

using StaticArrays: SVector, SMatrix, MMatrix
using LinearAlgebra: cross, dot, norm, inv, det

# =============================================================================
# Integer linear algebra (3×3)
# =============================================================================

"""
    _int_det3(M::SMatrix{3, 3, Int}) -> Int

Exact determinant of a 3×3 integer matrix via the rule of Sarrus.
"""
@inline function _int_det3(M::SMatrix{3, 3, Int})::Int
    return M[1, 1] * (M[2, 2] * M[3, 3] - M[2, 3] * M[3, 2]) -
           M[1, 2] * (M[2, 1] * M[3, 3] - M[2, 3] * M[3, 1]) +
           M[1, 3] * (M[2, 1] * M[3, 2] - M[2, 2] * M[3, 1])
end

"""
    _adjugate3(M::SMatrix{3, 3, Int}) -> SMatrix{3, 3, Int}

Adjugate (classical adjoint) of a 3×3 integer matrix: the transpose of the
cofactor matrix. Satisfies `M * adj(M) = adj(M) * M = det(M) * I`, so
`M^{-1} = adj(M) / det(M)` for `det(M) != 0`. Pure integer arithmetic.
"""
@inline function _adjugate3(M::SMatrix{3, 3, Int})::SMatrix{3, 3, Int}
    a, b, c = M[1, 1], M[1, 2], M[1, 3]
    d, e, f = M[2, 1], M[2, 2], M[2, 3]
    g, h, i = M[3, 1], M[3, 2], M[3, 3]
    # Cofactors C[r, s]; adj = transpose(C), i.e. adj[r, s] = C[s, r].
    return SMatrix{3, 3, Int}(
        # column 1 (adj[:, 1]) = C[1, :]
        e * i - f * h,
        -(d * i - f * g),
        d * h - e * g,
        # column 2 (adj[:, 2]) = C[2, :]
        -(b * i - c * h),
        a * i - c * g,
        -(a * h - b * g),
        # column 3 (adj[:, 3]) = C[3, :]
        b * f - c * e,
        -(a * f - c * d),
        a * e - b * d
    )
end

# Column operations on a 3×3 mutable integer matrix (right-multiplication by
# unimodular elementary matrices). Used by `_col_hermite`.
@inline function _swap_cols!(A::MMatrix{3, 3, Int}, p::Int, q::Int)
    for r in 1:3
        A[r, p], A[r, q] = A[r, q], A[r, p]
    end
    return A
end

@inline function _negate_col!(A::MMatrix{3, 3, Int}, p::Int)
    for r in 1:3
        A[r, p] = -A[r, p]
    end
    return A
end

# col_dst -= k * col_src
@inline function _col_axpy!(A::MMatrix{3, 3, Int}, dst::Int, src::Int, k::Int)
    for r in 1:3
        A[r, dst] -= k * A[r, src]
    end
    return A
end

"""
    _col_hermite(M::SMatrix{3, 3, Int}) -> (H::SMatrix{3, 3, Int}, U::SMatrix{3, 3, Int})

Column Hermite normal form of a non-singular 3×3 integer matrix.

Returns `(H, U)` such that `H == M * U`, with `U` unimodular (`|det U| == 1`)
and `H` **lower triangular** with strictly positive diagonal (`H[r, r] > 0`,
`H[r, s] == 0` for `s > r`). Because `U` is unimodular, the columns of `H` span
the same integer lattice as the columns of `M` (`H * ℤ³ == M * ℤ³`), and the
diagonal product `H[1,1] * H[2,2] * H[3,3] == |det(M)|`.

The lower-triangular diagonal box `{0:H[1,1]-1} × {0:H[2,2]-1} × {0:H[3,3]-1}`
is then a complete set of coset representatives of `ℤ³ / (M ℤ³)` — exactly the
`|det(M)|` supercell cells. (Sub-diagonal entries are left unreduced; only the
diagonal and triangular structure matter for the cell enumeration.)

`M` must be non-singular; callers guard `det(M) != 0` first.
"""
function _col_hermite(M::SMatrix{3, 3, Int})
    H = MMatrix{3, 3, Int}(M)
    U = MMatrix{3, 3, Int}(1, 0, 0, 0, 1, 0, 0, 0, 1)  # identity
    for r in 1:3
        # Use columns r:3 to clear H[r, s] for s > r via Euclidean reduction:
        # repeatedly pivot the smallest nonzero entry of row r into column r and
        # subtract integer multiples from the other columns.
        while any(H[r, s] != 0 for s in (r + 1):3)
            cmin = r
            for s in r:3
                if H[r, s] != 0 && (H[r, cmin] == 0 || abs(H[r, s]) < abs(H[r, cmin]))
                    cmin = s
                end
            end
            if cmin != r
                _swap_cols!(H, r, cmin)
                _swap_cols!(U, r, cmin)
            end
            for s in (r + 1):3
                if H[r, s] != 0
                    q = div(H[r, s], H[r, r])
                    _col_axpy!(H, s, r, q)
                    _col_axpy!(U, s, r, q)
                end
            end
        end
        if H[r, r] < 0
            _negate_col!(H, r)
            _negate_col!(U, r)
        end
    end
    return SMatrix{3, 3, Int}(H), SMatrix{3, 3, Int}(U)
end

"""
    _wrap_offset_into_supercell(c::NTuple{3, Int}, M, adjM, detM) -> NTuple{3, Int}

Fold an integer cell offset `c` (in primitive-cell units) into the fundamental
domain of the supercell lattice spanned by the columns of `M`.

Returns the unique representative `rep = c - M * floor(M^{-1} c)` of the coset
`c + M ℤ³`, which lies in `M * [0, 1)³`. The map is constant on cosets, so two
offsets that differ by a supercell lattice vector wrap to the same `rep`; this
makes `rep` a canonical key into the cell-index table.

`adjM = adj(M)` and `detM = det(M)` are passed in (precomputed once per
supercell). Uses exact integer floor division (`fld`), so it is correct for any
sign of `detM` without floating point.
"""
@inline function _wrap_offset_into_supercell(
        c::NTuple{3, Int},
        M::SMatrix{3, 3, Int},
        adjM::SMatrix{3, 3, Int},
        detM::Int
)::NTuple{3, Int}
    cv = SVector{3, Int}(c[1], c[2], c[3])
    # f = adj(M) * c = detM * (M^{-1} c); so (M^{-1} c)_i = f_i / detM.
    f = adjM * cv
    # floor(M^{-1} c) componentwise via exact integer floor division.
    q = SVector{3, Int}(fld(f[1], detM), fld(f[2], detM), fld(f[3], detM))
    rep = cv - M * q
    return (rep[1], rep[2], rep[3])
end

# =============================================================================
# Primitive cell extraction
# =============================================================================

"""
    PrimitiveCell

The primitive cell underlying a `jphi.xml` base cell, recovered from the base
cell's discrete translation table (`map_sym` / `n_trans`).

The base (XML) cell is itself a supercell of this primitive cell:
`base_lattice == primitive_lattice * reshape_base`, with
`|det(reshape_base)| == n_trans` and `n_prim == base_n_atoms / n_trans` atoms in
the primitive cell.

# Fields

- `lattice`: 3×3 matrix whose columns are the primitive lattice vectors
  `[b1 b2 b3]` (Å), right-handed (`det > 0`).
- `pos_frac`: 3×n_prim fractional positions of the primitive-cell sublattices,
  each wrapped into `[0, 1)` of the primitive cell.
- `n_prim`: number of sublattices (atoms) in the primitive cell.
- `base_to_prim`: for each base atom `a`, the pair `(s, Δ)` giving its
  sublattice index `s ∈ 1:n_prim` and integer primitive-cell offset `Δ`
  (`primitive coords of a == pos_frac[:, s] + Δ`).
- `prim_to_base`: inverse map `(s, Δ) => a` for the base atoms (the `n_trans`
  cells that fill the base cell).
- `reshape_base`: integer 3×3 matrix with
  `base_lattice == primitive_lattice * reshape_base`, `|det| == n_trans`.
"""
struct PrimitiveCell
    lattice::Matrix{Float64}
    pos_frac::Matrix{Float64}
    n_prim::Int
    base_to_prim::Vector{Tuple{Int, NTuple{3, Int}}}
    prim_to_base::Dict{Tuple{Int, NTuple{3, Int}}, Int}
    reshape_base::Matrix{Int}
end

# Greedily pick the three shortest linearly independent vectors from `cands`
# (already filtered of (near-)zero vectors). Returns them as the columns of a
# 3×3 matrix. `reltol` is a dimensionless tolerance: the independence tests use
# normalized sine/volume ratios so they behave correctly for skewed or
# large-magnitude lattices.
function _shortest_independent3(cands::Vector{SVector{3, Float64}}, reltol::Float64)
    order = sortperm(cands; by = norm)
    b1 = nothing
    b2 = nothing
    b3 = nothing
    for idx in order
        v = cands[idx]
        if b1 === nothing
            b1 = v
        elseif b2 === nothing
            # Independent from b1 iff not collinear: sin(angle) > reltol.
            if norm(cross(b1, v)) > reltol * norm(b1) * norm(v)
                b2 = v
            end
        else
            # Independent from {b1, b2} iff not coplanar: the triple product
            # (parallelepiped volume) relative to the edge-length product.
            if abs(dot(cross(b1, b2), v)) > reltol * norm(b1) * norm(b2) * norm(v)
                b3 = v
                break
            end
        end
    end
    (b1 === nothing || b2 === nothing || b3 === nothing) && error(
        "could not find 3 linearly independent primitive translations; " *
        "the base cell's translation table may be degenerate"
    )
    return hcat(b1, b2, b3)
end

"""
    extract_primitive(lattice, pos_frac, map_sym, n_trans) -> PrimitiveCell

Recover the primitive cell from a base cell's geometry and translation table,
following the same construction as Magesty's `SunnyExport._sunny_primitive`.

Arguments are raw arrays so both engines can call it with their own parsed
structs: `lattice` (3×3, columns are base lattice vectors), `pos_frac`
(3×base_n fractional positions), `map_sym` (base_n × n_trans translation table),
`n_trans`.

Algorithm:
 1. Collect candidate primitive translations as the displacements of atom 1
    under every column of `map_sym` (`pos[map_sym[1, t]] - pos[1]`, minimum
    image), plus the three base lattice vectors as a fallback.
 2. Choose the three shortest linearly independent candidates as the primitive
    lattice `Lp`; flip the third vector if needed so `det(Lp) > 0`.
 3. Classify every base atom into a sublattice by its primitive fractional
    position `mod(Lp^{-1} * cart, 1)`, and record its integer cell offset.
 4. Build `reshape_base = round(Lp^{-1} * base_lattice)` and assert consistency
    (`n_prim * n_trans == base_n`, `|det(reshape_base)| == n_trans`, each
    sublattice has exactly `n_trans` base atoms).
"""
function extract_primitive(
        lattice::AbstractMatrix{<:Real},
        pos_frac::AbstractMatrix{<:Real},
        map_sym::AbstractMatrix{Int},
        n_trans::Int
)::PrimitiveCell
    base_n = size(pos_frac, 2)

    scale = norm(lattice[:, 1]) + norm(lattice[:, 2]) + norm(lattice[:, 3])
    tol_len = 1.0e-8 * scale
    tol_frac = 1.0e-6

    # --- 1. candidate primitive translations -------------------------------
    # Atom 1 is the reference; its image under translation `t` differs from it
    # by the lattice translation vector t_k (independent of which column is the
    # identity translation).
    cands = SVector{3, Float64}[]
    p1 = SVector{3, Float64}(pos_frac[1, 1], pos_frac[2, 1], pos_frac[3, 1])
    for t in 1:n_trans
        dest = map_sym[1, t]
        df = SVector{3, Float64}(
            pos_frac[1, dest] - p1[1],
            pos_frac[2, dest] - p1[2],
            pos_frac[3, dest] - p1[3]
        )
        df = df - round.(df)                         # minimum image (base-cell frac)
        v = SVector{3, Float64}(lattice * df)        # Cartesian
        norm(v) > tol_len && push!(cands, v)
    end
    # Always include the base lattice vectors (Magesty `_sunny_primitive`): for
    # 3D-periodic crystals the translation-derived candidates already span, and
    # this also covers open directions.
    for j in 1:3
        push!(cands, SVector{3, Float64}(lattice[:, j]))
    end

    # --- 2. primitive lattice ----------------------------------------------
    Lp = _shortest_independent3(cands, 1.0e-8)
    if det(Lp) < 0
        Lp = hcat(Lp[:, 1], Lp[:, 2], -Lp[:, 3])     # enforce right-handed
    end
    Lpi = inv(Lp)

    # --- 3. sublattice classification --------------------------------------
    sub_frac = Vector{SVector{3, Float64}}()         # canonical in-cell position per sublattice
    base_to_prim = Vector{Tuple{Int, NTuple{3, Int}}}(undef, base_n)
    for a in 1:base_n
        cart = SVector{3, Float64}(lattice * pos_frac[:, a])
        g = SVector{3, Float64}(Lpi * cart)          # primitive coords
        frac = g - floor.(g)                         # in [0, 1)
        # Snap components within tol of 1.0 back to 0.0 to avoid split groups.
        frac = SVector{3, Float64}(
            (1.0 - frac[1] < tol_frac ? 0.0 : frac[1]),
            (1.0 - frac[2] < tol_frac ? 0.0 : frac[2]),
            (1.0 - frac[3] < tol_frac ? 0.0 : frac[3])
        )
        s = findfirst(f -> norm(f - frac) < tol_frac, sub_frac)
        if s === nothing
            push!(sub_frac, frac)
            s = length(sub_frac)
        end
        Δf = g - sub_frac[s]
        Δ = (round(Int, Δf[1]), round(Int, Δf[2]), round(Int, Δf[3]))
        base_to_prim[a] = (s, Δ)
    end
    n_prim = length(sub_frac)

    # --- 4. reshape_base and consistency -----------------------------------
    rb_f = Lpi * lattice
    reshape_base = round.(Int, rb_f)
    maximum(abs.(rb_f - reshape_base)) < tol_frac ||
        error("base lattice is not an integer multiple of the recovered " *
              "primitive lattice (Lp^{-1} * base_lattice not integer)")

    n_prim * n_trans == base_n || error(
        "primitive extraction inconsistency: n_prim=$n_prim * n_trans=$n_trans " *
        "!= base_n=$base_n"
    )
    abs(_int_det3(SMatrix{3, 3, Int}(reshape_base))) == n_trans || error(
        "|det(reshape_base)|=$(abs(det(reshape_base))) != n_trans=$n_trans"
    )
    # Each sublattice must contain exactly n_trans base atoms.
    counts = zeros(Int, n_prim)
    for a in 1:base_n
        counts[base_to_prim[a][1]] += 1
    end
    all(==(n_trans), counts) || error(
        "sublattice occupancy $(counts) != n_trans=$n_trans for all sublattices"
    )

    # Inverse map for the base atoms.
    prim_to_base = Dict{Tuple{Int, NTuple{3, Int}}, Int}()
    for a in 1:base_n
        prim_to_base[base_to_prim[a]] = a
    end

    pos_prim = Matrix{Float64}(undef, 3, n_prim)
    for s in 1:n_prim
        pos_prim[:, s] .= sub_frac[s]
    end

    return PrimitiveCell(
        Matrix{Float64}(Lp),
        pos_prim,
        n_prim,
        base_to_prim,
        prim_to_base,
        reshape_base
    )
end

# =============================================================================
# Tiling primitives
# =============================================================================

"""
    _cluster_base_stabilizer(atoms, map_sym, n_trans) -> Int

Number of base-cell translations that map the cluster's atom SET onto itself
(the cluster's stabilizer under the base-cell translation group). For a pair
whose partner sits at a half period (2Δ ≡ 0, a cell-boundary "face" pair) this
is 2; for an interior cluster it is 1. The XML `multiplicity` folds in this
self-overlap (equal-distance ± images), so dividing it out un-folds the cluster
for general supercell tiling.
"""
function _cluster_base_stabilizer(
        atoms::AbstractVector{<:Integer},
        map_sym::AbstractMatrix{Int},
        n_trans::Int
)::Int
    target = sort(collect(Int, atoms))
    n = length(atoms)
    s = 0
    buf = Vector{Int}(undef, n)
    for t in 1:n_trans
        for (k, a) in enumerate(atoms)
            buf[k] = map_sym[a, t]
        end
        sort(buf) == target && (s += 1)
    end
    return s
end

"""
    _cluster_offsets(atoms, prim::PrimitiveCell)
        -> (pivot_subl::Int, site_subl::Vector{Int}, site_delta::Vector{NTuple{3,Int}})

Express a base-cell cluster (atom indices `atoms`) relative to the primitive
cell: each site's sublattice and its pivot-relative (site 1) integer
primitive-cell offset (`site_delta[1] == (0, 0, 0)`). The offset is the actual
listed displacement in primitive-cell units (minimum image is not applied; the
supercell wrap handles periodicity).
"""
function _cluster_offsets(atoms::AbstractVector{<:Integer}, prim::PrimitiveCell)
    a1 = atoms[1]
    subl1, δ1 = prim.base_to_prim[a1]
    N = length(atoms)
    site_subl = Vector{Int}(undef, N)
    site_delta = Vector{NTuple{3, Int}}(undef, N)
    for (k, ak) in enumerate(atoms)
        sk, δk = prim.base_to_prim[ak]
        site_subl[k] = sk
        site_delta[k] = (δk[1] - δ1[1], δk[2] - δ1[2], δk[3] - δ1[3])
    end
    return subl1, site_subl, site_delta
end

"""
    _supercell_from_repeat(reshape_base, repeat) -> SMatrix{3, 3, Int}

The primitive-units supercell matrix equivalent to tiling the *base* cell by the
diagonal `repeat = (n1, n2, n3)`: `M = reshape_base * diag(n1, n2, n3)`.
"""
@inline function _supercell_from_repeat(
        reshape_base::AbstractMatrix{Int},
        repeat::NTuple{3, Int}
)::SMatrix{3, 3, Int}
    rb = SMatrix{3, 3, Int}(reshape_base)
    D = SMatrix{3, 3, Int}(repeat[1], 0, 0, 0, repeat[2], 0, 0, 0, repeat[3])
    return rb * D
end

"""
    _enumerate_cells(M, adjM, detM) -> (cell_index, cells_by_id)

Enumerate the `|det(M)|` supercell cells (canonical primitive-cell offsets) from
the HNF diagonal box. `cell_index[wrapped_offset] = cell_id` and
`cells_by_id[cell_id] = wrapped_offset`.
"""
function _enumerate_cells(
        M::SMatrix{3, 3, Int},
        adjM::SMatrix{3, 3, Int},
        detM::Int
)
    H, _ = _col_hermite(M)
    ncells = abs(detM)
    cell_index = Dict{NTuple{3, Int}, Int}()
    cells_by_id = Vector{NTuple{3, Int}}(undef, ncells)
    cid = 0
    for c3 in 0:(H[3, 3] - 1), c2 in 0:(H[2, 2] - 1), c1 in 0:(H[1, 1] - 1)
        w = _wrap_offset_into_supercell((c1, c2, c3), M, adjM, detM)
        if !haskey(cell_index, w)
            cid += 1
            cell_index[w] = cid
            cells_by_id[cid] = w
        end
    end
    cid == ncells ||
        error("cell enumeration produced $cid cells, expected $ncells")
    return cell_index, cells_by_id
end

end # module SupercellCommon
