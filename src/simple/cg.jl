"""
CGTable construction. For every `<basis>` in the parsed SALC list we need the
tesseral Clebsch-Gordan tensor `T_real[m_1, …, m_N; Mf]` keyed by
`(ls, Lf, Lseq)`. We delegate the math to
`Magesty.AngularMomentumCoupling.build_all_real_bases`, which produces every
`(Lseq, Lf)` allowed by the left-coupling tree of a given `ls` in one call,
already converted from complex `Y_l^m` to real `Z_l^m` with the correct
phase compensation.

Why depend on Magesty here at all
---------------------------------

Tesseral CG with phase compensation is ~150–300 lines of careful linear
algebra; rewriting it on the Simple side risks subtle sign errors. The Magesty
implementation is the reference, and depending on it for this one piece of
math keeps the Simple energy code aligned with the Magesty technical notes
(see CLAUDE.md's "spherical harmonics are tesseral / real" section).

Per-`ls` caching
----------------

`build_all_real_bases(ls)` is the expensive call. We collect every distinct
`ls` pattern referenced in the SALC list (across all bases), call
`build_all_real_bases` once per unique `ls`, and register every
`(Lseq, Lf)` it returns. The XML may not actually use every Lseq/Lf
combination Magesty enumerates; the unused entries are cheap and we leave
them in the table so that the loader stays a straight one-pass build.
"""

using Magesty.AngularMomentumCoupling: build_all_real_bases

"""
    build_cg_table(salcs) -> CGTable

Walk a SALC list and return a `CGTable` covering every `(ls, Lf, Lseq)`
combination reachable from the `ls` patterns used by those SALCs. One call
to `Magesty.AngularMomentumCoupling.build_all_real_bases` per unique `ls`.

The XML's `<basis Lseq="...">` follows the same left-coupling-tree convention
Magesty uses for its path enumeration, so the keys produced here match the
keys an energy-evaluation code constructs from `ClusterInstance.ls / Lf / Lseq`.
"""
function build_cg_table(salcs::AbstractVector{SALCData})::CGTable
    unique_ls = Set{Vector{Int}}()
    for salc in salcs
        for basis in salc.bases
            push!(unique_ls, collect(Int, basis.ls))
        end
    end
    entries = Dict{Tuple{Vector{Int}, Int, Vector{Int}}, Array{Float64}}()
    for ls in unique_ls
        bases_by_L, paths_by_L = build_all_real_bases(ls)
        for (Lf, tensor_list) in bases_by_L
            path_list = paths_by_L[Lf]
            length(path_list) == length(tensor_list) || throw(
                ArgumentError(
                "build_all_real_bases inconsistency: ls=$ls Lf=$Lf has $(length(path_list)) paths but $(length(tensor_list)) tensors"
            )
            )
            for (path, tensor) in zip(path_list, tensor_list)
                entries[(copy(ls), Int(Lf), collect(Int, path))] = tensor
            end
        end
    end
    return CGTable(entries)
end
