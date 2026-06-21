# 06_general_supercell.jl
#
# General supercells via an integer matrix `supercell_matrix`.
#
# `:repeat = (n1, n2, n3)` is sugar for `:supercell_matrix = reshape_base *
# diag(n1, n2, n3)`: both build a supercell of the *primitive* cell (recovered
# from the XML translation table) and run the same un-fold path. `:supercell_matrix`
# additionally lets you pass an arbitrary 3×3 integer matrix, so you can build:
#   * non-diagonal / sheared cells (commensurate spiral / AFM ordering vectors),
#   * cells that are not integer multiples of the base cell,
#   * down to a single primitive cell.
# Clusters are placed by their relative vector (self-overlapping "face" pairs are
# un-folded into distinct ±Δ neighbors). Since the base cell is itself a supercell
# of the primitive cell, even `:repeat = (1,1,1)` un-folds into primitive cells.
#
#     julia --project=. examples/06_general_supercell.jl
#
# Notes
# -----
# * `:repeat` and `:supercell_matrix` are mutually exclusive, and the equivalent
#   `:repeat` / `:supercell_matrix` give an identical Hamiltonian.
# * Atoms are numbered primitive cell-major on both paths (no longer the old
#   tile-major order). If you pass an explicit `:initial_spins` matrix or index
#   atoms in `:extra_measure`, mind the numbering. Base-cell `:initial_spins`
#   tiling is not available — use `:random` (default) or a full `3 × n_atoms`
#   config.
# * The optimized engine takes the same keyword: `JPhiSpinMC` /
#   `load_sce_hamiltonian(xml; supercell_matrix = M)`, and both the `:tensor` and
#   `:tensor_template` kernels serve any matrix (un-fold).

using LinearAlgebra: det
using Random: MersenneTwister
using Carlo

using SpinClusterMC
using SpinClusterMC.Simple

const XML = joinpath(@__DIR__, "..", "test", "bcc_2x2x2", "jphi.xml")

function run_cell(label, extra)
    params = Dict{Symbol, Any}(
        :T => 100.0, :xml_path => XML,
        :thermalization => 0, :binsize => 50, :seed => 42,
        :spin_theta_max => 0.3, :renorm_every => 100)
    merge!(params, extra)
    mc = SCEMC(params)
    ctx = Carlo.MCContext{MersenneTwister}(params)
    Carlo.init!(mc, ctx, params)
    e0 = mc.energy / mc.h.n_atoms
    for _ in 1:200
        Carlo.sweep!(mc, ctx)
    end
    println(rpad(label, 34), "n_atoms=", lpad(mc.h.n_atoms, 3),
        "   E0/atom=", round(e0, digits = 6),
        "   E/atom@200=", round(mc.energy / mc.h.n_atoms, digits = 6))
    return nothing
end

println("bcc_2x2x2 (base cell = 16 atoms; primitive cell = 1 atom):\n")

# Diagonal `repeat` (sugar for supercell_matrix = reshape_base * diag(n)).
run_cell("repeat = (1,1,1)", Dict(:repeat => (1, 1, 1)))
run_cell("repeat = (2,2,2)", Dict(:repeat => (2, 2, 2)))

# General matrix path (primitive-cell units).
run_cell("supercell_matrix diag(2,2,2)", Dict(:supercell_matrix => [2 0 0; 0 2 0; 0 0 2]))
run_cell("supercell_matrix diag(3,2,2)", Dict(:supercell_matrix => [3 0 0; 0 2 0; 0 0 2]))
run_cell("supercell_matrix non-diagonal", Dict(:supercell_matrix => [2 1 0; 0 2 0; 0 0 2]))

println()
M = [2 1 0; 0 2 0; 0 0 2]
println("Non-diagonal M = ", M, "  (|det| = ", round(Int, det(M)),
    " primitive cells → ", round(Int, det(M)), " atoms for this monatomic cell)")
println("After equilibration the per-atom (ground-state) energy is the same across")
println("all cells above: the model is intensive at the ground state.")
