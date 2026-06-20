# 06_general_supercell.jl
#
# General supercells via an integer matrix `supercell_matrix`.
#
# The default `:repeat = (n1, n2, n3)` tiles the XML *base cell* (the Magesty
# training cell) by an integer diagonal. `:supercell_matrix` instead takes an
# arbitrary 3×3 integer matrix in units of the *primitive* cell (recovered from
# the XML translation table), so you can build:
#   * non-diagonal / sheared cells (commensurate spiral / AFM ordering vectors),
#   * cells that are not integer multiples of the base cell,
#   * down to a single primitive cell.
# Clusters are placed by their relative vector (self-overlapping "face" pairs are
# un-folded into distinct ±Δ neighbors). For the ground state this matches the
# base-cell per-atom energy; for n>1 non-collinear configs it intentionally
# differs from the folded diagonal `repeat` path (and is geometrically correct).
#
#     julia --project=. examples/06_general_supercell.jl
#
# Notes
# -----
# * `:repeat` and `:supercell_matrix` are mutually exclusive.
# * The two paths number atoms differently (`:repeat` is base-cell tile-major;
#   `:supercell_matrix` is primitive cell-major). Energies agree; if you pass an
#   explicit `:initial_spins` matrix or index atoms in `:extra_measure`, mind the
#   numbering. Base-cell `:initial_spins` tiling is not available on the matrix
#   path — use `:random` (default) or a full `3 × n_atoms` config.
# * The optimized engine takes the same keyword: `JPhiSpinMC` /
#   `load_sce_hamiltonian(xml; supercell_matrix = M)` (it runs the `:tensor`
#   kernel for a general matrix; the diagonal `:repeat` keeps the fast template).

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

# Diagonal base-cell tiling (legacy path).
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
