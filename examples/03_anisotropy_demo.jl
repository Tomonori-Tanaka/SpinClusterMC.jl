# 03_anisotropy_demo.jl
#
# Detect the easy axis of the `fege_2x2x2` Hamiltonian by comparing the
# total energy of three ferromagnetic configurations aligned along +x̂,
# +ŷ, +ẑ, and the body-diagonal (1,1,1)/√3. fege carries `Lf ∈ {0..4}`
# SALCs so the energy depends on the global spin direction — unlike the
# `bcc_2x2x2` fixture (Lf=0 only) which is isotropic.
#
#     julia --project=. examples/03_anisotropy_demo.jl
#
# Pedagogical vs production notes
# -------------------------------
# * This example does not run an MC sweep at all. It just builds the
#   Hamiltonian and evaluates `total_energy` four times. Production
#   anisotropy studies would scan a finer grid of directions and / or run
#   short MCs at low T from each direction; we keep it brittle and
#   explicit here to highlight where the anisotropy comes from.
# * The "ferromagnetic + direction" initial state is built via
#   `init_spins((dx, dy, dz), n_atoms, base_n_atoms)`, which accepts a
#   3-tuple direction and aligns every atom along it.

using Printf: @printf
using LinearAlgebra: normalize

using SpinClusterMC
using SpinClusterMC.Simple

const XML = joinpath(@__DIR__, "..", "test", "fege_2x2x2", "jphi.xml")

h = SpinClusterHamiltonian(XML)
n_atoms = h.n_atoms
base_n = h.base_n_atoms
println(
    "Loaded fege Hamiltonian: $(n_atoms) atoms, $(length(h.instances)) cluster instances",
)
println("Largest single-site l = $(h.max_l)")
println()

# Four trial directions. The first three are axis-aligned; the fourth is
# the cubic body diagonal, which often differs in energy from the axes
# when cubic anisotropy is present.
directions = [
    ("+x̂", (1.0, 0.0, 0.0)),
    ("+ŷ", (0.0, 1.0, 0.0)),
    ("+ẑ", (0.0, 0.0, 1.0)),
    ("(1,1,1)/√3", Tuple(normalize([1.0, 1.0, 1.0]))),
]

energies = [
    (label, total_energy(h, init_spins(dir, n_atoms, base_n))) for
    (label, dir) in directions
]
E_z = first(E for (label, E) in energies if label == "+ẑ")

println(rpad("Direction", 12), rpad("E_total (eV)", 24), "ΔE (eV) wrt +ẑ")
println(rpad("---------", 12), rpad("------------", 24), "--------------")
for (label, E) in energies
    @printf "%-12s%-24.10f%+.6e\n" label E (E - E_z)
end

println()
println("Anisotropy ΔE (energy difference between directions) is set by")
println("the Lf > 0 SALCs in the XML; for an Lf=0-only Hamiltonian (e.g.,")
println("bcc_2x2x2) every direction gives the same energy by construction.")
