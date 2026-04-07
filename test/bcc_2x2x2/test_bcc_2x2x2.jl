using SpinClusterMC.JPhiMagestyCarlo
using Test

const XML_2x2x2 = joinpath(@__DIR__, "jphi.xml")

@testset "bcc_2x2x2 ferromagnetic energy" begin
    h = load_sce_hamiltonian(XML_2x2x2; repeat = (1, 1, 1))
    @test h.n_atoms == 16

    # Ferromagnetic configuration: all spins along z
    spins = zeros(3, h.n_atoms)
    spins[3, :] .= 1.0

    E_per_atom = sce_energy(h, spins) / h.n_atoms

    # For Jij = -1 eV/bond with 8 nearest neighbours and 6 second-nearest
    # neighbours per atom, the ferromagnetic ground-state energy is -(2+√3) eV/atom.
    @test E_per_atom ≈ -(2 + sqrt(3)) rtol = 1e-8
end
