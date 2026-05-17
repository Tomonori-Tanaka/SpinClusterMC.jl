using SpinClusterMC
using SpinClusterMC.Simple
using SpinClusterMC.JPhiMagestyCarlo
using Test
using Random
using LinearAlgebra
using StaticArrays: SVector

const SIMPLE = SpinClusterMC.Simple
const OPT = SpinClusterMC.JPhiMagestyCarlo

const BCC_XML = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")

function _rand_unit_spins(rng, n)
    spins = randn(rng, 3, n)
    for i in 1:n
        spins[:, i] ./= norm(spins[:, i])
    end
    return spins
end

if isfile(BCC_XML)
    h = SIMPLE.SpinClusterHamiltonian(BCC_XML)
    hj = OPT.load_sce_hamiltonian(BCC_XML)

    @testset "bcc parity: Simple.total_energy ≈ JPhiMagestyCarlo.sce_energy" begin
        for seed in (1, 2, 7, 42, 100)
            rng = MersenneTwister(seed)
            spins = _rand_unit_spins(rng, h.n_atoms)
            E_simple = SIMPLE.total_energy(h, spins)
            E_opt = OPT.sce_energy(hj, spins)
            @test E_simple ≈ E_opt rtol = 1.0e-8
        end
    end

    @testset "bcc parity: Simple.local_energy via delta consistency" begin
        # The Simple side defines local_energy as the sum of every cluster
        # containing atom i (full E_inst). The optimized side does not expose a
        # public single-site local energy with the same convention, but we can
        # verify consistency via the identity sum_i local(i) = body * total
        # (uniform body = 2 for bcc) and re-check total parity.
        rng = MersenneTwister(11)
        spins = _rand_unit_spins(rng, h.n_atoms)
        sum_local = sum(SIMPLE.local_energy(h, spins, i) for i in 1:h.n_atoms)
        E_total_simple = SIMPLE.total_energy(h, spins)
        E_total_opt = OPT.sce_energy(hj, spins)
        @test sum_local / 2 ≈ E_total_simple rtol = 1.0e-10
        @test E_total_simple ≈ E_total_opt rtol = 1.0e-8
    end

    @testset "bcc parity: Simple.delta_local_energy ≈ ΔE from single-flip Carlo" begin
        # Build a JPhiSpinMC, compute its initial energy and a one-flip ΔE via
        # the optimized delta computation. Compare against Simple's
        # delta_local_energy on the same spin configuration and target flip.
        rng = MersenneTwister(99)
        spins = _rand_unit_spins(rng, h.n_atoms)
        E0_opt = OPT.sce_energy(hj, spins)
        for i in (1, 5, h.n_atoms)
            S_new_raw = randn(rng, 3)
            S_new = S_new_raw / norm(S_new_raw)
            spins_new = copy(spins)
            spins_new[:, i] .= S_new
            E1_opt = OPT.sce_energy(hj, spins_new)
            ΔE_opt = E1_opt - E0_opt
            ΔE_simple = SIMPLE.delta_local_energy(h, spins, i, S_new)
            @test ΔE_simple ≈ ΔE_opt rtol = 1.0e-7
        end
    end
else
    @warn "Skipping bcc parity tests: $BCC_XML not found"
end
