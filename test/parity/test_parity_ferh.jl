using SpinClusterMC
using SpinClusterMC.Simple
using SpinClusterMC.JPhiMagestyCarlo
using Test
using Random
using LinearAlgebra

const SIMPLE = SpinClusterMC.Simple
const OPT = SpinClusterMC.JPhiMagestyCarlo

const FERH_XML = joinpath(@__DIR__, "..", "ferh_4x4x4", "jphi.xml")

function _rand_unit_spins(rng, n)
    spins = randn(rng, 3, n)
    for i in 1:n
        spins[:, i] ./= norm(spins[:, i])
    end
    return spins
end

if isfile(FERH_XML)
    # ferh_4x4x4 has 128 atoms, 839 936 cluster instances and mixes body ∈
    # {2, 3, 4} SALCs with Lf > 0 — the largest parity gate we have. Simple's
    # per-instance loop takes ~25 s per `total_energy` on this fixture, so
    # we keep the seed count small and rely on the body-uniform sum_local
    # identity from the bcc / fege parity tests.
    h = SIMPLE.SpinClusterHamiltonian(FERH_XML)
    hj = OPT.load_sce_hamiltonian(FERH_XML)

    @testset "ferh parity: Simple.total_energy ≈ JPhiMagestyCarlo.sce_energy" begin
        for seed in (1, 42)
            rng = MersenneTwister(seed)
            spins = _rand_unit_spins(rng, h.n_atoms)
            E_simple = SIMPLE.total_energy(h, spins)
            E_opt = OPT.sce_energy(hj, spins)
            @test E_simple ≈ E_opt rtol = 1.0e-8
        end
    end

    @testset "ferh parity: Simple.delta_local_energy ≈ ΔE from explicit recompute" begin
        rng = MersenneTwister(99)
        spins = _rand_unit_spins(rng, h.n_atoms)
        E0_opt = OPT.sce_energy(hj, spins)
        for i in (1, h.n_atoms ÷ 2, h.n_atoms)
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
    @warn "Skipping ferh parity tests: $FERH_XML not found"
end
