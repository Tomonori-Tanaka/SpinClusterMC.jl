using SpinClusterMC
using SpinClusterMC.Simple
using SpinClusterMC.JPhiMagestyCarlo
using Test
using Random
using LinearAlgebra
using StaticArrays: SVector

const SIMPLE = SpinClusterMC.Simple
const OPT = SpinClusterMC.JPhiMagestyCarlo

const FEGE_XML = joinpath(@__DIR__, "..", "fege_2x2x2", "jphi.xml")

function _rand_unit_spins(rng, n)
    spins = randn(rng, 3, n)
    for i in 1:n
        spins[:, i] ./= norm(spins[:, i])
    end
    return spins
end

if isfile(FEGE_XML)
    h = SIMPLE.SpinClusterHamiltonian(FEGE_XML)
    hj = OPT.load_sce_hamiltonian(FEGE_XML)

    @testset "fege parity: Simple.total_energy ≈ JPhiMagestyCarlo.sce_energy" begin
        # fege carries Lf ∈ {0, 1, 2, 3, 4} (anisotropic SALCs), so this is
        # the first parity check that exercises the high-l tesseral CG path
        # of `_instance_energy` end-to-end.
        for seed in (1, 2, 7, 42, 100)
            rng = MersenneTwister(seed)
            spins = _rand_unit_spins(rng, h.n_atoms)
            E_simple = SIMPLE.total_energy(h, spins)
            E_opt = OPT.sce_energy(hj, spins)
            @test E_simple ≈ E_opt rtol = 1.0e-8
        end
    end

    @testset "fege parity: Simple.delta_local_energy ≈ ΔE from explicit recompute" begin
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

    @testset "fege parity: sum(local) = body · total (uniform body=2)" begin
        # fege only contains body=2 SALCs, so the uniform-body identity
        # applies. This catches any sign/normalization slip in the
        # higher-Lf branches that the bcc fixture (Lf=0 only) cannot.
        rng = MersenneTwister(53)
        spins = _rand_unit_spins(rng, h.n_atoms)
        sum_local = sum(SIMPLE.local_energy(h, spins, i) for i in 1:h.n_atoms)
        E_total = SIMPLE.total_energy(h, spins)
        @test sum_local / 2 ≈ E_total rtol = 1.0e-10
    end
else
    @warn "Skipping fege parity tests: $FEGE_XML not found"
end
