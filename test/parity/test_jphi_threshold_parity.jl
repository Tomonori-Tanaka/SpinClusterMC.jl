using SpinClusterMC
using SpinClusterMC.Simple
using SpinClusterMC.JPhiMagestyCarlo
using Test
using Random
using LinearAlgebra

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
    # `keep(s) = abs(J_s) ≥ thr` on both sides reads the same Float64 jphi
    # vector and applies the same `≥ thr` predicate, so the keep mask must
    # be byte-identical. Pin the threshold to an exact |J| from the fixture
    # to lock in the boundary case (the SALC with abs(J) == thr must keep).
    data = SIMPLE.parse_jphi_xml(FEGE_XML)
    abs_J = sort!(abs.(data.jphi))
    thr_boundary = abs_J[3]

    @testset "jphi_threshold parity: matching keep mask at exact |J| boundary" begin
        h_simple = SIMPLE.SpinClusterHamiltonian(FEGE_XML; jphi_threshold = thr_boundary)
        h_opt = OPT.load_sce_hamiltonian(FEGE_XML; jphi_threshold = thr_boundary)
        # Optimized stores jphi directly; Simple stores per-instance J values
        # tiled across the supercell. Compare the underlying SALC count and
        # the surviving |J| set.
        @test length(h_opt.salc_list) == length(h_opt.jphi)
        kept_opt = sort(abs.(h_opt.jphi))
        kept_simple = sort(unique(abs(inst.J) for inst in h_simple.instances))
        # Every surviving Simple |J| must appear in the optimized kept set,
        # and vice versa — they came from the same filter on the same vector.
        @test issubset(Set(kept_simple), Set(kept_opt))
        @test all(j -> j ≥ thr_boundary, kept_opt)
        @test all(j -> j ≥ thr_boundary, kept_simple)
    end

    @testset "jphi_threshold parity: Simple.total_energy ≈ OPT.sce_energy under filter" begin
        # Use a threshold that drops a non-trivial subset; total_energy on
        # the filtered Hamiltonians must agree to standard parity rtol.
        thr = abs_J[2]
        h_simple = SIMPLE.SpinClusterHamiltonian(FEGE_XML; jphi_threshold = thr)
        h_opt = OPT.load_sce_hamiltonian(FEGE_XML; jphi_threshold = thr)
        for seed in (1, 7, 42)
            rng = MersenneTwister(seed)
            spins = _rand_unit_spins(rng, h_simple.n_atoms)
            E_simple = SIMPLE.total_energy(h_simple, spins)
            E_opt = OPT.sce_energy(h_opt, spins)
            @test E_simple ≈ E_opt rtol = 1.0e-8
        end
    end

    @testset "jphi_threshold parity: filtered energy differs from unfiltered" begin
        # Sanity check: the filter must actually change the energy, otherwise
        # the parity check above would pass trivially.
        thr = abs_J[2]
        h_full_simple = SIMPLE.SpinClusterHamiltonian(FEGE_XML)
        h_full_opt = OPT.load_sce_hamiltonian(FEGE_XML)
        h_simple = SIMPLE.SpinClusterHamiltonian(FEGE_XML; jphi_threshold = thr)
        h_opt = OPT.load_sce_hamiltonian(FEGE_XML; jphi_threshold = thr)
        rng = MersenneTwister(123)
        spins = _rand_unit_spins(rng, h_simple.n_atoms)
        E_simple_full = SIMPLE.total_energy(h_full_simple, spins)
        E_opt_full = OPT.sce_energy(h_full_opt, spins)
        E_simple_thr = SIMPLE.total_energy(h_simple, spins)
        E_opt_thr = OPT.sce_energy(h_opt, spins)
        @test E_simple_full != E_simple_thr
        @test E_opt_full != E_opt_thr
    end
else
    @warn "Skipping jphi_threshold parity tests: $FEGE_XML not found"
end
