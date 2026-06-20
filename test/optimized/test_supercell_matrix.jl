using SpinClusterMC
using Test
using Carlo
import Serialization
using Random: MersenneTwister
using LinearAlgebra: norm

const OPT = SpinClusterMC.JPhiMagestyCarlo

const BCC = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")

_ferro(n) = repeat(Float64[0, 0, 1], 1, n)

@testset "load_sce_hamiltonian supercell_matrix" begin
    h = OPT.load_sce_hamiltonian(BCC; supercell_matrix = [2 1 0; 0 2 0; 0 0 2])
    @test h.supercell_matrix == [2 1 0; 0 2 0; 0 0 2]
    @test h.prim !== nothing
    @test h.repeat == (0, 0, 0)            # sentinel for the matrix path
    @test h.n_atoms == 8                   # n_prim(=1) * |det| (=8)
    # Energy is finite and intensive (matches the base cell per-atom value).
    e_base = OPT.sce_energy(
        OPT.load_sce_hamiltonian(BCC; repeat = (1, 1, 1)), _ferro(16)) / 16
    @test isapprox(OPT.sce_energy(h, _ferro(8)) / 8, e_base; atol = 1.0e-8)
end

@testset "load_sce_hamiltonian supercell_matrix errors" begin
    @test_throws ArgumentError OPT.load_sce_hamiltonian(BCC; supercell_matrix = [1 0; 0 1])
    @test_throws ArgumentError OPT.load_sce_hamiltonian(
        BCC; supercell_matrix = [1 0 0; 0 1 0; 0 0 0])     # singular
    @test_throws ArgumentError OPT.load_sce_hamiltonian(
        BCC; repeat = (2, 1, 1), supercell_matrix = [1 0 0; 0 1 0; 0 0 1])
end

@testset "JPhiSpinMC supercell_matrix: kernel, sweep, errors" begin
    p = Dict{Symbol, Any}(
        :xml_path => BCC, :T => 0.05,
        :supercell_matrix => [2 1 0; 0 2 0; 0 0 2],
        :sweeps => 10, :thermalization => 0, :binsize => 1, :seed => 1)
    mc = OPT.JPhiSpinMC(p)
    @test mc.energy_kernel === :tensor         # forced for the matrix path
    @test mc.supercell_matrix == [2 1 0; 0 2 0; 0 0 2]
    @test mc.ham.n_atoms == 8

    ctx = Carlo.MCContext{MersenneTwister}(p)
    Carlo.init!(mc, ctx, p)
    @test all(i -> isapprox(norm(mc.spins[i]), 1.0; atol = 1.0e-12), 1:mc.ham.n_atoms)
    @test isapprox(mc.energy, OPT.sce_energy(mc.ham, mc.spins); atol = 1.0e-8)
    for _ in 1:10
        Carlo.sweep!(mc, ctx)
    end
    @test isapprox(mc.energy, OPT.sce_energy(mc.ham, mc.spins); atol = 1.0e-6, rtol = 1.0e-6)

    # :tensor_template is not supported with supercell_matrix (Phase 1).
    @test_throws ArgumentError OPT.JPhiSpinMC(Dict{Symbol, Any}(
        :xml_path => BCC, :T => 0.05,
        :supercell_matrix => [1 0 0; 0 1 0; 0 0 1],
        :energy_kernel => :tensor_template))

    # Base-cell tiling of initial_spins is rejected on the matrix path.
    pbad = Dict{Symbol, Any}(
        :xml_path => BCC, :T => 0.05,
        :supercell_matrix => [2 0 0; 0 2 0; 0 0 2],
        :thermalization => 0, :binsize => 1, :seed => 1,
        :initial_spins => zeros(3, 16))
    mcbad = OPT.JPhiSpinMC(pbad)
    ctxbad = Carlo.MCContext{MersenneTwister}(pbad)
    @test_throws ArgumentError Carlo.init!(mcbad, ctxbad, pbad)
end

@testset "JPhiSpinMC supercell_matrix serialize round-trip" begin
    p = Dict{Symbol, Any}(
        :xml_path => BCC, :T => 0.05,
        :supercell_matrix => [2 1 0; 0 2 0; 0 0 2],
        :sweeps => 5, :thermalization => 0, :binsize => 1, :seed => 3)
    mc = OPT.JPhiSpinMC(p)
    ctx = Carlo.MCContext{MersenneTwister}(p)
    Carlo.init!(mc, ctx, p)
    io = IOBuffer()
    Serialization.serialize(io, mc)
    seekstart(io)
    mc2 = Serialization.deserialize(io)
    @test mc2.supercell_matrix == mc.supercell_matrix
    @test mc2.ham.n_atoms == mc.ham.n_atoms
    @test mc2.energy_kernel === :tensor
    @test isapprox(mc2.energy, mc.energy; atol = 1.0e-10)
    @test isapprox(
        OPT.sce_energy(mc2.ham, mc2.spins), OPT.sce_energy(mc.ham, mc.spins);
        atol = 1.0e-10)
end
