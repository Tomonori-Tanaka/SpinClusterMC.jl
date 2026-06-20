using SpinClusterMC
using Test
using Carlo
import Serialization
using Random: MersenneTwister
using LinearAlgebra: norm

const OPT = SpinClusterMC.JPhiMagestyCarlo

const BCC = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")
const FEGE = joinpath(@__DIR__, "..", "fege_2x2x2", "jphi.xml")

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

@testset "primitive cell-major template SAI matches matrix instances" begin
    # P2-M1: the un-fold template construction (_build_prim_cluster_templates +
    # _build_sai_table_cellmajor) must reconstruct exactly the same set of
    # supercell cluster instances as the :tensor un-fold reference
    # (_build_cluster_instances_matrix), for several general supercell matrices.
    # (xml, M) pairs: BCC (n_prim=1) and FeGe (n_prim=8, multi-sublattice).
    cases = [
        (BCC, [2 1 0; 0 2 0; 0 0 2]),
        (BCC, [2 0 0; 0 2 0; 0 0 2]),
        (BCC, [3 1 0; 0 1 0; 0 0 2]),
        (FEGE, [1 0 0; 0 1 0; 0 0 2]),
        (FEGE, [1 1 0; 0 2 0; 0 0 1]),
    ]
    for (xml, M) in cases
        h = OPT.load_sce_hamiltonian(xml; supercell_matrix = M)
        templates, related = OPT._build_prim_cluster_templates(h)
        flat, offsets = OPT._build_sai_table_cellmajor(templates, related, h)
        n_prim = h.prim.n_prim

        # Reconstruct (cbc id, sorted atoms) from the cell-major SAI table.
        tmpl_set = Set{Tuple{UInt, Vector{Int}}}()
        for i in 1:h.n_atoms
            subl = ((i - 1) % n_prim) + 1
            pos = offsets[i] - 1
            for rc in related[subl]
                t = templates[rc.inst_idx]
                N = length(t.site_subl)
                atoms = Int[flat[pos + k] for k in 1:N]
                pos += N
                @test atoms[rc.pivot_k] == i          # pivot site lands on atom i
                push!(tmpl_set, (t.cbc_id, sort(atoms)))
            end
        end

        # Reference set from the matrix un-fold instance list.
        ref_set = Set{Tuple{UInt, Vector{Int}}}()
        for inst in OPT._build_cluster_instances_matrix(h)
            push!(ref_set, (objectid(inst.cbc), sort(inst.atoms)))
        end

        @test tmpl_set == ref_set
    end
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
