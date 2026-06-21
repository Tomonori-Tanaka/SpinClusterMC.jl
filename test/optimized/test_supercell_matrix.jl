using SpinClusterMC
using Test
using Carlo
import Serialization
using Random: MersenneTwister
using LinearAlgebra: norm

const OPT = SpinClusterMC.JPhiMagestyCarlo

const BCC = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")
const FEGE = joinpath(@__DIR__, "..", "fege_2x2x2", "jphi.xml")
# FeRh has N=3 (triplet) clusters; a single primitive cell (supercell_matrix = I,
# n_atoms = 2) reproduces them under extreme self-overlap, exercising the N=3
# un-fold fast path (`_contract_n3_unfold_changed`) cheaply in the regular suite.
# (BCC/FeGe are N=2 only; the full FeRh supercell parity lives in the slow tests.)
const FERH = joinpath(@__DIR__, "..", "ferh_4x4x4", "jphi.xml")

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
    @test mc.energy_kernel === :tensor_template   # Phase 2: default, un-fold M
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

    # Phase 2: :tensor_template now supports supercell_matrix (un-fold). It must
    # build, run, and stay energy-consistent like the :tensor kernel.
    mctpl = OPT.JPhiSpinMC(Dict{Symbol, Any}(
        :xml_path => BCC, :T => 0.05,
        :supercell_matrix => [2 1 0; 0 2 0; 0 0 2],
        :sweeps => 10, :thermalization => 0, :binsize => 1, :seed => 1,
        :energy_kernel => :tensor_template))
    @test mctpl.energy_kernel === :tensor_template
    ctxtpl = Carlo.MCContext{MersenneTwister}(p)
    Carlo.init!(mctpl, ctxtpl, p)
    @test isapprox(mctpl.energy, OPT.sce_energy(mctpl.ham, mctpl.spins); atol = 1.0e-8)
    for _ in 1:10
        Carlo.sweep!(mctpl, ctxtpl)
    end
    @test isapprox(
        mctpl.energy, OPT.sce_energy(mctpl.ham, mctpl.spins); atol = 1.0e-6, rtol = 1.0e-6)

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
    # (_build_cluster_instances), for several general supercell matrices.
    # (xml, M) pairs: BCC (n_prim=1) and FeGe (n_prim=8, multi-sublattice).
    cases = [
        (BCC, [2 1 0; 0 2 0; 0 0 2]),
        (BCC, [2 0 0; 0 2 0; 0 0 2]),
        (BCC, [3 1 0; 0 1 0; 0 0 2]),
        (FEGE, [1 0 0; 0 1 0; 0 0 2]),
        (FEGE, [1 1 0; 0 2 0; 0 0 1]),
    ]
    # N=3 clusters (single primitive cell). FeRh is the slow-tier fixture, so
    # include it only when present (mirrors the isfile guard in runtests.jl).
    isfile(FERH) && push!(cases, (FERH, [1 0 0; 0 1 0; 0 0 1]))
    for (xml, M) in cases
        h = OPT.load_sce_hamiltonian(xml; supercell_matrix = M)
        templates, related = OPT._build_prim_cluster_templates(h)
        tab = OPT._build_sai_table_cellmajor(templates, related, h)

        # Reconstruct from the per-atom de-duplicated table, grouping by
        # (cbc id, sorted atoms) and summing per-entry prefactors. The grouped
        # prefactors must equal those of the matrix un-fold instances touching i
        # (fold-accumulation preserved, self-overlap repeats collapsed).
        instances = OPT._build_cluster_instances(h)
        for i in 1:h.n_atoms
            tg = Dict{Tuple{UInt, Vector{Int}}, Float64}()
            for ent in tab.entry_off[i]:(tab.entry_off[i + 1] - 1)
                t = templates[tab.entry_tmpl[ent]]
                atoms = tab.sai[tab.sai_off[ent]:(tab.sai_off[ent + 1] - 1)]
                @test i in atoms                       # touched-atom invariant
                key = (t.cbc_id, sort(atoms))
                tg[key] = get(tg, key, 0.0) + t.prefactor
            end
            rg = Dict{Tuple{UInt, Vector{Int}}, Float64}()
            for inst in instances
                i in inst.atoms || continue
                key = (objectid(inst.cbc), sort(inst.atoms))
                rg[key] = get(rg, key, 0.0) + inst.prefactor
            end
            @test keys(tg) == keys(rg)
            for key in keys(tg)
                @test isapprox(tg[key], rg[key]; rtol = 1.0e-12)
            end
        end
    end
end

@testset "tensor_template ≡ tensor under general supercell_matrix" begin
    # P2-M2: the un-fold :tensor_template kernel must reproduce the :tensor kernel
    # bit-for-bit. With identical seed and proposal stream, exact ΔE agreement
    # makes the whole Metropolis trajectory identical (final energy and spins).
    cases = [
        (BCC, [2 1 0; 0 2 0; 0 0 2]),
        (BCC, [1 0 0; 0 1 0; 0 0 1]),      # single primitive cell (extreme self-overlap)
        (BCC, [3 1 0; 0 1 0; 0 0 2]),
        (FEGE, [1 0 0; 0 1 0; 0 0 2]),
        (FEGE, [1 1 0; 0 2 0; 0 0 1]),
    ]
    # N=3 fast path vs generic (self-overlap). FeRh is the slow-tier fixture, so
    # include it only when present (mirrors the isfile guard in runtests.jl).
    isfile(FERH) && push!(cases, (FERH, [1 0 0; 0 1 0; 0 0 1]))
    for (xml, M) in cases
        res = Dict{Symbol, Any}()
        for kern in (:tensor, :tensor_template)
            p = Dict{Symbol, Any}(
                :xml_path => xml, :T => 0.08, :supercell_matrix => M,
                :thermalization => 0, :binsize => 1, :seed => 42,
                :energy_kernel => kern)
            mc = OPT.JPhiSpinMC(p)
            @test mc.energy_kernel === kern
            ctx = Carlo.MCContext{MersenneTwister}(p)
            Carlo.init!(mc, ctx, p)
            # Init energy matches the un-fold reference for both kernels.
            @test isapprox(mc.energy, OPT.sce_energy(mc.ham, mc.spins); atol = 1.0e-8)
            for _ in 1:50
                Carlo.sweep!(mc, ctx)
            end
            res[kern] = (mc.energy, copy(mc.spins))
        end
        Et, St = res[:tensor]
        Etp, Stp = res[:tensor_template]
        @test isapprox(Et, Etp; atol = 1.0e-9, rtol = 1.0e-9)
        @test maximum(norm(St[i] - Stp[i]) for i in eachindex(St)) < 1.0e-10
    end
end

@testset "enabled_bodies filters body sizes identically in both kernels" begin
    # Regression: the :tensor_template kernel used to ignore params[:enabled_bodies]
    # and silently sum all body sizes, diverging from :tensor (which filters). FeRh
    # has mixed N=2/N=3 SALCs; a single primitive cell (M = I, 2 atoms) exercises
    # both. With identical seed the spin stream is independent of kernel/filter, so
    # bit-for-bit ΔE makes the whole trajectory (init energy + spins) identical.
    if isfile(FERH)
        M = [1 0 0; 0 1 0; 0 0 1]
        function run_kernel(kern, eb)
            p = Dict{Symbol, Any}(
                :xml_path => FERH, :T => 0.08, :supercell_matrix => M,
                :thermalization => 0, :binsize => 1, :seed => 7,
                :energy_kernel => kern)
            eb === nothing || (p[:enabled_bodies] = eb)
            mc = OPT.JPhiSpinMC(p)
            ctx = Carlo.MCContext{MersenneTwister}(p)
            Carlo.init!(mc, ctx, p)
            e0 = mc.energy
            for _ in 1:50
                Carlo.sweep!(mc, ctx)
            end
            (e0, mc.energy, copy(mc.spins))
        end

        # Pairs only (N=2): both kernels agree bit-for-bit over the trajectory.
        e0t2, eft2, st2 = run_kernel(:tensor, [2])
        e0p2, efp2, sp2 = run_kernel(:tensor_template, [2])
        @test isapprox(e0t2, e0p2; atol = 1.0e-9, rtol = 1.0e-9)
        @test isapprox(eft2, efp2; atol = 1.0e-9, rtol = 1.0e-9)
        @test maximum(norm(st2[i] - sp2[i]) for i in eachindex(st2)) < 1.0e-10

        # Triplets only (N=3): also bit-for-bit across kernels.
        e0t3, eft3, _ = run_kernel(:tensor, [3])
        e0p3, efp3, _ = run_kernel(:tensor_template, [3])
        @test isapprox(e0t3, e0p3; atol = 1.0e-9, rtol = 1.0e-9)
        @test isapprox(eft3, efp3; atol = 1.0e-9, rtol = 1.0e-9)

        # The filter must actually drop contributions: with identical initial
        # spins, pairs-only + triplets-only init energy sums to the full energy
        # (additive over disjoint body sizes), and neither equals the full energy.
        e0p_all, _, _ = run_kernel(:tensor_template, nothing)
        @test isapprox(e0p2 + e0p3, e0p_all; atol = 1.0e-9, rtol = 1.0e-9)
        @test !isapprox(e0p2, e0p_all; atol = 1.0e-6)
        @test !isapprox(e0p3, e0p_all; atol = 1.0e-6)

        # Serialize round-trip must preserve the body filter: deserialize rebuilds
        # the template via build_local_energy_template(...; enabled_bodies), so the
        # restored energy must equal the pairs-only reference, not the full energy.
        let p = Dict{Symbol, Any}(
                :xml_path => FERH, :T => 0.08, :supercell_matrix => M,
                :sweeps => 5, :thermalization => 0, :binsize => 1, :seed => 7,
                :energy_kernel => :tensor_template, :enabled_bodies => [2])
            mc = OPT.JPhiSpinMC(p)
            ctx = Carlo.MCContext{MersenneTwister}(p)
            Carlo.init!(mc, ctx, p)
            io = IOBuffer()
            Serialization.serialize(io, mc)
            seekstart(io)
            mc2 = Serialization.deserialize(io)
            @test mc2.enabled_bodies == [2]
            @test isapprox(mc2.energy, mc.energy; atol = 1.0e-10)
            @test isapprox(
                mc2.energy, OPT.sce_energy(mc2.ham, mc2.spins; enabled_bodies = [2]);
                atol = 1.0e-9, rtol = 1.0e-9)
            @test !isapprox(mc2.energy, OPT.sce_energy(mc2.ham, mc2.spins); atol = 1.0e-6)
            # Continued sweeps stay consistent with the filtered reference (the
            # rebuilt template's ΔE respects the filter).
            for _ in 1:5
                Carlo.sweep!(mc2, ctx)
            end
            @test isapprox(
                mc2.energy, OPT.sce_energy(mc2.ham, mc2.spins; enabled_bodies = [2]);
                atol = 1.0e-6, rtol = 1.0e-6)
        end

        # Error parity: an unknown body size errors on BOTH kernels (the template
        # kernel must validate just like the :tensor kernel does).
        for kern in (:tensor, :tensor_template)
            pbad = Dict{Symbol, Any}(
                :xml_path => FERH, :T => 0.08, :supercell_matrix => M,
                :thermalization => 0, :binsize => 1, :seed => 7,
                :energy_kernel => kern, :enabled_bodies => [99])
            @test_throws ArgumentError OPT.JPhiSpinMC(pbad)
        end
    end
end

@testset "JPhiSpinMC supercell_matrix serialize round-trip" begin
    # Round-trip both kernels on the un-fold matrix path (deserialize rebuilds
    # the un-fold template for :tensor_template via build_local_energy_template).
    for kern in (:tensor, :tensor_template)
        p = Dict{Symbol, Any}(
            :xml_path => BCC, :T => 0.05,
            :supercell_matrix => [2 1 0; 0 2 0; 0 0 2],
            :sweeps => 5, :thermalization => 0, :binsize => 1, :seed => 3,
            :energy_kernel => kern)
        mc = OPT.JPhiSpinMC(p)
        ctx = Carlo.MCContext{MersenneTwister}(p)
        Carlo.init!(mc, ctx, p)
        io = IOBuffer()
        Serialization.serialize(io, mc)
        seekstart(io)
        mc2 = Serialization.deserialize(io)
        @test mc2.supercell_matrix == mc.supercell_matrix
        @test mc2.ham.n_atoms == mc.ham.n_atoms
        @test mc2.energy_kernel === kern
        @test isapprox(mc2.energy, mc.energy; atol = 1.0e-10)
        @test isapprox(
            OPT.sce_energy(mc2.ham, mc2.spins), OPT.sce_energy(mc.ham, mc.spins);
            atol = 1.0e-10)
        # After deserialization, sweeps continue to stay energy-consistent.
        for _ in 1:5
            Carlo.sweep!(mc2, ctx)
        end
        @test isapprox(
            mc2.energy, OPT.sce_energy(mc2.ham, mc2.spins); atol = 1.0e-6, rtol = 1.0e-6)
    end
end
