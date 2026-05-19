using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo
using Serialization
using Test

const OPT = SpinClusterMC.JPhiMagestyCarlo
const FEGE_XML = joinpath(@__DIR__, "..", "fege_2x2x2", "jphi.xml")
const BCC_XML = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")

@testset "OPT.load_sce_hamiltonian jphi_threshold defaults match unfiltered" begin
    # threshold = 0.0 (default) must reproduce the unfiltered Hamiltonian
    # bit-exactly: same SALC count and identical jphi vector.
    for xml in (BCC_XML, FEGE_XML)
        isfile(xml) || continue
        h_default = OPT.load_sce_hamiltonian(xml)
        h_zero = OPT.load_sce_hamiltonian(xml; jphi_threshold = 0.0)
        @test length(h_default.salc_list) == length(h_zero.salc_list)
        @test h_default.jphi == h_zero.jphi
    end
end

@testset "OPT.load_sce_hamiltonian jphi_threshold validates input" begin
    @test_throws ArgumentError OPT.load_sce_hamiltonian(
        BCC_XML; jphi_threshold = -1.0
    )
end

if isfile(FEGE_XML)
    @testset "OPT.load_sce_hamiltonian jphi_threshold drops low-|J| SALCs" begin
        h_full = OPT.load_sce_hamiltonian(FEGE_XML)
        n_full = length(h_full.salc_list)
        abs_J = sort!(abs.(h_full.jphi))
        thr = abs_J[2]
        h_filtered = OPT.load_sce_hamiltonian(FEGE_XML; jphi_threshold = thr)
        @test length(h_filtered.salc_list) < n_full
        @test length(h_filtered.jphi) == length(h_filtered.salc_list)
        @test all(j -> abs(j) ≥ thr, h_filtered.jphi)
    end

    @testset "OPT.load_sce_hamiltonian jphi_threshold above max(|J|) errors" begin
        h_full = OPT.load_sce_hamiltonian(FEGE_XML)
        max_abs = maximum(abs, h_full.jphi)
        @test_throws ArgumentError OPT.load_sce_hamiltonian(
            FEGE_XML; jphi_threshold = max_abs * 2
        )
    end

    @testset "OPT.load_sce_hamiltonian boundary: thr == |J| keeps that SALC" begin
        # `keep(s) = abs(J_s) ≥ thr`: exact equality keeps the SALC.
        h_full = OPT.load_sce_hamiltonian(FEGE_XML)
        abs_J = sort!(abs.(h_full.jphi))
        thr = abs_J[3]
        @test count(j -> abs(j) < thr, h_full.jphi) > 0
        h = OPT.load_sce_hamiltonian(FEGE_XML; jphi_threshold = thr)
        @test all(j -> abs(j) ≥ thr, h.jphi)
        @test length(h.salc_list) < length(h_full.salc_list)
    end

    @testset "OPT.JPhiSpinMC plumbs params[:jphi_threshold]" begin
        h_full = OPT.load_sce_hamiltonian(FEGE_XML)
        thr = sort!(abs.(h_full.jphi))[2]
        params_full = Dict{Symbol, Any}(
            :T => 0.05,
            :xml_path => FEGE_XML,
        )
        params_thr = Dict{Symbol, Any}(
            :T => 0.05,
            :xml_path => FEGE_XML,
            :jphi_threshold => thr,
        )
        mc_full = OPT.JPhiSpinMC(params_full)
        mc_thr = OPT.JPhiSpinMC(params_thr)
        @test mc_full.jphi_threshold == 0.0
        @test mc_thr.jphi_threshold == thr
        @test length(mc_thr.ham.salc_list) < length(mc_full.ham.salc_list)

        @test_throws ArgumentError OPT.JPhiSpinMC(Dict{Symbol, Any}(
            :T => 0.05,
            :xml_path => FEGE_XML,
            :jphi_threshold => -1.0,
        ))
    end

    @testset "OPT.JPhiSpinMC serialize round-trips jphi_threshold" begin
        h_full = OPT.load_sce_hamiltonian(FEGE_XML)
        thr = sort!(abs.(h_full.jphi))[2]
        params = Dict{Symbol, Any}(
            :T => 0.05,
            :xml_path => FEGE_XML,
            :jphi_threshold => thr,
            :energy_kernel => :tensor,
        )
        mc = OPT.JPhiSpinMC(params)
        io = IOBuffer()
        Serialization.serialize(io, mc)
        seekstart(io)
        mc2 = Serialization.deserialize(io)
        @test mc2 isa typeof(mc)
        @test mc2.jphi_threshold == thr
        @test length(mc2.ham.salc_list) == length(mc.ham.salc_list)
    end
else
    @warn "Skipping fege-based optimized jphi_threshold tests: $FEGE_XML not found"
end
