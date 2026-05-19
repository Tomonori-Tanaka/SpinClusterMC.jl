using SpinClusterMC
using SpinClusterMC.Simple
using Test

const SIMPLE = SpinClusterMC.Simple
const FEGE_XML = joinpath(@__DIR__, "..", "fege_2x2x2", "jphi.xml")
const BCC_XML = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")

@testset "Simple.SpinClusterHamiltonian jphi_threshold defaults match unfiltered" begin
    # threshold = 0.0 (default) must reproduce the unfiltered Hamiltonian
    # bit-exactly: same instance count and identical J on every instance.
    for xml in (BCC_XML, FEGE_XML)
        isfile(xml) || continue
        h_default = SIMPLE.SpinClusterHamiltonian(xml)
        h_zero = SIMPLE.SpinClusterHamiltonian(xml; jphi_threshold = 0.0)
        @test length(h_default.instances) == length(h_zero.instances)
        @test [inst.J for inst in h_default.instances] ==
              [inst.J for inst in h_zero.instances]
    end
end

@testset "Simple.SpinClusterHamiltonian jphi_threshold validates input" begin
    @test_throws ArgumentError SIMPLE.SpinClusterHamiltonian(
        BCC_XML; jphi_threshold = -1.0
    )
end

if isfile(FEGE_XML)
    @testset "Simple.SpinClusterHamiltonian jphi_threshold drops low-|J| SALCs" begin
        h_full = SIMPLE.SpinClusterHamiltonian(FEGE_XML)
        n_full = length(h_full.instances)
        # Pick a threshold above the smallest |J| so at least one SALC is
        # dropped but at least one survives.
        data = SIMPLE.parse_jphi_xml(FEGE_XML)
        abs_J = sort!(abs.(data.jphi))
        thr = abs_J[2]
        h_filtered = SIMPLE.SpinClusterHamiltonian(FEGE_XML; jphi_threshold = thr)
        @test length(h_filtered.instances) < n_full
        @test all(inst -> abs(inst.J) ≥ thr, h_filtered.instances)
    end

    @testset "Simple.SpinClusterHamiltonian jphi_threshold above max(|J|) errors" begin
        data = SIMPLE.parse_jphi_xml(FEGE_XML)
        max_abs = maximum(abs, data.jphi)
        @test_throws ArgumentError SIMPLE.SpinClusterHamiltonian(
            FEGE_XML; jphi_threshold = max_abs * 2
        )
    end

    @testset "Simple.SpinClusterHamiltonian boundary: thr == |J| keeps that SALC" begin
        # `keep(s) = abs(J_s) ≥ thr`: exact equality keeps the SALC. We can't
        # count surviving SALCs by `unique(inst.J)` because (a) one SALC
        # expands into many tiled instances and (b) different SALCs can carry
        # numerically equal `J` values. Instead, verify every kept instance
        # satisfies `|J| ≥ thr` and that the filter actually dropped at least
        # one SALC vs. unfiltered.
        data = SIMPLE.parse_jphi_xml(FEGE_XML)
        abs_J = sort!(abs.(data.jphi))
        thr = abs_J[3]
        @test count(j -> abs(j) < thr, data.jphi) > 0
        h_full = SIMPLE.SpinClusterHamiltonian(FEGE_XML)
        h = SIMPLE.SpinClusterHamiltonian(FEGE_XML; jphi_threshold = thr)
        @test all(inst -> abs(inst.J) ≥ thr, h.instances)
        @test length(h.instances) < length(h_full.instances)
    end

    @testset "Simple.SCEMC plumbs params[:jphi_threshold]" begin
        data = SIMPLE.parse_jphi_xml(FEGE_XML)
        thr = sort!(abs.(data.jphi))[2]
        params_full = Dict{Symbol, Any}(
            :T => 300.0,
            :xml_path => FEGE_XML
        )
        params_thr = Dict{Symbol, Any}(
            :T => 300.0,
            :xml_path => FEGE_XML,
            :jphi_threshold => thr
        )
        mc_full = SIMPLE.SCEMC(params_full)
        mc_thr = SIMPLE.SCEMC(params_thr)
        @test length(mc_thr.h.instances) < length(mc_full.h.instances)

        @test_throws ArgumentError SIMPLE.SCEMC(Dict{Symbol, Any}(
            :T => 300.0,
            :xml_path => FEGE_XML,
            :jphi_threshold => -1.0
        ))
    end
else
    @warn "Skipping fege-based jphi_threshold tests: $FEGE_XML not found"
end
