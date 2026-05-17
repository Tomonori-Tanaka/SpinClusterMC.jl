using SpinClusterMC
using SpinClusterMC.Simple
using Carlo
using Test
using Random
using LinearAlgebra: norm
using StaticArrays: SVector

const SIMPLE = SpinClusterMC.Simple
const BCC_XML = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")

# Convenience builders for the canonical Kelvin-input params + an MCContext.
function _build_simple_params(; T_K = 300.0, kwargs...)
    return Dict{Symbol, Any}(
        :T => T_K,
        :xml_path => BCC_XML,
        :thermalization => 0,
        :binsize => 1,
        :seed => 42,
        kwargs...
    )
end
_build_ctx(params) = Carlo.MCContext{MersenneTwister}(params)

if isfile(BCC_XML)
    @testset "SCEMC constructor: defaults" begin
        params = _build_simple_params(; T_K = 300.0)
        mc = SIMPLE.SCEMC(params)
        # 300 K * 8.6173e-5 ≈ 0.025852 eV.
        @test mc.T ≈ 300.0 * SIMPLE.BOLTZMANN_EV_PER_KELVIN rtol = 1.0e-14
        @test mc.h.n_atoms == 16
        @test mc.external === nothing
        @test mc.theta_max ≈ Float64(π)
        @test mc.renorm_every == 1000
        @test mc.sweep_count == 0
        @test mc.repeat == (1, 1, 1)
        @test mc.xml_path == BCC_XML
    end

    @testset "SCEMC constructor: user-supplied params are reflected" begin
        cb_measure_called = Ref(0)
        cb_eval_called = Ref(0)
        cb_measure = (mc, ctx) -> (cb_measure_called[] += 1; nothing)
        cb_eval = (eval, params) -> (cb_eval_called[] += 1; nothing)
        zeeman = SIMPLE.Zeeman([0.0, 0.0, 0.1])
        params = _build_simple_params(;
            T_K = 500.0,
            repeat = (2, 1, 1),
            external = zeeman,
            spin_theta_max = 0.3,
            renorm_every = 250,
            update_scheme = :metropolis,
            extra_measure = cb_measure,
            extra_evaluables = cb_eval
        )
        mc = SIMPLE.SCEMC(params)

        # Temperature: 500 K -> internal eV.
        @test mc.T ≈ 500.0 * SIMPLE.BOLTZMANN_EV_PER_KELVIN rtol = 1.0e-14
        # Geometry: repeat=(2,1,1) doubles the atom count of the base XML.
        @test mc.repeat == (2, 1, 1)
        @test mc.h.n_atoms == 32
        @test mc.h.base_n_atoms == 16
        # External term: stored by identity.
        @test mc.external === zeeman
        # Proposal / housekeeping.
        @test mc.theta_max == 0.3
        @test mc.renorm_every == 250
        # Callbacks: stored by identity (compared via the call counters below).
        @test mc.extra_measure === cb_measure
        @test mc.extra_evaluables === cb_eval
        # Provenance for future PT serialize.
        @test mc.xml_path == BCC_XML

        # The callbacks fire on Carlo.measure!.
        ctx = _build_ctx(params)
        Carlo.init!(mc, ctx, params)
        Carlo.measure!(mc, ctx)
        @test cb_measure_called[] == 1
    end

    @testset "SCEMC constructor: required params and validation" begin
        @test_throws ArgumentError SIMPLE.SCEMC(Dict{Symbol, Any}(:T => 100.0))
        @test_throws ArgumentError SIMPLE.SCEMC(
            Dict{Symbol, Any}(:xml_path => BCC_XML)
        )
        # Negative / zero temperature.
        @test_throws ArgumentError SIMPLE.SCEMC(_build_simple_params(; T_K = 0.0))
        @test_throws ArgumentError SIMPLE.SCEMC(_build_simple_params(; T_K = -10.0))
        # Wrong external type.
        @test_throws ArgumentError SIMPLE.SCEMC(
            _build_simple_params(; external = "not an external term")
        )
        # Unsupported update scheme.
        @test_throws ArgumentError SIMPLE.SCEMC(
            _build_simple_params(; update_scheme = :heatbath)
        )
    end

    @testset "Carlo.init!: spins are unit vectors; energy matches recompute" begin
        params = _build_simple_params(;
            T_K = 300.0, initial_spins = :ferromagnetic, renorm_every = 5
        )
        mc = SIMPLE.SCEMC(params)
        ctx = _build_ctx(params)
        Carlo.init!(mc, ctx, params)

        @test mc.sweep_count == 0
        for i in 1:mc.h.n_atoms
            @test norm(mc.spins[:, i]) ≈ 1.0 rtol = 1.0e-12
        end
        @test mc.energy ≈ SIMPLE.total_energy(mc.h, mc.spins) rtol = 1.0e-13
    end

    @testset "Carlo.sweep! runs and tracks energy incrementally" begin
        params = _build_simple_params(;
            T_K = 300.0,
            initial_spins = :ferromagnetic,
            spin_theta_max = 0.3,
            renorm_every = 5
        )
        mc = SIMPLE.SCEMC(params)
        ctx = _build_ctx(params)
        Carlo.init!(mc, ctx, params)
        E0 = mc.energy
        for _ in 1:25
            Carlo.sweep!(mc, ctx)
        end
        @test mc.sweep_count == 25
        # The drift check at sweep 5, 10, 15, 20, 25 must have passed; if it
        # had not, Carlo.sweep! would have errored on its own.
        @test isfinite(mc.energy)
        # The drift check itself reconciles mc.energy with a full recompute.
        @test mc.energy ≈ SIMPLE.total_energy(mc.h, mc.spins) rtol = 1.0e-13
    end

    @testset "Carlo.measure! does not throw on a fresh context" begin
        params = _build_simple_params(;
            T_K = 300.0, initial_spins = :ferromagnetic
        )
        mc = SIMPLE.SCEMC(params)
        ctx = _build_ctx(params)
        Carlo.init!(mc, ctx, params)
        Carlo.measure!(mc, ctx)
        # Sanity: the per-atom energy is what mc.energy records over n_atoms.
        @test mc.energy / mc.h.n_atoms ≈
              SIMPLE.total_energy(mc.h, mc.spins) / mc.h.n_atoms
    end

    @testset "SCEMC + Zeeman: total energy and ΔE include the external term" begin
        # Ferromagnetic init: every spin is +ẑ. The SCE energy is fixed,
        # and the entire dependence on the +z field comes from the Zeeman
        # term itself.
        zeeman = SIMPLE.Zeeman([0.0, 0.0, 0.1])  # 0.1 eV/μ_B
        params = _build_simple_params(;
            T_K = 100.0,
            initial_spins = :ferromagnetic,
            external = zeeman,
            renorm_every = 10
        )
        mc = SIMPLE.SCEMC(params)
        ctx = _build_ctx(params)
        Carlo.init!(mc, ctx, params)

        # Energy is SCE part + Zeeman part.
        E_sce = SIMPLE.total_energy(mc.h, mc.spins)
        E_zeeman = SIMPLE.total_energy(zeeman, mc.spins)
        @test mc.energy ≈ E_sce + E_zeeman rtol = 1.0e-13

        # A few sweeps; drift check at sweep 10 verifies the incremental
        # accounting handles the additive external term.
        for _ in 1:15
            Carlo.sweep!(mc, ctx)
        end
        @test mc.energy ≈
              SIMPLE.total_energy(mc.h, mc.spins) +
              SIMPLE.total_energy(zeeman, mc.spins) rtol = 1.0e-13
    end

    @testset "Carlo.register_evaluables runs without error" begin
        params = _build_simple_params(; T_K = 300.0)
        # Build an empty Evaluator. The observables it would normally know
        # about (`:Energy`, `:Energy2`, …) are absent, so Carlo will warn and
        # skip the registration internally; the call itself must still
        # succeed without throwing.
        eval = Carlo.Evaluator(Dict{Symbol, Carlo.ResultObservable}(), false)
        @test Carlo.register_evaluables(SIMPLE.SCEMC, eval, params) === nothing
    end
else
    @warn "Skipping SCEMC tests: bcc fixture missing"
end
