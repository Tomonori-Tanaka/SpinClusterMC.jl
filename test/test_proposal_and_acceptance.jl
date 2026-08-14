# Regression tests for the spin-proposal distribution and the Metropolis
# acceptance counters. Covers both backends: `JPhiMagestyCarlo` (tuple-returning
# `_propose_spin_geodesic`, `JPhiSpinMC`) and `Simple` (SVector-returning
# `_propose_spin_geodesic`, `SCEMC`).
#
# Oracles are analytic, not captured output. The proposal is specified to be
# uniform on the spherical cap of half-angle θmax about the current spin, so
# `c = u·u'` must satisfy `c ~ U(cos θmax, 1)`, whose moments are closed-form:
#
#     E[c]    = 1 - w/2                 with w = 1 - cos θmax
#     Var(c)  = w²/12
#     E[c²]   = (1 + a + a²)/3          with a = cos θmax
#
# and, for a uniform draw, the sampling errors of the estimators are
#
#     σ(mean)     = sqrt(Var(c)/N)  = w / sqrt(12 N)
#     σ(variance) = sqrt((μ₄ - σ⁴)/N), μ₄ = w⁴/80, σ⁴ = w⁴/144
#                 = w² / sqrt(180 N)
#
# Each gate below states its σ and the headroom multiple it allows.
#
# The pre-2026-08 implementation drew `θ ~ U(-θmax, θmax)` instead, which puts a
# `1/sin θ` density on the sphere. Measured against that implementation (200k
# draws, same seeds), the gates below fail by:
#
#     Var(c), θmax = π          251σ   (E[c²] = 1/2 instead of 1/3)
#     mean(c), θmax = 0.3       255σ
#     CDF of c, θmax = π         86σ   (P(c > 1/2) = 1/3 instead of 1/4)
#     <n nᵀ> at θmax = π      0.107 deviation vs the 0.006 tolerance (18×)
#
# so they resolve the defect with margin rather than being fitted to whatever
# the fixed seed happens to show. Both moments are needed: at θmax = π the two
# distributions share the mean (0) and differ only from the variance up.

using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo
using SpinClusterMC.Simple
using Carlo
using Test
using Random
using Statistics
using LinearAlgebra
using StaticArrays: SVector

const JMCC_P = SpinClusterMC.JPhiMagestyCarlo
const SIMPLE_P = SpinClusterMC.Simple
const BCC_XML_P = joinpath(@__DIR__, "bcc_2x2x2", "jphi.xml")

# The two backends spell the same proposal differently; wrap both so a single
# oracle can be applied to each.
_propose_optimized(rng, u, θmax) =
    SVector{3,Float64}(JMCC_P._propose_spin_geodesic(rng, u[1], u[2], u[3], θmax))
_propose_simple(rng, u, θmax) = SIMPLE_P._propose_spin_geodesic(rng, u, θmax)

const PROPOSERS = (
    ("optimized", _propose_optimized),
    ("simple", _propose_simple),
)

@testset "spin proposal is uniform on the cap of half-angle θmax" begin
    N = 200_000
    # Deliberately not only ẑ: the proposal must depend on (u, u') through their
    # angle alone, so every reference direction has to give the same moments.
    references = (
        SVector(0.0, 0.0, 1.0),
        SVector(1.0, 0.0, 0.0),
        normalize(SVector(1.0, 1.0, 1.0)),
    )

    for (name, propose) in PROPOSERS, θmax in (0.05, 0.3, 1.0, Float64(π))
        a = cos(θmax)
        w = 1.0 - a
        mean_exact = 1.0 - 0.5 * w
        var_exact = w * w / 12
        σ_mean = w / sqrt(12 * N)
        σ_var = w * w / sqrt(180 * N)

        for (k, u) in enumerate(references)
            rng = MersenneTwister(1000 + k)
            c = Vector{Float64}(undef, N)
            for j in 1:N
                v = propose(rng, u, θmax)
                @assert isapprox(norm(v), 1.0; atol = 1e-12)
                c[j] = clamp(dot(u, v), -1.0, 1.0)
            end

            # cos θ ~ U(cos θmax, 1): first two moments, each with 8σ of headroom.
            @test isapprox(mean(c), mean_exact; atol = 8 * σ_mean)
            @test isapprox(var(c), var_exact; atol = 8 * σ_var)

            # The whole CDF, not just two moments: `c` is uniform on `[a, 1]`, so
            # `P(c > x) = (1 - x)/w` is linear. Checking it at the quartiles
            # catches a wrong *shape* that happens to match a moment — which is
            # exactly the pre-fix failure mode at θmax = π, where the mean agrees
            # (both are 0) and only the shape differs. Each point is a Bernoulli
            # trial, σ = sqrt(p(1-p)/N); 8σ of headroom again.
            for q in (0.25, 0.5, 0.75)
                x = a + q * w
                p = 1.0 - q                       # P(c > x)
                σ_p = sqrt(p * (1 - p) / N)
                @test isapprox(count(>(x), c) / N, p; atol = 8 * σ_p)
            end

            # Support: the cap is closed at θmax and reaches its far edge.
            @test minimum(c) ≥ a - 1e-12
            @test maximum(c) ≤ 1.0 + 1e-12
            @test minimum(c) < a + 0.02 * w   # not concentrated away from the rim
        end
    end
end

@testset "spin proposal is isotropic about the current spin" begin
    # <u'> = E[c] · u exactly: the transverse part averages to zero because the
    # tangent is isotropic. This gates the azimuthal marginal, which the polar
    # moments above cannot see.
    N = 200_000
    for (name, propose) in PROPOSERS, θmax in (0.3, Float64(π))
        a = cos(θmax)
        w = 1.0 - a
        mean_c = 1.0 - 0.5 * w
        mean_c2 = (1 + a + a * a) / 3
        σ_par = sqrt(w * w / 12 / N)              # component along u
        σ_perp = sqrt((1 - mean_c2) / 2 / N)      # each transverse component

        rng = MersenneTwister(2024)
        u = normalize(SVector(0.3, -0.5, 0.81))
        acc = SVector(0.0, 0.0, 0.0)
        for _ in 1:N
            acc = acc + propose(rng, u, θmax)
        end
        m = acc / N
        # Split into the component along u and the transverse remainder so each
        # can carry its own σ; 8σ of headroom on both.
        par = dot(m, u)
        perp = m - par * u
        @test isapprox(par, mean_c; atol = 8 * σ_par)
        @test norm(perp) ≤ 8 * sqrt(2) * σ_perp
    end
end

@testset "θmax = π is exactly the uniform-sphere proposal" begin
    # <n nᵀ> = I/3 holds for any isotropic distribution of unit vectors and for no
    # anisotropic one, so it is the sharpest single statement of "uniform on S²".
    # Var(n_z²) = 1/5 - 1/9 = 4/45 gives σ = sqrt(4/45)/sqrt(N) ≈ 6.7e-4 at
    # N = 200_000; atol = 0.006 is ~9σ.
    #
    # For a cap-symmetric proposal <n nᵀ> = E[c²] u uᵀ + (E[s²]/2)(I - u uᵀ). The
    # pre-fix draw had E[c²] = 1/2 at θmax = π instead of 1/3, giving
    # 0.25 I + 0.25 u uᵀ; for the tilted `u` below the largest entry-wise
    # deviation from I/3 is 0.107, i.e. 18× this tolerance (measured).
    N = 200_000
    u = normalize(SVector(1.0, -2.0, 0.5))

    # `_rand_unit_spin` is the independent uniform-sphere draw the optimized
    # backend uses when :spin_theta_max is absent; include it so the test also
    # pins that θmax = π and the default agree in distribution (the report's
    # cross-backend inconsistency).
    samplers = (
        ("optimized θmax=π", (rng) -> _propose_optimized(rng, u, Float64(π))),
        ("simple θmax=π", (rng) -> _propose_simple(rng, u, Float64(π))),
        ("_rand_unit_spin", (rng) -> SVector{3,Float64}(JMCC_P._rand_unit_spin(rng))),
    )

    for (k, (name, draw)) in enumerate(samplers)
        rng = MersenneTwister(3000 + k)
        C = zeros(3, 3)
        for _ in 1:N
            n = draw(rng)
            C .+= n * n'
        end
        C ./= N
        @test isapprox(C, Matrix(I / 3, 3, 3); atol = 0.006)
    end
end

@testset "θmax = 0 leaves the spin untouched" begin
    # Exact, not statistical: the Simple proposal short-circuits at θmax == 0, and
    # the cap formula would return u anyway (d = 0 ⇒ c = 1, s = 0).
    rng = MersenneTwister(5)
    for _ in 1:50
        u = SIMPLE_P._rand_unit_spin(rng)
        @test _propose_simple(rng, u, 0.0) === u
    end
end

@testset "θmax above π is rejected" begin
    # Only cos(θmax) enters the cap draw, so 3π/2 would silently propose a
    # *narrower* cap than π. Both constructors must refuse it rather than
    # quietly sample the wrong distribution.
    if isfile(BCC_XML_P)
        base = Dict{Symbol,Any}(
            :T => 300.0, :xml_path => BCC_XML_P,
            :thermalization => 0, :binsize => 1, :seed => 42,
        )
        @test_throws ArgumentError SIMPLE_P.SCEMC(
            merge(base, Dict{Symbol,Any}(:spin_theta_max => 1.5π)))
        @test_throws ArgumentError JMCC_P.JPhiSpinMC(
            merge(base, Dict{Symbol,Any}(:T => 0.02, :spin_theta_max => 1.5π)))
        # π itself stays valid on both.
        @test SIMPLE_P.SCEMC(
            merge(base, Dict{Symbol,Any}(:spin_theta_max => Float64(π)))).theta_max ≈ π
        @test JMCC_P.JPhiSpinMC(
            merge(base, Dict{Symbol,Any}(:T => 0.02, :spin_theta_max => Float64(π)))
        ).spin_theta_max ≈ π
    end
end

# ---------------------------------------------------------------------------
# Acceptance counters
# ---------------------------------------------------------------------------

if isfile(BCC_XML_P)
    function _simple_mc(; T_K, θmax, repeat = (2, 2, 2), seed = 7,
            initial_spins = :ferromagnetic)
        params = Dict{Symbol,Any}(
            :T => T_K, :xml_path => BCC_XML_P, :repeat => repeat,
            :thermalization => 0, :binsize => 1, :seed => seed,
            :spin_theta_max => θmax, :initial_spins => initial_spins,
        )
        mc = SIMPLE_P.SCEMC(params)
        ctx = Carlo.MCContext{MersenneTwister}(params)
        Carlo.init!(mc, ctx, params)
        return mc, ctx, params
    end

    function _optimized_mc(; T_eV, θmax, repeat = (2, 2, 2), seed = 7)
        params = Dict{Symbol,Any}(
            :T => T_eV, :xml_path => BCC_XML_P, :repeat => repeat,
            :thermalization => 0, :binsize => 1, :seed => seed,
            :spin_theta_max => θmax,
        )
        mc = JMCC_P.JPhiSpinMC(params)
        ctx = Carlo.MCContext{MersenneTwister}(params)
        Carlo.init!(mc, ctx, params)
        return mc, ctx, params
    end

    @testset "acceptance counters track the sweep exactly" begin
        # One sweep is defined as n_atoms flip attempts, so n_proposed must be
        # exactly n_atoms per sweep — an identity, not an estimate.
        for nsweeps in (1, 5)
            mc, ctx, _ = _simple_mc(; T_K = 300.0, θmax = 0.3)
            @test mc.n_proposed == 0     # init! opens the window
            @test mc.n_accepted == 0
            @test isnan(SIMPLE_P.acceptance_rate(mc))
            for _ in 1:nsweeps
                Carlo.sweep!(mc, ctx)
            end
            @test mc.n_proposed == nsweeps * mc.h.n_atoms
            @test 0 ≤ mc.n_accepted ≤ mc.n_proposed
            @test SIMPLE_P.acceptance_rate(mc) == mc.n_accepted / mc.n_proposed

            mco, ctxo, _ = _optimized_mc(; T_eV = 0.02585, θmax = 0.3)
            for _ in 1:nsweeps
                Carlo.sweep!(mco, ctxo)
            end
            @test mco.n_proposed == nsweeps * mco.ham.n_atoms
            @test 0 ≤ mco.n_accepted ≤ mco.n_proposed
        end
    end

    @testset "acceptance is exactly 1 when every move is free" begin
        # Two independent routes to a hand-derivable acceptance of exactly 1.0:
        #
        # (a) θmax = 0 makes the proposal the identity, so ΔE is bitwise 0.0 and
        #     the `ΔE ≤ 0` branch is taken every time (Simple only — the
        #     optimized constructor requires θmax > 0).
        # (b) T → ∞ makes exp(-ΔE/T) evaluate to exactly 1.0 in Float64 once
        #     |ΔE|/T < 1.1e-16, and rand() ∈ [0,1) is always < 1.0.
        mc, ctx, _ = _simple_mc(; T_K = 300.0, θmax = 0.0)
        for _ in 1:10
            Carlo.sweep!(mc, ctx)
        end
        @test SIMPLE_P.acceptance_rate(mc) == 1.0

        mc, ctx, _ = _simple_mc(; T_K = 1.0e30, θmax = Float64(π))
        for _ in 1:10
            Carlo.sweep!(mc, ctx)
        end
        @test SIMPLE_P.acceptance_rate(mc) == 1.0

        mco, ctxo, _ = _optimized_mc(; T_eV = 1.0e30, θmax = Float64(π))
        for _ in 1:10
            Carlo.sweep!(mco, ctxo)
        end
        @test JMCC_P.acceptance_rate(mco) == 1.0
    end

    @testset "reset_acceptance! opens a new window" begin
        mc, ctx, _ = _simple_mc(; T_K = 300.0, θmax = 0.3)
        for _ in 1:3
            Carlo.sweep!(mc, ctx)
        end
        @test mc.n_proposed > 0
        SIMPLE_P.reset_acceptance!(mc)
        @test mc.n_proposed == 0 && mc.n_accepted == 0
        @test isnan(SIMPLE_P.acceptance_rate(mc))
        Carlo.sweep!(mc, ctx)
        @test mc.n_proposed == mc.h.n_atoms

        mco, ctxo, _ = _optimized_mc(; T_eV = 0.02585, θmax = 0.3)
        Carlo.sweep!(mco, ctxo)
        JMCC_P.reset_acceptance!(mco)
        @test mco.n_proposed == 0 && mco.n_accepted == 0
        @test isnan(JMCC_P.acceptance_rate(mco))
    end

    @testset ":AcceptanceRate matches the accessor and closes the window" begin
        # The two exposed surfaces — the Carlo observable and the accessor — must
        # report the same number for the same window, or a tuning loop reading one
        # and a report reading the other would disagree. `binsize = 1` puts each
        # recorded value in its own bin, so bins[1] is exactly what measure! saw.
        function _check(mc, ctx, rate, n_atoms)
            for _ in 1:4
                Carlo.sweep!(mc, ctx)
            end
            @test mc.n_proposed == 4 * n_atoms
            expected = rate(mc)
            Carlo.measure!(mc, ctx)
            @test ctx.measure.observables[:AcceptanceRate].bins[1] == expected
            # measure! records the window it just closed, then reopens it.
            @test mc.n_proposed == 0 && mc.n_accepted == 0
            @test isnan(rate(mc))
            # The next window is independent of the one already reported.
            Carlo.sweep!(mc, ctx)
            @test mc.n_proposed == n_atoms
            @test rate(mc) == mc.n_accepted / n_atoms
        end

        mc, ctx, _ = _simple_mc(; T_K = 300.0, θmax = 0.3)
        _check(mc, ctx, SIMPLE_P.acceptance_rate, mc.h.n_atoms)

        mco, ctxo, _ = _optimized_mc(; T_eV = 0.02585, θmax = 0.3)
        _check(mco, ctxo, JMCC_P.acceptance_rate, mco.ham.n_atoms)
    end

    @testset "acceptance collapses at low T for a whole-sphere proposal" begin
        # The end-to-end statement the counter exists to support: that
        # :spin_theta_max is a live knob whose effect the counter can see, which
        # is what the reporting campaign had no way to check. Physics oracle: at
        # 25 K, kT = 2.15 meV against a single-spin reorientation cost of order
        # tens of meV, so a proposal drawn uniformly over the whole sphere is
        # essentially never accepted, while a 0.05 rad nudge usually is.
        # Acceptance must therefore be strongly decreasing in θmax.
        #
        # This gate is about the counter, not about the proposal shape: on this
        # fixture the pre-fix proposal also passes it (0.497 / 0.088 / 0.024 /
        # 0.007 across the four θmax, measured). The distributional defect is
        # caught by the moment and CDF gates above, which kill it by ≥ 86σ; the
        # bounds here are loose on purpose so they track the physics rather than
        # a seed.
        rates = [
            (θ, SIMPLE_P.acceptance_rate(
                let (mc, ctx, _) = _simple_mc(; T_K = 25.0, θmax = θ)
                    for _ in 1:200
                        Carlo.sweep!(mc, ctx)
                    end
                    mc
                end))
            for θ in (0.05, 0.3, 1.0, Float64(π))
        ]
        r = last.(rates)
        @test issorted(r; rev = true)          # monotone in θmax
        @test r[1] > 0.2                       # small steps are often accepted
        @test r[end] < 0.05                    # whole-sphere steps are not
        @test r[1] / r[end] > 20               # the knob visibly moves acceptance
    end
end
