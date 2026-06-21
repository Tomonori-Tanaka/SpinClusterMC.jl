using SpinClusterMC
using SpinClusterMC.Simple
using Test
using Random
using LinearAlgebra
using StaticArrays: SVector

const SIMPLE = SpinClusterMC.Simple

@testset "Simple._rand_unit_spin" begin
    rng = MersenneTwister(0)
    for _ in 1:200
        u = SIMPLE._rand_unit_spin(rng)
        @test u isa SVector{3, Float64}
        @test norm(u) ≈ 1.0 rtol = 1.0e-12
    end

    # Rough sanity: the mean of many samples is close to zero.
    rng = MersenneTwister(1)
    n = 5_000
    s = SVector{3, Float64}(0.0, 0.0, 0.0)
    for _ in 1:n
        s = s + SIMPLE._rand_unit_spin(rng)
    end
    mean_norm = norm(s ./ n)
    @test mean_norm < 0.05
end

@testset "Simple._propose_spin_geodesic stays on the sphere" begin
    rng = MersenneTwister(2)
    for _ in 1:50
        u = SIMPLE._rand_unit_spin(rng)
        for θmax in (0.05, 0.3, 1.0, π)
            v = SIMPLE._propose_spin_geodesic(rng, u, θmax)
            @test v isa SVector{3, Float64}
            @test norm(v) ≈ 1.0 rtol = 1.0e-12
        end
    end
end

@testset "Simple._propose_spin_geodesic with θmax=0 is identity" begin
    rng = MersenneTwister(3)
    for _ in 1:20
        u = SIMPLE._rand_unit_spin(rng)
        v = SIMPLE._propose_spin_geodesic(rng, u, 0.0)
        @test v ≈ u rtol = 1.0e-12
    end
end

@testset "Simple._propose_spin_geodesic step size respects θmax" begin
    # Angle between u and u' is at most θmax (modulo round-off).
    rng = MersenneTwister(4)
    θmax = 0.1
    for _ in 1:200
        u = SIMPLE._rand_unit_spin(rng)
        v = SIMPLE._propose_spin_geodesic(rng, u, θmax)
        # Skip the rare uniform-fallback branch (norm of tangent ~ 0).
        c = clamp(dot(u, v), -1.0, 1.0)
        if c > -0.5  # i.e., not the fallback uniform draw
            θ = acos(c)
            @test θ ≤ θmax + 1.0e-12
        end
    end
end

@testset "Simple.init_spins(:random)" begin
    spins = SIMPLE.init_spins(:random, 7, 7; rng = MersenneTwister(0))
    @test size(spins) == (3, 7)
    @test eltype(spins) === Float64
    for i in 1:7
        @test norm(spins[:, i]) ≈ 1.0 rtol = 1.0e-12
    end

    # Reproducible with the same rng seed.
    a = SIMPLE.init_spins(:random, 5, 5; rng = MersenneTwister(99))
    b = SIMPLE.init_spins(:random, 5, 5; rng = MersenneTwister(99))
    @test a == b
end

@testset "Simple.init_spins(:ferromagnetic)" begin
    spins = SIMPLE.init_spins(:ferromagnetic, 4, 4)
    @test size(spins) == (3, 4)
    @test all(spins[3, :] .== 1.0)
    @test all(spins[1, :] .== 0.0)
    @test all(spins[2, :] .== 0.0)
end

@testset "Simple.init_spins with Tuple / Vector / SVector direction" begin
    # Tuple
    spins = SIMPLE.init_spins((1.0, 0.0, 0.0), 3, 3)
    @test all(spins[1, :] .≈ 1.0)
    @test all(spins[2:3, :] .≈ 0.0)

    # Non-normalized vector — get normalized on the way in.
    spins = SIMPLE.init_spins([2.0, 0.0, 0.0], 3, 3)
    @test all(spins[1, :] .≈ 1.0)

    # SVector route.
    spins = SIMPLE.init_spins(SVector{3, Float64}(0.0, 0.0, 3.0), 2, 2)
    @test all(spins[3, :] .≈ 1.0)
end

@testset "Simple.init_spins with Matrix (full supercell config)" begin
    # Phase 2: only a full 3 × n_atoms config is accepted (primitive cell-major
    # order); base-cell tiling (3 × base_n replicated) is no longer supported.
    super = [1.0 0.0 0.5 -0.5; 0.0 1.0 0.5 -0.5; 0.0 0.0 sqrt(0.5) -sqrt(0.5)]
    n_atoms = 4
    spins = SIMPLE.init_spins(super, n_atoms, 2)
    @test size(spins) == (3, n_atoms)
    for i in 1:n_atoms
        @test norm(spins[:, i]) ≈ 1.0 rtol = 1.0e-12
    end
    # A base-cell-sized matrix (ncols = base_n ≠ n_atoms) is rejected.
    base = [1.0 0.0; 0.0 0.0; 0.0 2.0]
    @test_throws ArgumentError SIMPLE.init_spins(base, 6, 2)
end

@testset "Simple.init_spins(::AbstractDict) reads :initial_spins" begin
    @test SIMPLE.init_spins(Dict{Symbol, Any}(), 3, 3; rng = MersenneTwister(0)) ==
          SIMPLE.init_spins(:random, 3, 3; rng = MersenneTwister(0))
    @test SIMPLE.init_spins(Dict{Symbol, Any}(:initial_spins => :ferromagnetic), 3, 3) ==
          SIMPLE.init_spins(:ferromagnetic, 3, 3)
    @test SIMPLE.init_spins(
        Dict{Symbol, Any}(:initial_spins => (1.0, 0.0, 0.0)), 4, 4
    ) == SIMPLE.init_spins((1.0, 0.0, 0.0), 4, 4)
end

@testset "Simple.init_spins argument validation" begin
    # Bad symbol.
    @test_throws ArgumentError SIMPLE.init_spins(:antiferromagnetic, 4, 4)
    # Wrong tuple / vector length.
    @test_throws ArgumentError SIMPLE.init_spins((1.0, 0.0), 4, 4)
    @test_throws ArgumentError SIMPLE.init_spins([1.0, 0.0, 0.0, 0.0], 4, 4)
    # Wrong matrix row count.
    @test_throws ArgumentError SIMPLE.init_spins(zeros(2, 4), 4, 4)
    # Matrix column count not matching base or supercell.
    @test_throws ArgumentError SIMPLE.init_spins(zeros(3, 5), 4, 2)
    # Zero direction → can't normalize.
    @test_throws ArgumentError SIMPLE.init_spins((0.0, 0.0, 0.0), 3, 3)
    @test_throws ArgumentError SIMPLE.init_spins([0.0 1.0; 0.0 0.0; 0.0 0.0], 4, 2)
end
