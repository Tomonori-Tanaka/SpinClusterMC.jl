using SpinClusterMC
using SpinClusterMC.Simple
using Test
using Random
using LinearAlgebra
using StaticArrays: SVector

const SIMPLE = SpinClusterMC.Simple

# Uniform-body fixtures used for the sum(local) = body * total identity.
const UNIFORM_BODY_FIXTURES = [
    (name = "bcc_2x2x2",
        path = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml"),
        body = 2),
    (name = "fege_2x2x2",
        path = joinpath(@__DIR__, "..", "fege_2x2x2", "jphi.xml"),
        body = 2)
]

function _rand_unit_spins(rng, n)
    spins = randn(rng, 3, n)
    for i in 1:n
        spins[:, i] ./= norm(spins[:, i])
    end
    return spins
end

@testset "Simple.total_energy = (1/body) * sum_i local_energy(i) for uniform-body" begin
    for fix in UNIFORM_BODY_FIXTURES
        if !isfile(fix.path)
            @info "Skipping $(fix.name): fixture missing"
            continue
        end
        @testset "$(fix.name)" begin
            h = SIMPLE.SpinClusterHamiltonian(fix.path)
            rng = MersenneTwister(13)
            spins = _rand_unit_spins(rng, h.n_atoms)
            E_total = SIMPLE.total_energy(h, spins)
            sum_local = sum(SIMPLE.local_energy(h, spins, i) for i in 1:h.n_atoms)
            @test sum_local / fix.body ≈ E_total rtol = 1.0e-10
        end
    end
end

@testset "Simple.delta_local_energy = local_after - local_before" begin
    for fix in UNIFORM_BODY_FIXTURES
        if !isfile(fix.path)
            @info "Skipping $(fix.name): fixture missing"
            continue
        end
        @testset "$(fix.name)" begin
            h = SIMPLE.SpinClusterHamiltonian(fix.path)
            rng = MersenneTwister(31)
            spins = _rand_unit_spins(rng, h.n_atoms)
            for i in (1, h.n_atoms ÷ 2, h.n_atoms)
                S_new_raw = randn(rng, 3)
                S_new = S_new_raw / norm(S_new_raw)
                E_old_i = SIMPLE.local_energy(h, spins, i)
                spins_new = copy(spins)
                spins_new[:, i] .= S_new
                E_new_i = SIMPLE.local_energy(h, spins_new, i)
                @test SIMPLE.delta_local_energy(h, spins, i, S_new)≈
                E_new_i-E_old_i rtol=1.0e-12 atol=1.0e-14
            end
        end
    end
end

@testset "Simple.gradient matches central finite differences" begin
    for fix in UNIFORM_BODY_FIXTURES
        if !isfile(fix.path)
            @info "Skipping $(fix.name): fixture missing"
            continue
        end
        @testset "$(fix.name)" begin
            h = SIMPLE.SpinClusterHamiltonian(fix.path)
            rng = MersenneTwister(57)
            spins = _rand_unit_spins(rng, h.n_atoms)
            eps = 1.0e-6
            for i in (2, h.n_atoms - 2)
                g = SIMPLE.gradient(h, spins, i)
                g_fd = zeros(3)
                for axis in 1:3
                    sp_p = copy(spins)
                    sp_p[axis, i] += eps
                    sp_m = copy(spins)
                    sp_m[axis, i] -= eps
                    g_fd[axis] = (SIMPLE.local_energy(h, sp_p, i) -
                                  SIMPLE.local_energy(h, sp_m, i)) / (2 * eps)
                end
                @test maximum(abs.(g .- g_fd)) < 1.0e-7
            end
        end
    end
end

@testset "Simple.total_energy / local_energy argument validation" begin
    h = SIMPLE.SpinClusterHamiltonian(UNIFORM_BODY_FIXTURES[1].path)
    rng = MersenneTwister(0)
    good = _rand_unit_spins(rng, h.n_atoms)
    @test_throws ArgumentError SIMPLE.total_energy(h, good[1:2, :])
    @test_throws ArgumentError SIMPLE.total_energy(h, good[:, 1:(end - 1)])
    @test_throws ArgumentError SIMPLE.local_energy(h, good, 0)
    @test_throws ArgumentError SIMPLE.local_energy(h, good, h.n_atoms + 1)
    @test_throws ArgumentError SIMPLE.delta_local_energy(h, good, 1, [1.0, 0.0])
end
