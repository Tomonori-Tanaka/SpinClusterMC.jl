using SpinClusterMC
using SpinClusterMC.Simple
using Test
using Random
using LinearAlgebra
using StaticArrays: SVector

const SIMPLE = SpinClusterMC.Simple

function _rand_unit_spins(rng, n)
    spins = randn(rng, 3, n)
    for i in 1:n
        spins[:, i] ./= norm(spins[:, i])
    end
    return spins
end

@testset "Zeeman with UniformMoment" begin
    rng = MersenneTwister(0)
    n_atoms = 8
    spins = _rand_unit_spins(rng, n_atoms)
    B = [0.0, 0.0, 0.3]
    z = SIMPLE.Zeeman(B)

    # total = -Σ B·S for unit moment.
    expected_total = -sum(B[k] * spins[k, i] for k in 1:3 for i in 1:n_atoms)
    @test SIMPLE.total_energy(z, spins) ≈ expected_total rtol = 1.0e-14

    # total = Σ local_energy(i).
    sum_local = sum(SIMPLE.local_energy(z, spins, i) for i in 1:n_atoms)
    @test sum_local ≈ SIMPLE.total_energy(z, spins) rtol = 1.0e-14

    # Local at each site matches the formula.
    for i in 1:n_atoms
        @test SIMPLE.local_energy(z, spins, i) ≈
              -dot(B, spins[:, i]) rtol = 1.0e-14
    end

    # Gradient is -B everywhere when m = 1.
    for i in 1:n_atoms
        @test SIMPLE.gradient(z, spins, i) ≈ -SVector{3}(B)
    end

    # Linearity in B: doubling field doubles energy.
    z2 = SIMPLE.Zeeman(2 .* B)
    @test SIMPLE.total_energy(z2, spins) ≈ 2 * SIMPLE.total_energy(z, spins)
end

@testset "Zeeman with PerSiteMoment" begin
    rng = MersenneTwister(1)
    n_atoms = 6
    spins = _rand_unit_spins(rng, n_atoms)
    B = [0.1, -0.2, 0.5]
    moments = [3.0, 1.0, 3.0, 1.0, 3.0, 1.0]  # alternating Fe / Rh-like (FM FeRh values)
    z = SIMPLE.Zeeman(B; moments = SIMPLE.PerSiteMoment(moments))

    # total = -Σ m_i (B·S_i)
    expected_total = -sum(moments[i] * dot(B, spins[:, i]) for i in 1:n_atoms)
    @test SIMPLE.total_energy(z, spins) ≈ expected_total rtol = 1.0e-14

    # total = Σ local
    sum_local = sum(SIMPLE.local_energy(z, spins, i) for i in 1:n_atoms)
    @test sum_local ≈ SIMPLE.total_energy(z, spins) rtol = 1.0e-14

    # local at each site
    for i in 1:n_atoms
        @test SIMPLE.local_energy(z, spins, i) ≈
              -moments[i] * dot(B, spins[:, i]) rtol = 1.0e-14
    end

    # gradient = -m_i · B
    for i in 1:n_atoms
        @test SIMPLE.gradient(z, spins, i) ≈ -moments[i] .* SVector{3}(B)
    end
end

@testset "Zeeman.delta_local_energy = local_after - local_before" begin
    rng = MersenneTwister(2)
    n_atoms = 5
    spins = _rand_unit_spins(rng, n_atoms)
    B = [0.4, 0.0, -0.1]
    moments = [1.0, 0.7, 1.3, 0.9, 1.1]

    for z in (
        SIMPLE.Zeeman(B),
        SIMPLE.Zeeman(B; moments = SIMPLE.UniformMoment(0.8)),
        SIMPLE.Zeeman(B; moments = SIMPLE.PerSiteMoment(moments))
    )
        for i in (1, 3, n_atoms)
            S_new_raw = randn(rng, 3)
            S_new = S_new_raw ./ norm(S_new_raw)
            E_before = SIMPLE.local_energy(z, spins, i)
            spins_new = copy(spins)
            spins_new[:, i] .= S_new
            E_after = SIMPLE.local_energy(z, spins_new, i)
            @test SIMPLE.delta_local_energy(z, spins, i, S_new)≈
            E_after-E_before rtol=1.0e-13 atol=1.0e-15
        end
    end
end

@testset "Zeeman gradient matches central finite differences" begin
    rng = MersenneTwister(3)
    n_atoms = 4
    spins = _rand_unit_spins(rng, n_atoms)
    B = [0.05, 0.1, -0.2]
    moments = [1.0, 1.5, 2.0, 0.5]
    z = SIMPLE.Zeeman(B; moments = SIMPLE.PerSiteMoment(moments))
    eps = 1.0e-6
    for i in 1:n_atoms
        g = SIMPLE.gradient(z, spins, i)
        g_fd = zeros(3)
        for axis in 1:3
            sp = copy(spins)
            sp[axis, i] += eps
            Ep = SIMPLE.local_energy(z, sp, i)
            sp[axis, i] -= 2eps
            Em = SIMPLE.local_energy(z, sp, i)
            g_fd[axis] = (Ep - Em) / (2eps)
        end
        @test maximum(abs.(g .- g_fd)) < 1.0e-9
    end
end

@testset "Zeeman composes additively with SCE Hamiltonian" begin
    xml = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")
    if !isfile(xml)
        @info "Skipping composite test: bcc fixture missing"
    else
        rng = MersenneTwister(123)
        h = SIMPLE.SpinClusterHamiltonian(xml)
        spins = _rand_unit_spins(rng, h.n_atoms)
        z = SIMPLE.Zeeman([0.0, 0.0, 0.3])

        E_sce = SIMPLE.total_energy(h, spins)
        E_ext = SIMPLE.total_energy(z, spins)
        E_full = E_sce + E_ext

        # Verify total parity via summing per-atom contributions on the
        # external side and per-cluster on the SCE side; the two sums add.
        sum_local_ext = sum(SIMPLE.local_energy(z, spins, i) for i in 1:h.n_atoms)
        @test sum_local_ext ≈ E_ext rtol = 1.0e-14

        # Single-flip ΔE: full = SCE part + Zeeman part.
        i = 3
        S_new_raw = randn(rng, 3)
        S_new = S_new_raw ./ norm(S_new_raw)
        ΔE_sce = SIMPLE.delta_local_energy(h, spins, i, S_new)
        ΔE_ext = SIMPLE.delta_local_energy(z, spins, i, S_new)
        spins_new = copy(spins)
        spins_new[:, i] .= S_new
        E_full_new = SIMPLE.total_energy(h, spins_new) +
                     SIMPLE.total_energy(z, spins_new)
        @test ΔE_sce + ΔE_ext ≈ E_full_new - E_full rtol = 1.0e-10
    end
end

@testset "Zeeman: unit=:tesla converts to eV/μ_B internally" begin
    # 1 Tesla along +z, paired with a 1 μ_B uniform moment, should give
    # E_per_atom = -μ_B · B = -5.7883818060e-5 eV per spin perfectly aligned.
    rng = MersenneTwister(7)
    n_atoms = 4
    spins = _rand_unit_spins(rng, n_atoms)

    field_T = [0.0, 0.0, 1.0]
    z_T = SIMPLE.Zeeman(field_T; unit = :tesla)
    z_eV = SIMPLE.Zeeman(
        field_T .* SIMPLE.BOHR_MAGNETON_EV_PER_TESLA; unit = :eV_per_muB
    )

    # The :tesla path is mathematically equivalent to pre-converting and
    # passing :eV_per_muB (which is the default).
    @test z_T.field ≈ z_eV.field rtol = 1.0e-14
    @test SIMPLE.total_energy(z_T, spins) ≈
          SIMPLE.total_energy(z_eV, spins) rtol = 1.0e-14

    # Fully aligned spins with 1 μ_B moment in 1 T field: E = -μ_B · 1 T per
    # atom = -BOHR_MAGNETON_EV_PER_TESLA per atom.
    aligned = zeros(3, n_atoms)
    aligned[3, :] .= 1.0
    @test SIMPLE.total_energy(z_T, aligned) ≈
          -n_atoms * SIMPLE.BOHR_MAGNETON_EV_PER_TESLA rtol = 1.0e-14

    # Fe / Rh sublattices: per-atom contribution scales with the moment
    # (using FM FeRh experimental values, 3.0 μ_B / 1.0 μ_B).
    moments = [3.0, 1.0, 3.0, 1.0]
    z_sub = SIMPLE.Zeeman(
        [0.0, 0.0, 1.0]; unit = :tesla, moments = SIMPLE.PerSiteMoment(moments)
    )
    expected = -SIMPLE.BOHR_MAGNETON_EV_PER_TESLA * sum(moments)
    @test SIMPLE.total_energy(z_sub, aligned) ≈ expected rtol = 1.0e-14
end

@testset "Zeeman / MomentModel argument validation" begin
    rng = MersenneTwister(4)
    spins = _rand_unit_spins(rng, 4)

    # Field length must be 3.
    @test_throws ArgumentError SIMPLE.Zeeman([0.0, 1.0])
    @test_throws ArgumentError SIMPLE.Zeeman([0.0, 1.0, 0.0, 2.0])

    # PerSiteMoment length must match spins.
    z_bad = SIMPLE.Zeeman([0.0, 0.0, 0.1]; moments = SIMPLE.PerSiteMoment([1.0, 2.0]))
    @test_throws ArgumentError SIMPLE.total_energy(z_bad, spins)
    @test_throws ArgumentError SIMPLE.local_energy(z_bad, spins, 1)

    # Spin matrix must have 3 rows.
    z = SIMPLE.Zeeman([0.0, 0.0, 0.1])
    @test_throws ArgumentError SIMPLE.total_energy(z, spins[1:2, :])

    # Atom index out of range.
    @test_throws ArgumentError SIMPLE.local_energy(z, spins, 0)
    @test_throws ArgumentError SIMPLE.local_energy(z, spins, 5)
    @test_throws ArgumentError SIMPLE.gradient(z, spins, 0)
    @test_throws ArgumentError SIMPLE.gradient(z, spins, 5)

    # S_new must have length 3.
    @test_throws ArgumentError SIMPLE.delta_local_energy(z, spins, 1, [1.0, 0.0])

    # Invalid unit kwarg.
    @test_throws ArgumentError SIMPLE.Zeeman([0.0, 0.0, 1.0]; unit = :gauss)
end
