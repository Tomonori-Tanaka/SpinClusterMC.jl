using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo
using Test
using Random
using LinearAlgebra

const JMCC = SpinClusterMC.JPhiMagestyCarlo
const XML = joinpath(@__DIR__, "..", "examples", "bccFe", "metropolis", "jphi.xml")

# ---------------------------------------------------------------------------
# Helper: random unit spin matrix
# ---------------------------------------------------------------------------
function rand_unit_spins(rng, n)
    spins = randn(rng, 3, n)
    for i in 1:n
        spins[:, i] ./= norm(spins[:, i])
    end
    return spins
end

@testset "SpinClusterMC.jl" begin

    # -----------------------------------------------------------------------
    @testset "supercell_atom_index" begin
        # repeat=(1,1,1): only tile (0,0,0) → same as base_atom
        @test supercell_atom_index(1, 0, 0, 0, 4, (1, 1, 1)) == 1
        @test supercell_atom_index(4, 0, 0, 0, 4, (1, 1, 1)) == 4

        # repeat=(2,1,1): 8 atoms; tile (1,0,0) → base + 4
        @test supercell_atom_index(1, 0, 0, 0, 4, (2, 1, 1)) == 1
        @test supercell_atom_index(1, 1, 0, 0, 4, (2, 1, 1)) == 5
        @test supercell_atom_index(3, 1, 0, 0, 4, (2, 1, 1)) == 7

        # repeat=(2,2,2): 32 atoms
        n = supercell_atom_index(1, 1, 1, 1, 4, (2, 2, 2))
        @test 1 ≤ n ≤ 32

        # out-of-range tile → error
        @test_throws ArgumentError supercell_atom_index(1, 2, 0, 0, 4, (2, 1, 1))
        # out-of-range base_atom → error
        @test_throws ArgumentError supercell_atom_index(5, 0, 0, 0, 4, (2, 1, 1))
    end

    # -----------------------------------------------------------------------
    @testset "_min_image_frac" begin
        f = JMCC._min_image_frac
        @test f([0.0, 0.0, 0.0]) ≈ [0.0, 0.0, 0.0]
        @test f([0.6, 0.0, 0.0]) ≈ [-0.4, 0.0, 0.0]
        @test f([-0.6, 0.0, 0.0]) ≈ [0.4, 0.0, 0.0]
        @test all(abs.(f([1.2, -0.8, 0.5])) .<= 0.5 + 1e-12)
    end

    # -----------------------------------------------------------------------
    @testset "_compute_instance_strides" begin
        f = JMCC._compute_instance_strides
        # single site l=1: dims=[3], strides=[1,3]
        s = f([1])
        @test s == [1, 3]
        # two sites l=1,l=2: dims=[3,5], strides=[1,3,15]
        s = f([1, 2])
        @test s == [1, 3, 15]
        # three sites l=0,l=0,l=1: dims=[1,1,3], strides=[1,1,1,3]
        s = f([0, 0, 1])
        @test s == [1, 1, 1, 3]
    end

    # -----------------------------------------------------------------------
    @testset "_cluster_scaling" begin
        @test JMCC._cluster_scaling(2) ≈ 4π
        @test JMCC._cluster_scaling(4) ≈ (4π)^2
        @test JMCC._cluster_scaling(1) ≈ sqrt(4π)
    end

    # -----------------------------------------------------------------------
    @testset "_zlm_col" begin
        # l=0, m_idx=1 → 0
        @test JMCC._zlm_col(0, 1) == 1
        # l=1, m_idx=1 → 1^2+1=2
        @test JMCC._zlm_col(1, 1) == 2
        @test JMCC._zlm_col(1, 3) == 4
        # l=2, m_idx=1 → 4+1=5
        @test JMCC._zlm_col(2, 1) == 5
        @test JMCC._zlm_col(2, 5) == 9
    end

    # -----------------------------------------------------------------------
    @testset "spin proposals" begin
        rng = MersenneTwister(42)
        N = 1000

        @testset "_rand_unit_spin" begin
            for _ in 1:N
                sx, sy, sz = JMCC._rand_unit_spin(rng)
                @test sx^2 + sy^2 + sz^2 ≈ 1.0 atol=1e-12
            end
        end

        @testset "_propose_spin_geodesic" begin
            theta_max = 0.5
            for _ in 1:N
                ux, uy, uz = JMCC._rand_unit_spin(rng)
                nx, ny, nz = JMCC._propose_spin_geodesic(rng, ux, uy, uz, theta_max)
                # unit norm
                @test nx^2 + ny^2 + nz^2 ≈ 1.0 atol=1e-12
                # angle ≤ theta_max
                cosθ = clamp(nx * ux + ny * uy + nz * uz, -1.0, 1.0)
                @test acos(cosθ) ≤ theta_max + 1e-12
            end
        end
    end

    # -----------------------------------------------------------------------
    # Tests requiring the bcc Fe XML
    # -----------------------------------------------------------------------
    if isfile(XML)

        @testset "load_sce_hamiltonian" begin
            h = load_sce_hamiltonian(XML)
            @test h.n_atoms == h.base_n_atoms  # repeat=(1,1,1)
            @test h.n_atoms > 0
            @test size(h.pos_frac, 2) == h.n_atoms
            @test size(h.lattice) == (3, 3)
            @test length(h.jphi) == length(h.salc_list)

            h2 = load_sce_hamiltonian(XML; repeat = (2, 1, 1))
            @test h2.n_atoms == 2 * h.n_atoms
            @test h2.base_n_atoms == h.base_n_atoms
        end

        @testset "sce_energy reference vs fast" begin
            # sce_energy uses coupled_cluster_energy (reference);
            # _energy_from_instances uses _tensor_contract_instance (fast path).
            # They should agree on any spin configuration.
            rng = MersenneTwister(1)
            h = load_sce_hamiltonian(XML)
            spins = rand_unit_spins(rng, h.n_atoms)

            E_ref = sce_energy(h, spins)

            cache = JMCC.build_local_energy_cache(h)
            E_fast = h.j0 + JMCC._energy_from_instances(cache.instances, spins)

            @test E_ref ≈ E_fast rtol=1e-8
        end

        @testset "cached tensor contraction matches reference" begin
            # _tensor_contract_instance_cached should equal _tensor_contract_instance
            # for all instances on random spins.
            rng = MersenneTwister(2)
            h = load_sce_hamiltonian(XML)
            spins = rand_unit_spins(rng, h.n_atoms)
            cache = JMCC.build_local_energy_cache(h)

            max_l = JMCC._max_l_in_instances(cache.instances)
            zlm = JMCC._alloc_zlm_cache(h.n_atoms, max_l)
            for ia in 1:h.n_atoms
                JMCC._update_atom_zlm_cache!(zlm, ia, @view(spins[:, ia]), max_l)
            end

            for inst in cache.instances
                ref = JMCC._tensor_contract_instance(inst.cbc, inst.atoms, spins)
                cached = JMCC._tensor_contract_instance_cached(inst, zlm)
                @test ref ≈ cached rtol=1e-10
            end
        end

        @testset "changed-atom contraction matches full cached" begin
            # _tensor_contract_instance_cached_changed! should equal
            # _tensor_contract_instance_cached regardless of which atom is labelled changed.
            rng = MersenneTwister(3)
            h = load_sce_hamiltonian(XML)
            spins = rand_unit_spins(rng, h.n_atoms)
            cache = JMCC.build_local_energy_cache(h)

            max_l = JMCC._max_l_in_instances(cache.instances)
            zlm = JMCC._alloc_zlm_cache(h.n_atoms, max_l)
            for ia in 1:h.n_atoms
                JMCC._update_atom_zlm_cache!(zlm, ia, @view(spins[:, ia]), max_l)
            end

            max_sites = JMCC._max_sites_in_instances(cache.instances)
            buf_other = Vector{Int}(undef, max_sites)
            buf_cart = Vector{Int}(undef, max_sites)

            for inst in cache.instances[1:min(20, end)]
                expected = JMCC._tensor_contract_instance_cached(inst, zlm)
                for changed_atom in inst.atoms
                    got = JMCC._tensor_contract_instance_cached_changed!(
                        buf_other, buf_cart, inst, zlm, changed_atom,
                    )
                    @test got ≈ expected rtol=1e-10
                end
            end
        end

        @testset "delta energy consistency" begin
            # Flip one spin; delta energy from local instances should equal
            # the change in full sce_energy.
            rng = MersenneTwister(4)
            h = load_sce_hamiltonian(XML)
            spins = rand_unit_spins(rng, h.n_atoms)
            cache = JMCC.build_local_energy_cache(h)

            max_l = JMCC._max_l_in_instances(cache.instances)
            zlm = JMCC._alloc_zlm_cache(h.n_atoms, max_l)
            for ia in 1:h.n_atoms
                JMCC._update_atom_zlm_cache!(zlm, ia, @view(spins[:, ia]), max_l)
            end

            # active: all body indices
            active_body_indices = collect(eachindex(cache.body_list))
            related = JMCC._build_related_instances_by_atom(cache, active_body_indices, h.n_atoms)

            max_sites = JMCC._max_sites_in_instances(cache.instances)
            buf_other = Vector{Int}(undef, max_sites)
            buf_cart = Vector{Int}(undef, max_sites)

            E0 = sce_energy(h, spins)

            for atom in [1, 3, h.n_atoms]
                # old local energy
                E_old_local = sum(
                    cache.instances[idx].prefactor *
                    JMCC._tensor_contract_instance_cached_changed!(
                        buf_other, buf_cart, cache.instances[idx], zlm, atom,
                    )
                    for idx in related[atom];
                    init = 0.0,
                )

                # flip spin
                spins_new = copy(spins)
                sx, sy, sz = JMCC._rand_unit_spin(rng)
                spins_new[1, atom] = sx; spins_new[2, atom] = sy; spins_new[3, atom] = sz

                # update zlm
                JMCC._update_atom_zlm_cache!(zlm, atom, @view(spins_new[:, atom]), max_l)

                E_new_local = sum(
                    cache.instances[idx].prefactor *
                    JMCC._tensor_contract_instance_cached_changed!(
                        buf_other, buf_cart, cache.instances[idx], zlm, atom,
                    )
                    for idx in related[atom];
                    init = 0.0,
                )

                dE_local = E_new_local - E_old_local
                dE_full = sce_energy(h, spins_new) - E0

                @test dE_local ≈ dE_full rtol=1e-7

                # restore for next iteration
                spins_new[1, atom] = spins[1, atom]
                spins_new[2, atom] = spins[2, atom]
                spins_new[3, atom] = spins[3, atom]
                JMCC._update_atom_zlm_cache!(zlm, atom, @view(spins[:, atom]), max_l)
            end
        end

        @testset "supercell interaction energy scales linearly" begin
            # Ferromagnetic config (all spins identical).
            # j0 is a fixed constant and does not scale with the supercell.
            # The interaction part (E - j0) must scale exactly with n_atoms
            # because the (2,1,1) tiling produces exactly 2× the cluster instances,
            # each with the same spin values.
            h1 = load_sce_hamiltonian(XML; repeat = (1, 1, 1))
            h2 = load_sce_hamiltonian(XML; repeat = (2, 1, 1))

            spin_vec = [1.0, 0.0, 0.0]
            spins1 = repeat(spin_vec, 1, h1.n_atoms)
            spins2 = repeat(spin_vec, 1, h2.n_atoms)

            E_int1 = sce_energy(h1, spins1) - h1.j0
            E_int2 = sce_energy(h2, spins2) - h2.j0

            @test E_int2 ≈ 2 * E_int1 rtol=1e-8
        end

    else
        @warn "Skipping XML-dependent tests: $XML not found"
    end

end
