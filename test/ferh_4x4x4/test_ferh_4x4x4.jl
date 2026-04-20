using SpinClusterMC.JPhiMagestyCarlo
using Carlo
using Test
using Random
using Statistics

const JMCC = SpinClusterMC.JPhiMagestyCarlo
const XML_4x4x4 = joinpath(@__DIR__, "jphi.xml")

# ---------------------------------------------------------------------------
@testset "ferh_4x4x4: load_sce_hamiltonian" begin
    h = load_sce_hamiltonian(XML_4x4x4)
    @test h.n_atoms == 128
    @test h.base_n_atoms == 128
    @test h.repeat == (1, 1, 1)
    @test size(h.pos_frac, 2) == 128
    @test size(h.lattice) == (3, 3)
    @test length(h.jphi) == length(h.salc_list)

    h2 = load_sce_hamiltonian(XML_4x4x4; repeat = (2, 1, 1))
    @test h2.n_atoms == 256
    @test h2.base_n_atoms == 128
end

# ---------------------------------------------------------------------------
# Validate that reference path (sce_energy) and fast path (_energy_from_instances)
# agree for ferh_4x4x4. This is the only ferh test that calls sce_energy; it is
# intentionally slow (~70 s) and serves as a one-time sanity check.
@testset "ferh_4x4x4: reference path agrees with fast path" begin
    h = load_sce_hamiltonian(XML_4x4x4)
    rng = MersenneTwister(42)
    spins = let s = randn(rng, 3, h.n_atoms)
        for i in 1:h.n_atoms; s[:, i] ./= sqrt(sum(s[:, i].^2)); end
        s
    end

    E_ref  = sce_energy(h, spins)
    cache  = JMCC.build_local_energy_cache(h)
    E_fast = h.j0 + JMCC._energy_from_instances(cache.instances, spins)

    @test E_ref ≈ E_fast rtol = 1e-8
end

# ---------------------------------------------------------------------------
# All tests below use only the fast path (_energy_from_instances). sce_energy is
# not called again because reference≈fast has been established above.
# ---------------------------------------------------------------------------

@testset "ferh_4x4x4: SCE interaction energy is extensive for repeat=(2,1,1)" begin
    h1 = load_sce_hamiltonian(XML_4x4x4; repeat = (1, 1, 1))
    h2 = load_sce_hamiltonian(XML_4x4x4; repeat = (2, 1, 1))

    spins1 = zeros(3, h1.n_atoms); spins1[3, :] .= 1.0
    spins2 = zeros(3, h2.n_atoms)
    for ia in 1:h2.n_atoms
        spins2[:, ia] = spins1[:, ((ia - 1) % h1.n_atoms) + 1]
    end

    cache1 = JMCC.build_local_energy_cache(h1)
    cache2 = JMCC.build_local_energy_cache(h2)
    E_int1 = JMCC._energy_from_instances(cache1.instances, spins1)
    E_int2 = JMCC._energy_from_instances(cache2.instances, spins2)

    @test E_int2 ≈ 2 * E_int1 rtol = 1e-8
end

# ---------------------------------------------------------------------------
@testset "ferh_4x4x4: ferromagnetic energy consistent with tiled repeat" begin
    h1 = load_sce_hamiltonian(XML_4x4x4; repeat = (1, 1, 1))
    h2 = load_sce_hamiltonian(XML_4x4x4; repeat = (2, 1, 1))

    spins1 = zeros(3, h1.n_atoms); spins1[3, :] .= 1.0
    spins2 = zeros(3, h2.n_atoms); spins2[3, :] .= 1.0

    cache1 = JMCC.build_local_energy_cache(h1)
    cache2 = JMCC.build_local_energy_cache(h2)
    E_per_atom1 = (h1.j0 * prod(h1.repeat) + JMCC._energy_from_instances(cache1.instances, spins1)) / h1.n_atoms
    E_per_atom2 = (h2.j0 * prod(h2.repeat) + JMCC._energy_from_instances(cache2.instances, spins2)) / h2.n_atoms

    @test E_per_atom1 ≈ E_per_atom2 rtol = 1e-8
end

# ---------------------------------------------------------------------------
@testset "ferh_4x4x4: delta energy consistency" begin
    h = load_sce_hamiltonian(XML_4x4x4)
    rng = MersenneTwister(7)
    spins = let s = randn(rng, 3, h.n_atoms)
        for i in 1:h.n_atoms; s[:, i] ./= sqrt(sum(s[:, i].^2)); end
        s
    end
    cache = JMCC.build_local_energy_cache(h)

    max_l = JMCC._max_l_in_instances(cache.instances)
    zlm   = JMCC._alloc_zlm_cache(h.n_atoms, max_l)
    for ia in 1:h.n_atoms
        JMCC._update_atom_zlm_cache!(zlm, ia, @view(spins[:, ia]), max_l)
    end

    active_body_indices = collect(eachindex(cache.body_list))
    related = JMCC._build_related_instances_by_atom(cache, active_body_indices, h.n_atoms)

    max_sites = JMCC._max_sites_in_instances(cache.instances)
    buf_other = Vector{Int}(undef, max_sites)
    buf_cart  = Vector{Int}(undef, max_sites)

    E0 = h.j0 + JMCC._energy_from_instances(cache.instances, spins)

    for atom in [1, 64, h.n_atoms]
        E_old_local = sum(
            cache.instances[idx].prefactor *
            JMCC._tensor_contract_instance_cached_changed!(
                buf_other, buf_cart, cache.instances[idx], zlm, atom,
            )
            for idx in related[atom]; init = 0.0,
        )

        spins_new = copy(spins)
        sx, sy, sz = JMCC._rand_unit_spin(rng)
        spins_new[1, atom] = sx; spins_new[2, atom] = sy; spins_new[3, atom] = sz

        JMCC._update_atom_zlm_cache!(zlm, atom, @view(spins_new[:, atom]), max_l)

        E_new_local = sum(
            cache.instances[idx].prefactor *
            JMCC._tensor_contract_instance_cached_changed!(
                buf_other, buf_cart, cache.instances[idx], zlm, atom,
            )
            for idx in related[atom]; init = 0.0,
        )

        dE_local = E_new_local - E_old_local
        dE_full  = h.j0 + JMCC._energy_from_instances(cache.instances, spins_new) - E0

        @test dE_local ≈ dE_full rtol = 1e-7

        spins_new[1, atom] = spins[1, atom]
        spins_new[2, atom] = spins[2, atom]
        spins_new[3, atom] = spins[3, atom]
        JMCC._update_atom_zlm_cache!(zlm, atom, @view(spins[:, atom]), max_l)
    end
end
