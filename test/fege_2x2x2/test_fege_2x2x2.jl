using SpinClusterMC.JPhiMagestyCarlo
using Carlo
using Test
using Random

const JMCC = SpinClusterMC.JPhiMagestyCarlo
const XML_FEGE = joinpath(@__DIR__, "jphi.xml")

# ---------------------------------------------------------------------------
@testset "fege_2x2x2: load_sce_hamiltonian" begin
    h = load_sce_hamiltonian(XML_FEGE)
    @test h.n_atoms == 64
    @test h.base_n_atoms == 64
    @test h.repeat == (1, 1, 1)
    @test size(h.pos_frac, 2) == 64
    @test size(h.lattice) == (3, 3)
    @test length(h.jphi) == length(h.salc_list)

    h2 = load_sce_hamiltonian(XML_FEGE; repeat = (2, 1, 1))
    @test h2.n_atoms == 128
    @test h2.base_n_atoms == 64
end

# ---------------------------------------------------------------------------
@testset "fege_2x2x2: sce_energy reference vs fast" begin
    rng = MersenneTwister(11)
    h = load_sce_hamiltonian(XML_FEGE)
    spins = let s = randn(rng, 3, h.n_atoms)
        for i in 1:h.n_atoms; s[:, i] ./= sqrt(sum(s[:, i].^2)); end
        s
    end

    E_ref  = sce_energy(h, spins)
    cache  = JMCC.build_local_energy_cache(h)
    E_fast = JMCC._energy_from_instances(cache.instances, spins)

    @test E_ref ≈ E_fast rtol = 1e-8
end

# ---------------------------------------------------------------------------
@testset "fege_2x2x2: delta energy consistency" begin
    h = load_sce_hamiltonian(XML_FEGE)
    rng = MersenneTwister(13)
    spins = let s = randn(rng, 3, h.n_atoms)
        for i in 1:h.n_atoms; s[:, i] ./= sqrt(sum(s[:, i].^2)); end
        s
    end
    cache = JMCC.build_local_energy_cache(h)

    max_l = JMCC._max_l_in_instances(cache.instances)
    zlm   = JMCC._alloc_zlm_cache(h.n_atoms, max_l)
    sph   = JMCC.SphericalHarmonics(max_l)
    for ia in 1:h.n_atoms
        JMCC._update_atom_zlm_cache!(zlm, ia, @view(spins[:, ia]), sph)
    end

    active_body_indices = collect(eachindex(cache.body_list))
    related = JMCC._build_related_instances_by_atom(cache, active_body_indices, h.n_atoms)

    max_sites = JMCC._max_sites_in_instances(cache.instances)
    buf_other = Vector{Int}(undef, max_sites)
    buf_cart  = Vector{Int}(undef, max_sites)

    E0 = JMCC._energy_from_instances(cache.instances, spins)

    for atom in [1, 32, h.n_atoms]
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

        JMCC._update_atom_zlm_cache!(zlm, atom, @view(spins_new[:, atom]), sph)

        E_new_local = sum(
            cache.instances[idx].prefactor *
            JMCC._tensor_contract_instance_cached_changed!(
                buf_other, buf_cart, cache.instances[idx], zlm, atom,
            )
            for idx in related[atom]; init = 0.0,
        )

        dE_local = E_new_local - E_old_local
        dE_full  = JMCC._energy_from_instances(cache.instances, spins_new) - E0

        @test dE_local ≈ dE_full rtol = 1e-7

        spins_new[1, atom] = spins[1, atom]
        spins_new[2, atom] = spins[2, atom]
        spins_new[3, atom] = spins[3, atom]
        JMCC._update_atom_zlm_cache!(zlm, atom, @view(spins[:, atom]), sph)
    end
end
