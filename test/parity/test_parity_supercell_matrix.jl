using SpinClusterMC
using SpinClusterMC.Simple
using Test
using StaticArrays: SMatrix
using Random: MersenneTwister, randn!
using LinearAlgebra: norm

const SIMPLE = SpinClusterMC.Simple
const OPT = SpinClusterMC.JPhiMagestyCarlo

# Simple matrix-path Hamiltonian for a given supercell matrix.
function _simple_matrix_ham(path, Md)
    data = SIMPLE.parse_jphi_xml(path)
    prim = SIMPLE.extract_primitive(data.system)
    tpls = SIMPLE.build_templates(data.salcs, data.jphi, data.system, prim)
    inst, na = SIMPLE._generate_instances_matrix(tpls, prim, SMatrix{3, 3, Int}(Md))
    cg = SIMPLE.build_cg_table(data.salcs)
    return SIMPLE.SpinClusterHamiltonian(
        na, data.system.n_atoms, (0, 0, 0), inst, cg,
        SIMPLE._max_l_in_instances(inst),
        SIMPLE._build_atom_to_instance_indices(inst, na))
end

function _rand_spins(n, rng)
    s = randn!(rng, Matrix{Float64}(undef, 3, n))
    for i in 1:n
        s[:, i] ./= norm(@view s[:, i])
    end
    return s
end

# Both engines recover the same primitive cell and number atoms primitive
# cell-major via the shared `SupercellCommon`, so the atom indices coincide and
# even random configurations can be compared column-for-column.
@testset "optimized ↔ Simple supercell_matrix parity" begin
    fixtures = [("bcc_2x2x2", joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml"))]
    if "slow" in ARGS
        push!(fixtures,
            ("fege_2x2x2", joinpath(@__DIR__, "..", "fege_2x2x2", "jphi.xml")))
    end
    matrices = [
        [2 0 0; 0 2 0; 0 0 2],     # base-multiple-equivalent (diagonal)
        [3 0 0; 0 2 0; 0 0 2],     # non-base-multiple
        [2 1 0; 0 2 0; 0 0 2]     # non-diagonal
    ]
    for (name, path) in fixtures
        if !isfile(path)
            @info "Skipping $name: fixture missing"
            continue
        end
        @testset "$name" begin
            for Md in matrices
                sh = _simple_matrix_ham(path, Md)
                oh = OPT.load_sce_hamiltonian(path; supercell_matrix = Md)
                @test oh.n_atoms == sh.n_atoms
                rng = MersenneTwister(0x5ce)
                for trial in 1:3
                    spins = trial == 1 ? repeat(Float64[0, 0, 1], 1, sh.n_atoms) :
                            _rand_spins(sh.n_atoms, rng)
                    es = SIMPLE.total_energy(sh, spins)
                    eo = OPT.sce_energy(oh, spins)
                    @test isapprox(es, eo; atol = 1.0e-7, rtol = 1.0e-7)
                end
            end
        end
    end
end

# Map each atom of `hm` (matrix path) to the atom of `hl` (legacy path) at the
# same fractional position. Both share the lattice `primitive * (reshape_base *
# diag)`, so positions coincide; only the numbering differs.
function _position_perm(hm, hl)
    n = hm.n_atoms
    key(p) = (round(Int, mod(p[1], 1.0) * 1_000_000),
        round(Int, mod(p[2], 1.0) * 1_000_000),
        round(Int, mod(p[3], 1.0) * 1_000_000))
    legacy_by_pos = Dict(key(hl.pos_frac[:, j]) => j for j in 1:n)
    perm = Vector{Int}(undef, n)
    for i in 1:n
        perm[i] = legacy_by_pos[key(hm.pos_frac[:, i])]
    end
    @assert sort(perm) == collect(1:n)
    return perm
end

# Phase 2: `repeat` is now sugar for `supercell_matrix = reshape_base * diag(n)`,
# so the diagonal `repeat` path and the equivalent matrix path are the SAME
# un-fold model. They must agree for ANY configuration at every n (not just for a
# ferromagnet). Both paths even share the same cell-major numbering, so the same
# spin array gives the same energy without re-permuting. (Pre-Phase-2 these were
# intentionally different for non-collinear configs at n > 1; that divergence is
# gone now that repeat un-folds.)
@testset "repeat ≡ supercell_matrix = reshape_base*diag(n) (un-fold unification)" begin
    path = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")
    if !isfile(path)
        @info "Skipping: bcc fixture missing"
    else
        rb = OPT.load_sce_hamiltonian(
            path; supercell_matrix = [1 0 0; 0 1 0; 0 0 1]).prim.reshape_base
        rng = MersenneTwister(0xfeed)
        for n in (1, 2)
            hl = OPT.load_sce_hamiltonian(path; repeat = (n, n, n))
            hm = OPT.load_sce_hamiltonian(path; supercell_matrix = rb *
                                                                   [n 0 0; 0 n 0; 0 0 n])
            @test hm.n_atoms == hl.n_atoms
            # Identical numbering: positions agree index-for-index.
            @test _position_perm(hm, hl) == collect(1:hm.n_atoms)
            # Same energy for the same config — ferro and a random non-collinear one.
            fz = repeat(Float64[0, 0, 1], 1, hm.n_atoms)
            @test isapprox(OPT.sce_energy(hm, fz), OPT.sce_energy(hl, fz); atol = 1.0e-9)
            sm = _rand_spins(hm.n_atoms, rng)
            @test isapprox(OPT.sce_energy(hm, sm), OPT.sce_energy(hl, sm); atol = 1.0e-9)
        end
    end
end
