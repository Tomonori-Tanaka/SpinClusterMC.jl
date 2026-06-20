using SpinClusterMC
using SpinClusterMC.Simple
using Test
using Carlo
using Random: MersenneTwister, randn!
using LinearAlgebra: det, I, norm
using StaticArrays: SMatrix, SVector

const SIMPLE = SpinClusterMC.Simple

# Random unit-vector spin matrix (3 × n), columns normalized.
function _random_spins(n::Int, rng)
    s = randn!(rng, Matrix{Float64}(undef, 3, n))
    for i in 1:n
        c = norm(@view s[:, i])
        s[:, i] ./= c
    end
    return s
end

# A spread of non-singular 3×3 integer matrices: diagonal, sheared (non-diagonal,
# unimodular shear), a general supercell, and two left-handed (det < 0) cases.
const SUPERCELL_MATRICES = [
    SMatrix{3, 3, Int}([1 0 0; 0 1 0; 0 0 1]),
    SMatrix{3, 3, Int}([2 0 0; 0 2 0; 0 0 2]),
    SMatrix{3, 3, Int}([3 0 0; 0 1 0; 0 0 2]),
    SMatrix{3, 3, Int}([2 1 0; 0 2 0; 0 0 3]),
    SMatrix{3, 3, Int}([1 1 0; 0 1 0; 0 0 1]),    # unimodular shear, det = 1
    SMatrix{3, 3, Int}([2 1 1; 1 2 0; 0 1 2]),    # general, det = 7
    SMatrix{3, 3, Int}([-1 0 0; 0 1 0; 0 0 1]),   # det = -1
    SMatrix{3, 3, Int}([2 1 0; 0 -2 0; 0 0 1])   # det = -4
]

@testset "supercell integer linear algebra (M1)" begin
    @testset "_int_det3 matches LinearAlgebra.det" begin
        for M in SUPERCELL_MATRICES
            @test SIMPLE._int_det3(M) == round(Int, det(M))
        end
    end

    @testset "_adjugate3: M * adj(M) == det(M) I" begin
        for M in SUPERCELL_MATRICES
            d = SIMPLE._int_det3(M)
            adjM = SIMPLE._adjugate3(M)
            @test M * adjM == d * SMatrix{3, 3, Int}(I)
            @test adjM * M == d * SMatrix{3, 3, Int}(I)
        end
    end

    @testset "_col_hermite: H = M*U, lower-triangular, positive diagonal" begin
        for M in SUPERCELL_MATRICES
            H, U = SIMPLE._col_hermite(M)
            # Factorization and unimodularity.
            @test H == M * U
            @test abs(SIMPLE._int_det3(U)) == 1
            # Lower triangular with strictly positive diagonal.
            @test H[1, 2] == 0 && H[1, 3] == 0 && H[2, 3] == 0
            @test H[1, 1] > 0 && H[2, 2] > 0 && H[3, 3] > 0
            # Diagonal product == |det(M)| == number of supercell cells.
            @test H[1, 1] * H[2, 2] * H[3, 3] == abs(SIMPLE._int_det3(M))
        end
    end

    @testset "_wrap_offset_into_supercell: coset canonicalization" begin
        for M in SUPERCELL_MATRICES
            d = SIMPLE._int_det3(M)
            adjM = SIMPLE._adjugate3(M)
            ncells = abs(d)

            # Enumerate one representative per coset from the HNF diagonal box,
            # then canonicalize via wrap.
            H, _ = SIMPLE._col_hermite(M)
            reps = Set{NTuple{3, Int}}()
            for c3 in 0:(H[3, 3] - 1), c2 in 0:(H[2, 2] - 1), c1 in 0:(H[1, 1] - 1)
                push!(reps, SIMPLE._wrap_offset_into_supercell((c1, c2, c3), M, adjM, d))
            end
            # Distinct canonical reps == number of cells (no collisions).
            @test length(reps) == ncells

            # Each rep lies in the fundamental domain M*[0,1)^3:
            # M^{-1} rep == adjM*rep / d must be in [0,1) componentwise.
            for rep in reps
                fv = adjM * SVector{3, Int}(rep[1], rep[2], rep[3])
                for i in 1:3
                    # 0 <= fv_i / d < 1 (handle sign of d via the fraction).
                    frac = fv[i] / d
                    @test 0.0 <= frac < 1.0
                end
            end

            # Wrap is constant on cosets: c and c + M*k map to the same rep.
            for k in [(1, 0, 0), (0, -2, 1), (3, 1, -1), (-1, -1, -1)]
                kv = SVector{3, Int}(k[1], k[2], k[3])
                shift = M * kv
                base = (0, 0, 0)
                shifted = (base[1] + shift[1], base[2] + shift[2], base[3] + shift[3])
                @test SIMPLE._wrap_offset_into_supercell(base, M, adjM, d) ==
                      SIMPLE._wrap_offset_into_supercell(shifted, M, adjM, d)
            end

            # Wrapping an already-canonical rep is idempotent.
            for rep in reps
                @test SIMPLE._wrap_offset_into_supercell(rep, M, adjM, d) == rep
            end
        end
    end
end

const SUPERCELL_FIXTURES = [
    (name = "bcc_2x2x2", path = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml"),
        n_atoms = 16, n_prim = 1),
    (name = "fege_2x2x2", path = joinpath(@__DIR__, "..", "fege_2x2x2", "jphi.xml"),
        n_atoms = 64, n_prim = 8),
    (name = "ferh_4x4x4", path = joinpath(@__DIR__, "..", "ferh_4x4x4", "jphi.xml"),
        n_atoms = 128, n_prim = 2)
]

@testset "extract_primitive (M2)" begin
    for fix in SUPERCELL_FIXTURES
        if !isfile(fix.path)
            @info "Skipping $(fix.name): fixture missing"
            continue
        end
        @testset "$(fix.name)" begin
            data = SIMPLE.parse_jphi_xml(fix.path)
            sys = data.system
            prim = SIMPLE.extract_primitive(sys)

            # Sublattice count and base-cell factorization.
            @test prim.n_prim == fix.n_prim
            @test prim.n_prim * sys.n_trans == fix.n_atoms
            @test abs(round(Int, det(prim.reshape_base))) == sys.n_trans

            # Recovered primitive lattice is right-handed.
            @test det(prim.lattice) > 0

            # base_lattice == primitive_lattice * reshape_base.
            @test isapprox(
                sys.lattice, prim.lattice * prim.reshape_base; atol = 1.0e-8,
                rtol = 1.0e-8)

            # Each sublattice holds exactly n_trans base atoms.
            counts = zeros(Int, prim.n_prim)
            for a in 1:fix.n_atoms
                counts[prim.base_to_prim[a][1]] += 1
            end
            @test all(==(sys.n_trans), counts)

            # base_to_prim ↔ prim_to_base round-trip.
            for a in 1:fix.n_atoms
                @test prim.prim_to_base[prim.base_to_prim[a]] == a
            end
            # prim_to_base covers exactly the base atoms.
            @test length(prim.prim_to_base) == fix.n_atoms
            @test Set(values(prim.prim_to_base)) == Set(1:fix.n_atoms)

            # Geometry check: each base atom's recovered primitive position
            # (sublattice frac + integer offset) maps back to its Cartesian
            # position within the base cell.
            for a in 1:fix.n_atoms
                s, Δ = prim.base_to_prim[a]
                g = prim.pos_frac[:, s] .+ collect(Float64, Δ)
                cart = prim.lattice * g
                @test isapprox(cart, sys.lattice * sys.pos_frac[:, a];
                    atol = 1.0e-6, rtol = 1.0e-6)
            end
        end
    end
end

@testset "build_templates (M3)" begin
    for fix in SUPERCELL_FIXTURES
        if !isfile(fix.path)
            @info "Skipping $(fix.name): fixture missing"
            continue
        end
        @testset "$(fix.name)" begin
            data = SIMPLE.parse_jphi_xml(fix.path)
            prim = SIMPLE.extract_primitive(data.system)
            templates = SIMPLE.build_templates(data.salcs, data.jphi, data.system, prim)

            @test !isempty(templates)
            for tpl in templates
                N = length(tpl.ls)
                # Pivot conventions.
                @test tpl.site_delta[1] == (0, 0, 0)
                @test tpl.pivot_subl == tpl.site_subl[1]
                # Structural length invariants (mirror ClusterInstance).
                @test length(tpl.site_subl) == N
                @test length(tpl.site_delta) == N
                @test length(tpl.Lseq) == max(0, N - 2)
                @test length(tpl.weights) == 2 * tpl.Lf + 1
                # Sublattices in range.
                @test all(s -> 1 <= s <= prim.n_prim, tpl.site_subl)
            end

            # jphi_threshold short-circuit: threshold above every |J| drops all,
            # threshold 0 keeps the full set.
            maxabs = maximum(abs, data.jphi)
            t_all = SIMPLE.build_templates(
                data.salcs, data.jphi, data.system, prim; jphi_threshold = 0.0)
            t_none = SIMPLE.build_templates(
                data.salcs, data.jphi, data.system, prim;
                jphi_threshold = nextfloat(maxabs))
            @test length(t_all) == length(templates)
            @test isempty(t_none)
        end
    end
end

# Build a Hamiltonian from the general-matrix tiling path (mirrors what the
# `supercell_matrix` constructor will assemble in M5).
function _matrix_hamiltonian(data, prim, templates, M)
    inst, na = SIMPLE._generate_instances_matrix(templates, prim, M)
    cg = SIMPLE.build_cg_table(data.salcs)
    maxl = SIMPLE._max_l_in_instances(inst)
    a2i = SIMPLE._build_atom_to_instance_indices(inst, na)
    return SIMPLE.SpinClusterHamiltonian(
        na, data.system.n_atoms, (0, 0, 0), inst, cg, maxl, a2i)
end

# Ferromagnetic (all +z) spin matrix; permutation-invariant, so its energy is
# independent of the atom numbering — letting us compare the matrix path
# (primitive numbering) against the legacy path (base-cell numbering).
_ferro(n) = repeat(Float64[0, 0, 1], 1, n)

@testset "matrix tiling energy equivalence with legacy (M4)" begin
    # bcc is fast; fege energy evaluations are heavy, so only run them under
    # the "slow" test arg.
    fixtures = "slow" in ARGS ? SUPERCELL_FIXTURES[1:2] : SUPERCELL_FIXTURES[1:1]
    for fix in fixtures
        if !isfile(fix.path)
            @info "Skipping $(fix.name): fixture missing"
            continue
        end
        @testset "$(fix.name)" begin
            data = SIMPLE.parse_jphi_xml(fix.path)
            sys = data.system
            prim = SIMPLE.extract_primitive(sys)
            templates = SIMPLE.build_templates(data.salcs, data.jphi, sys, prim)

            # Reference per-atom ferro energy from the legacy base cell.
            leg0 = SIMPLE.SpinClusterHamiltonian(fix.path; repeat = (1, 1, 1))
            e0 = SIMPLE.total_energy(leg0, _ferro(leg0.n_atoms)) / leg0.n_atoms

            # Diagonal base-multiples must match the legacy diagonal path exactly.
            for rep in [(1, 1, 1), (2, 1, 1), (2, 2, 2)]
                leg = SIMPLE.SpinClusterHamiltonian(fix.path; repeat = rep)
                M = SIMPLE._supercell_from_repeat(prim.reshape_base, rep)
                nh = _matrix_hamiltonian(data, prim, templates, M)
                @test nh.n_atoms == sys.n_atoms * rep[1] * rep[2] * rep[3]
                el = SIMPLE.total_energy(leg, _ferro(leg.n_atoms)) / leg.n_atoms
                en = SIMPLE.total_energy(nh, _ferro(nh.n_atoms)) / nh.n_atoms
                @test isapprox(el, en; atol = 1.0e-9)
            end

            # General matrices: non-base-multiple, non-diagonal, and a single
            # primitive cell must all give the SAME intensive per-atom energy.
            for Md in [[1 0 0; 0 1 0; 0 0 1],       # single primitive cell
                [2 0 0; 0 3 0; 0 0 2],              # non-base-multiple
                [2 1 0; 0 2 0; 0 0 2]]              # non-diagonal
                M = SMatrix{3, 3, Int}(Md)
                nh = _matrix_hamiltonian(data, prim, templates, M)
                @test nh.n_atoms == prim.n_prim * abs(SIMPLE._int_det3(M))
                @test all(inst -> all(a -> 1 <= a <= nh.n_atoms, inst.atoms),
                    nh.instances)
                en = SIMPLE.total_energy(nh, _ferro(nh.n_atoms)) / nh.n_atoms
                @test isapprox(e0, en; atol = 1.0e-9)
            end
        end
    end
end

@testset "matrix path ΔE consistency and scaling (M6)" begin
    path = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")
    if !isfile(path)
        @info "Skipping M6: bcc fixture missing"
    else
        rng = MersenneTwister(20260620)

        @testset "delta_local_energy matches total recompute" begin
            # Non-diagonal supercell; numbering-independent correctness check
            # that the matrix-path instances + atom_to_instance_indices drive a
            # consistent single-flip ΔE.
            h = SIMPLE.SpinClusterHamiltonian(path; supercell_matrix = [2 1 0; 0 2 0;
                                                                        0 0 2])
            spins = _random_spins(h.n_atoms, rng)
            for i in 1:h.n_atoms
                S_new = _random_spins(1, rng)[:, 1]
                e0 = SIMPLE.total_energy(h, spins)
                spins_new = copy(spins)
                spins_new[:, i] .= S_new
                e1 = SIMPLE.total_energy(h, spins_new)
                @test isapprox(
                    SIMPLE.delta_local_energy(h, spins, i, S_new), e1 - e0;
                    atol = 1.0e-9, rtol = 1.0e-9)
            end
        end

        @testset "ferro total energy scales with |det(M)|" begin
            # E_total = n_atoms * e0 = n_prim * |det(M)| * e0, so the per-cell
            # energy is constant across supercell matrices.
            ref = nothing
            for Md in [[1 0 0; 0 1 0; 0 0 1], [2 0 0; 0 2 0; 0 0 2],
                [3 0 0; 0 2 0; 0 0 2], [2 1 0; 0 1 0; 0 0 3]]
                M = SMatrix{3, 3, Int}(Md)
                h = SIMPLE.SpinClusterHamiltonian(path; supercell_matrix = Md)
                etot = SIMPLE.total_energy(h, _ferro(h.n_atoms))
                per_cell = etot / abs(SIMPLE._int_det3(M))
                if ref === nothing
                    ref = per_cell
                else
                    @test isapprox(per_cell, ref; atol = 1.0e-9)
                end
            end
        end
    end
end

@testset "SCEMC supercell_matrix integration (M6)" begin
    path = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")
    if !isfile(path)
        @info "Skipping M6 SCEMC: bcc fixture missing"
    else
        params = Dict{Symbol, Any}(
            :xml_path => path, :T => 300.0,
            :thermalization => 0, :binsize => 1, :seed => 42,
            :supercell_matrix => [2 1 0; 0 2 0; 0 0 2])
        mc = SIMPLE.SCEMC(params)
        @test mc.supercell_matrix == [2 1 0; 0 2 0; 0 0 2]
        @test mc.h.n_atoms == 8        # n_prim(=1) * |det| (=8)

        ctx = Carlo.MCContext{MersenneTwister}(params)
        Carlo.init!(mc, ctx, params)
        # Spins are unit vectors and energy matches a fresh recompute.
        @test all(i -> isapprox(norm(@view mc.spins[:, i]), 1.0; atol = 1.0e-12),
            1:mc.h.n_atoms)
        @test isapprox(mc.energy, SIMPLE.total_energy(mc.h, mc.spins); atol = 1.0e-9)
        for _ in 1:5
            Carlo.sweep!(mc, ctx)
        end
        @test isapprox(mc.energy, SIMPLE.total_energy(mc.h, mc.spins);
            atol = 1.0e-6, rtol = 1.0e-6)

        # Base-cell-sized :initial_spins cannot be tiled in the matrix path.
        bad = Dict{Symbol, Any}(
            :xml_path => path, :T => 300.0,
            :thermalization => 0, :binsize => 1, :seed => 42,
            :supercell_matrix => [2 0 0; 0 2 0; 0 0 2],
            :initial_spins => zeros(3, 16))   # 16 = base cell, != 8 atoms
        mc_bad = SIMPLE.SCEMC(bad)
        ctx_bad = Carlo.MCContext{MersenneTwister}(bad)
        @test_throws ArgumentError Carlo.init!(mc_bad, ctx_bad, bad)
    end
end

@testset "supercell_matrix error handling (M6)" begin
    path = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml")
    if !isfile(path)
        @info "Skipping M6 errors: bcc fixture missing"
    else
        # Non-3×3 matrix.
        @test_throws ArgumentError SIMPLE.SpinClusterHamiltonian(
            path; supercell_matrix = [1 0; 0 1])
        # Singular matrix (det = 0).
        @test_throws ArgumentError SIMPLE.SpinClusterHamiltonian(
            path; supercell_matrix = [1 0 0; 0 1 0; 0 0 0])
        # Both repeat and supercell_matrix specified.
        @test_throws ArgumentError SIMPLE.SpinClusterHamiltonian(
            path; repeat = (2, 1, 1), supercell_matrix = [1 0 0; 0 1 0; 0 0 1])
        # threshold drops every SALC.
        @test_throws ArgumentError SIMPLE.SpinClusterHamiltonian(
            path; supercell_matrix = [2 0 0; 0 2 0; 0 0 2], jphi_threshold = 1.0e9)

        # SCEMC param validation: non-integer matrix, dual spec.
        @test_throws ArgumentError SIMPLE._params_supercell_matrix(
            Dict{Symbol, Any}(:supercell_matrix => [2.5 0 0; 0 1 0; 0 0 1]))
        @test_throws ArgumentError SIMPLE._build_hamiltonian(Dict{Symbol, Any}(
            :xml_path => path, :repeat => (2, 1, 1),
            :supercell_matrix => [1 0 0; 0 1 0; 0 0 1]))
    end
end
