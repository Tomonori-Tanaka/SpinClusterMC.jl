using SpinClusterMC
using SpinClusterMC.Simple
using Test
using LinearAlgebra
import EzXML

const SIMPLE = SpinClusterMC.Simple

@testset "Simple.parse_jphi_xml (3 fixtures)" begin
    fixtures = [
        (
            name = "bcc_2x2x2",
            path = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml"),
            n_atoms = 16,
            n_salcs = 2
        ),
        (
            name = "fege_2x2x2",
            path = joinpath(@__DIR__, "..", "fege_2x2x2", "jphi.xml"),
            n_atoms = 64,
            n_salcs = 734
        ),
        (
            name = "ferh_4x4x4",
            path = joinpath(@__DIR__, "..", "ferh_4x4x4", "jphi.xml"),
            n_atoms = 128,
            n_salcs = 488
        )
    ]

    for fix in fixtures
        if !isfile(fix.path)
            @info "Skipping $(fix.name): fixture missing"
            continue
        end

        @testset "$(fix.name)" begin
            data = SIMPLE.parse_jphi_xml(fix.path)

            # System.
            @test data.system.n_atoms == fix.n_atoms
            @test size(data.system.lattice) == (3, 3)
            @test size(data.system.pos_frac) == (3, fix.n_atoms)
            # Fractional coordinates are stored as written; they may live in
            # [0, 1) but we only assert they round-trip without NaN/Inf.
            @test all(isfinite, data.system.pos_frac)
            # Lattice is non-degenerate.
            @test abs(det(data.system.lattice)) > 0
            # Translation table is fully populated.
            @test size(data.system.map_sym) == (fix.n_atoms, data.system.n_trans)
            @test all(>(0), data.system.map_sym)
            @test all(≤(fix.n_atoms), data.system.map_sym)
            # Each translation column is a permutation of 1:n_atoms.
            for t in 1:data.system.n_trans
                @test sort(data.system.map_sym[:, t]) == collect(1:fix.n_atoms)
            end

            # SALCs and JPhi.
            @test length(data.salcs) == fix.n_salcs
            @test length(data.jphi) == fix.n_salcs

            # Per-SALC / per-basis invariants enforced at parse time, re-checked
            # here as a defense-in-depth sanity sweep.
            for (s, salc) in enumerate(data.salcs)
                @test salc.body ≥ 2
                @test salc.Lf ≥ 0
                @test !isempty(salc.bases)
                expected_weights = 2 * salc.Lf + 1
                expected_lseq = max(0, salc.body - 2)
                for basis in salc.bases
                    @test length(basis.atoms) == salc.body
                    @test length(basis.ls) == salc.body
                    @test length(basis.Lseq) == expected_lseq
                    @test length(basis.weights) == expected_weights
                    @test basis.multiplicity ≥ 1
                    @test all(a -> 1 ≤ a ≤ fix.n_atoms, basis.atoms)
                    @test all(l -> l ≥ 0, basis.ls)
                end
            end
        end
    end
end

@testset "Simple.parse_jphi_xml: error paths" begin
    # Missing file.
    @test_throws EzXML.XMLError SIMPLE.parse_jphi_xml(
        joinpath(@__DIR__, "does_not_exist.xml"),
    )
end
