using SpinClusterMC
using SpinClusterMC.Simple
using Test
import Magesty.AngularMomentumCoupling: build_all_real_bases

const SIMPLE = SpinClusterMC.Simple

const CG_FIXTURES = [
    (name = "bcc_2x2x2", path = joinpath(@__DIR__, "..", "bcc_2x2x2", "jphi.xml"),
        n_atoms = 16),
    (name = "fege_2x2x2", path = joinpath(@__DIR__, "..", "fege_2x2x2", "jphi.xml"),
        n_atoms = 64),
    (name = "ferh_4x4x4", path = joinpath(@__DIR__, "..", "ferh_4x4x4", "jphi.xml"),
        n_atoms = 128)
]

@testset "Simple.SpinClusterHamiltonian loads (3 fixtures)" begin
    for fix in CG_FIXTURES
        if !isfile(fix.path)
            @info "Skipping $(fix.name): fixture missing"
            continue
        end
        @testset "$(fix.name)" begin
            h = SIMPLE.SpinClusterHamiltonian(fix.path)
            @test h.n_atoms == fix.n_atoms
            @test h.base_n_atoms == fix.n_atoms
            @test h.repeat == (1, 1, 1)
            @test !isempty(h.instances)
            # Every instance's atom set must point inside the supercell.
            @test all(
                inst -> all(a -> 1 ≤ a ≤ h.n_atoms, inst.atoms), h.instances
            )
            # Repeat tiling extends the supercell linearly.
            h2 = SIMPLE.SpinClusterHamiltonian(fix.path; repeat = (2, 1, 1))
            @test h2.n_atoms == 2 * fix.n_atoms
            @test h2.base_n_atoms == fix.n_atoms
            @test length(h2.instances) == 2 * length(h.instances)
        end
    end
end

@testset "Simple.CGTable shape and coverage (3 fixtures)" begin
    for fix in CG_FIXTURES
        if !isfile(fix.path)
            @info "Skipping $(fix.name): fixture missing"
            continue
        end
        @testset "$(fix.name)" begin
            h = SIMPLE.SpinClusterHamiltonian(fix.path)
            cg = h.cg_table

            # Every (ls, Lf, Lseq) actually used by an instance must be in the
            # table, with the right shape and finite entries. Tested on the
            # set of unique keys (small) rather than every instance (huge).
            instance_keys = Set(
                (copy(inst.ls), inst.Lf, copy(inst.Lseq)) for inst in h.instances
            )
            @test all(k -> haskey(cg, k), instance_keys)
            @test all(instance_keys) do key
                ls, Lf, _Lseq = key
                T = cg[key]
                N = length(ls)
                expected = ntuple(i -> i ≤ N ? 2 * ls[i] + 1 : 2 * Lf + 1, N + 1)
                size(T) == expected && eltype(T) === Float64 && all(isfinite, T)
            end
            # CGTable may carry a few extra entries that build_all_real_bases
            # produced but the XML did not actually use; that is by design.
            @test length(cg) ≥ length(instance_keys)
        end
    end
end

@testset "Simple.build_cg_table parity vs direct Magesty call" begin
    # Pick an ls present in all three fixtures: [1, 1] (body=2, p-orbitals).
    h = SIMPLE.SpinClusterHamiltonian(CG_FIXTURES[1].path)
    cg = h.cg_table
    ls = [1, 1]
    bases_by_L, paths_by_L = build_all_real_bases(ls)
    for (Lf, tensor_list) in bases_by_L
        paths = paths_by_L[Lf]
        for (path, tensor) in zip(paths, tensor_list)
            key = (ls, Lf, collect(Int, path))
            @test haskey(cg, key)
            # Same call -> bit-exact equality.
            @test cg[key] == tensor
        end
    end
end

@testset "Simple.CGTable invariant: Lseq length matches max(0, N-2)" begin
    # For ls=[1,1], Lf=0: expected_lseq = max(0, 2-2) = 0 and expected_shape =
    # (2*1+1, 2*1+1, 2*0+1) = (3, 3, 1). The bad inputs below each violate one
    # invariant exactly.

    # Lseq length wrong (should be 0 for N=2, got 1).
    bad_entries = Dict(([1, 1], 0, [99]) => zeros(3, 3, 1))
    @test_throws ArgumentError SIMPLE.CGTable(bad_entries)
    # Tensor shape wrong (got (3, 4, 1) vs expected (3, 3, 1)).
    bad_shape = Dict(([1, 1], 0, Int[]) => zeros(3, 4, 1))
    @test_throws ArgumentError SIMPLE.CGTable(bad_shape)
    # Tensor ndims wrong (got 2 vs expected N+1 = 3).
    bad_ndims = Dict(([1, 1], 0, Int[]) => zeros(3, 3))
    @test_throws ArgumentError SIMPLE.CGTable(bad_ndims)
end

@testset "Simple.ClusterInstance invariants" begin
    # Mismatched atoms length.
    @test_throws ArgumentError SIMPLE.ClusterInstance(
        [1, 2, 3], [1, 1], 0, Int[], [1.0], 1.0, 1
    )
    # Mismatched Lseq length for N=3 (should be length 1, not 2).
    @test_throws ArgumentError SIMPLE.ClusterInstance(
        [1, 2, 3], [1, 1, 1], 0, [1, 2], [1.0], 1.0, 1
    )
    # Mismatched salc_weights length for Lf=2 (should be 5, not 3).
    @test_throws ArgumentError SIMPLE.ClusterInstance(
        [1, 2], [1, 1], 2, Int[], [1.0, 2.0, 3.0], 1.0, 1
    )
    # Valid construction round-trips.
    inst = SIMPLE.ClusterInstance(
        [1, 5], [1, 1], 0, Int[], [0.5773502691896257], -0.5, 2
    )
    @test inst.atoms == [1, 5]
    @test inst.ls == [1, 1]
    @test inst.Lf == 0
    @test isempty(inst.Lseq)
    @test inst.salc_weights == [0.5773502691896257]
    @test inst.J == -0.5
    @test inst.multiplicity == 2
end
