using SpinClusterMC.JPhiMagestyCarlo
using Carlo
import Carlo.JobTools as JT
import Carlo.ResultTools
using MPI
using Serialization
using Test
using Logging
using Random
using Statistics

const XML_2x2x2 = joinpath(@__DIR__, "jphi.xml")

# ---------------------------------------------------------------------------
# Helpers for restart tests
# ---------------------------------------------------------------------------

function _make_jphi_job(dir::String, sweeps::Int; T::Float64 = 10.0)
    tm = JT.TaskMaker()
    tm.sweeps = sweeps
    tm.seed = 42
    tm.thermalization = 20
    tm.binsize = 10
    tm.xml_path = XML_2x2x2
    tm.T = T
    JT.task(tm)
    return JT.JobInfo(
        dir, JPhiSpinMC;
        tasks = JT.make_tasks(tm),
        checkpoint_time = "1:00",
        run_time = "5:00",
    )
end

function _compare_results(job1, job2)
    r1 = ResultTools.dataframe(JT.result_filename(job1))
    r2 = ResultTools.dataframe(JT.result_filename(job2))
    for (t1, t2) in zip(r1, r2)
        for key in keys(t1)
            startswith(key, "_ll_") && continue
            @test (key, t1[key]) == (key, t2[key])
        end
    end
end

# Spawn an MPI subprocess that runs `Carlo.start(scheduler, job)`.
# The job directory is created before spawning so the subprocess can write into it.
function _run_mpi_job(job; num_ranks::Int, scheduler = Carlo.MPIScheduler, silent::Bool = true)
    JT.create_job_directory(job)
    job_path = joinpath(job.dir, "jobfile")
    serialize(job_path, (job, scheduler))
    helper = joinpath(@__DIR__, "run_mpi_job.jl")
    project = Base.active_project()
    cmd = `$(mpiexec()) -n $num_ranks $(Base.julia_cmd()) --project=$project $helper $job_path`
    silent && (cmd = pipeline(cmd; stdout = devnull, stderr = devnull))
    run(cmd)
end

# ---------------------------------------------------------------------------

@testset "initial_spins tiling via Carlo.init!" begin
    # Verify that passing :initial_spins (3 × base_n_atoms) tiles correctly into
    # the supercell and that Carlo.init! honours it instead of randomising spins.
    base_n = 16   # the XML already defines the 2×2×2 supercell as the primitive cell

    @testset "repeat=$rep" for (rep, expected_n) in [((1,1,1), 16), ((2,2,2), 128)]
        # Ferromagnetic along +z as the base-cell configuration
        init_spins = zeros(3, base_n)
        init_spins[3, :] .= 1.0

        params = Dict(
            :xml_path       => XML_2x2x2,
            :repeat         => rep,
            :T              => 1.0,
            :thermalization => 0,
            :binsize        => 1,
            :seed           => 42,
            :initial_spins  => init_spins,
        )
        mc  = JPhiSpinMC(params)
        ctx = Carlo.MCContext{MersenneTwister}(params)
        Carlo.init!(mc, ctx, params)

        @test size(mc.spins) == (3, expected_n)
        # Every supercell spin must equal the tiled base-cell spin (unit +z)
        @test all(mc.spins[1, :] .≈ 0.0)
        @test all(mc.spins[2, :] .≈ 0.0)
        @test all(mc.spins[3, :] .≈ 1.0)
    end

    @testset "non-uniform base cell is tiled periodically" begin
        # Use a non-trivial base configuration: alternating +z / -z pattern
        init_spins = zeros(3, base_n)
        for i in 1:base_n
            init_spins[3, i] = iseven(i) ? 1.0 : -1.0
        end

        params = Dict(
            :xml_path       => XML_2x2x2,
            :repeat         => (2, 1, 1),
            :T              => 1.0,
            :thermalization => 0,
            :binsize        => 1,
            :seed           => 42,
            :initial_spins  => init_spins,
        )
        mc  = JPhiSpinMC(params)
        ctx = Carlo.MCContext{MersenneTwister}(params)
        Carlo.init!(mc, ctx, params)

        n = mc.ham.n_atoms   # 32 for repeat=(2,1,1)
        @test n == 2 * base_n
        for ia in 1:n
            ib = ((ia - 1) % base_n) + 1   # expected base atom
            @test mc.spins[:, ia] ≈ init_spins[:, ib]
        end
    end

    @testset "shape mismatch raises ArgumentError" begin
        wrong = zeros(3, 5)   # wrong number of base atoms
        params = Dict(
            :xml_path       => XML_2x2x2,
            :T              => 1.0,
            :thermalization => 0,
            :binsize        => 1,
            :seed           => 42,
            :initial_spins  => wrong,
        )
        mc  = JPhiSpinMC(params)
        ctx = Carlo.MCContext{MersenneTwister}(params)
        @test_throws ArgumentError Carlo.init!(mc, ctx, params)
    end
end

# ---------------------------------------------------------------------------

@testset "Ferromagnetic magnetization = 1 (no spin updates, repeat=$rep)" for rep in [(1,1,1), (2,2,2)]
    # Build MC for the bcc 2x2x2 system.
    # repeat=(1,1,1): 16 atoms (the XML already describes the 2×2×2 supercell)
    # repeat=(2,2,2): 128 atoms (further tiling of that supercell)
    params = Dict(
        :xml_path => XML_2x2x2,
        :repeat   => rep,
        :T        => 1.0,
        :thermalization => 0,
        :binsize  => 1,
        :seed     => 42,
    )
    mc = JPhiSpinMC(params)

    # Set ferromagnetic spin configuration: all spins along +z
    mc.spins .= 0.0
    mc.spins[3, :] .= 1.0

    # Measure once without any sweep (Carlo.measure! reads mc.spins directly)
    ctx = Carlo.MCContext{MersenneTwister}(params)
    Carlo.measure!(mc, ctx)

    # With every spin identical and |s| = 1, the vector magnetization magnitude
    # must be exactly 1.0 regardless of temperature or Hamiltonian.
    mag_mean = only(Statistics.mean(ctx.measure.observables[:Magnetization]))
    @test mag_mean ≈ 1.0
end

# ---------------------------------------------------------------------------

@testset "Low-T MC magnetization ≈ 1 (bcc_2x2x2, T=0.01 eV)" begin
    # The BCC Fe exchange coupling scale is ~1 eV (ground-state energy -(2+√3) eV/atom).
    # At T = 0.01 eV (T/J ≈ 0.01) the ferromagnetic state is essentially frozen:
    # virtually every Metropolis proposal that raises the energy is rejected.
    T_low   = 0.01   # eV
    n_therm = 200    # thermalization sweeps (starting from ferromagnetic state)
    n_meas  = 200    # measurement sweeps

    params = Dict(
        :xml_path       => XML_2x2x2,
        :T              => T_low,
        :thermalization => n_therm,
        :binsize        => 1,
        :seed           => 1234,
    )
    mc  = JPhiSpinMC(params)

    # Start from the ferromagnetic ground state (all spins along +z)
    mc.spins .= 0.0
    mc.spins[3, :] .= 1.0
    JMCC._rebuild_zlm_cache!(mc)
    mc.energy = mc.ham.j0 + JMCC._energy_from_instances(
        mc.local_cache.instances[mc.active_instance_indices], mc.spins,
    )

    ctx = Carlo.MCContext{MersenneTwister}(params)

    # Thermalization (state should remain near the ground state)
    for _ in 1:n_therm
        Carlo.sweep!(mc, ctx)
    end

    # Measurement: collect n_meas independent samples
    for _ in 1:n_meas
        Carlo.sweep!(mc, ctx)
        Carlo.measure!(mc, ctx)
    end

    # At T << J the magnetization must stay near 1
    mag_mean = only(Statistics.mean(ctx.measure.observables[:Magnetization]))
    @test mag_mean ≈ 1.0 atol=0.05
end

# ---------------------------------------------------------------------------

@testset "bcc_2x2x2 ferromagnetic energy" begin
    h = load_sce_hamiltonian(XML_2x2x2; repeat = (1, 1, 1))
    @test h.n_atoms == 16

    # Ferromagnetic configuration: all spins along z
    spins = zeros(3, h.n_atoms)
    spins[3, :] .= 1.0

    E_per_atom = sce_energy(h, spins) / h.n_atoms

    # For Jij = -1 eV/bond with 8 nearest neighbours and 6 second-nearest
    # neighbours per atom, the ferromagnetic ground-state energy is -(2+√3) eV/atom.
    @test E_per_atom ≈ -(2 + sqrt(3)) rtol = 1e-8
end

# ---------------------------------------------------------------------------

@testset "SCE energy: reference path agrees with fast path for repeat=(2,2,2)" begin
    # Verifies coupled_cluster_energy (reference) and _energy_from_instances (fast path)
    # give the same result for non-uniform spins after the cross-tile interaction fix.
    h = load_sce_hamiltonian(XML_2x2x2; repeat = (2, 2, 2))
    rng = MersenneTwister(7)
    spins = let s = randn(rng, 3, h.n_atoms)
        for i in 1:h.n_atoms; s[:, i] ./= sqrt(sum(s[:, i].^2)); end
        s
    end

    E_ref  = sce_energy(h, spins)
    cache  = JMCC.build_local_energy_cache(h)
    E_fast = h.j0 + JMCC._energy_from_instances(cache.instances, spins)

    @test E_ref ≈ E_fast rtol = 1e-8

    @testset "low-T MC from ferromagnetic state keeps magnetization near 1" begin
        # Start from fully ferromagnetic base-cell spins and run low-temperature
        # Metropolis updates on the repeat=(2,2,2) supercell.
        init_spins = zeros(3, h.base_n_atoms)
        init_spins[3, :] .= 1.0
        params = Dict(
            :xml_path       => XML_2x2x2,
            :repeat         => (2, 2, 2),
            :T              => 0.01,
            :thermalization => 0,
            :binsize        => 1,
            :seed           => 20260417,
            :initial_spins  => init_spins,
        )
        mc = JPhiSpinMC(params)
        ctx = Carlo.MCContext{MersenneTwister}(params)
        Carlo.init!(mc, ctx, params)

        n_sweeps = 200
        for _ in 1:n_sweeps
            Carlo.sweep!(mc, ctx)
            Carlo.measure!(mc, ctx)
        end

        mag_mean = only(Statistics.mean(ctx.measure.observables[:Magnetization]))
        @test mag_mean > 0.95
    end
end

# ---------------------------------------------------------------------------

@testset "SCE interaction energy is extensive for repeat=(2,2,2)" begin
    # (E - j0)/n_atoms for a periodically-tiled spin config must be
    # independent of repeat size.
    h1 = load_sce_hamiltonian(XML_2x2x2; repeat = (1, 1, 1))
    h2 = load_sce_hamiltonian(XML_2x2x2; repeat = (2, 2, 2))

    spins1 = zeros(3, h1.n_atoms); spins1[3, :] .= 1.0
    spins2 = zeros(3, h2.n_atoms)
    for ia in 1:h2.n_atoms
        spins2[:, ia] = spins1[:, ((ia - 1) % h1.n_atoms) + 1]
    end

    E_int1 = sce_energy(h1, spins1) - h1.j0
    E_int2 = sce_energy(h2, spins2) - h2.j0

    @test E_int2 ≈ 8 * E_int1 rtol = 1e-8
end

# ---------------------------------------------------------------------------

@testset "Cross-tile interaction energy detected (repeat=(2,2,2) bug regression)" begin
    # Bug regression: with the old (buggy) code all cluster atoms were placed
    # in the SAME tile, so inter-tile bonds were never evaluated.
    # The two spin configurations below differ ONLY in the inter-tile bonds:
    #   Config A: all spins +z  →  all bonds (intra + inter) are ferromagnetically aligned
    #   Config B: tile 0 = +z, tiles 1–7 = +x  →  intra-tile bonds still FM,
    #             but inter-tile bonds connect +z and +x (dot product = 0, not -1)
    # With the bug:  E_A = E_B  (inter-tile bonds simply absent)
    # After the fix: E_A < E_B  (inter-tile bonds penalize the misaligned config)
    h = load_sce_hamiltonian(XML_2x2x2; repeat = (2, 2, 2))
    n     = h.n_atoms        # 128
    base_n = h.base_n_atoms  # 16

    spins_A = zeros(3, n); spins_A[3, :] .= 1.0  # all +z

    spins_B = zeros(3, n)
    for ia in 1:n
        tile = (ia - 1) ÷ base_n   # 0 for tile-0 atoms, 1-7 for the rest
        if tile == 0
            spins_B[3, ia] = 1.0   # +z
        else
            spins_B[1, ia] = 1.0   # +x
        end
    end

    E_A = sce_energy(h, spins_A)
    E_B = sce_energy(h, spins_B)

    # The inter-tile bonds contribute negative energy for config A (FM) and zero for config B.
    # So E_A must be strictly less than E_B.
    @test E_A < E_B - 1e-6
end

# ---------------------------------------------------------------------------

@testset "Metropolis checkpoint restart" begin
    # Strategy (mirrors Carlo.jl test_scheduler.jl):
    #   full run  : 100 sweeps, single pass   → results_full
    #   half+half : 50 sweeps → checkpoint → resume to 100 sweeps → results_half
    # Because the RNG state is serialized in the checkpoint, both paths must
    # produce bit-identical measurement bins.
    mktempdir() do tmpdir
        T = 10.0   # high T → fast spin mixing, acceptance ≈ 1

        Logging.with_logger(Logging.NullLogger()) do
            # Continuous full run
            Carlo.start(Carlo.SingleScheduler, _make_jphi_job("$tmpdir/full", 100; T))

            # First half → checkpoint
            Carlo.start(Carlo.SingleScheduler, _make_jphi_job("$tmpdir/half", 50; T))
            # Resume to full
            Carlo.start(Carlo.SingleScheduler, _make_jphi_job("$tmpdir/half", 100; T))
        end

        # All target sweeps should be reached after restart
        for t in JT.read_progress(_make_jphi_job("$tmpdir/half", 100; T))
            @test t.sweeps >= t.target_sweeps
        end

        # Results must be bit-identical (same RNG seed + preserved RNG state)
        _compare_results(
            _make_jphi_job("$tmpdir/full", 100; T),
            _make_jphi_job("$tmpdir/half", 100; T),
        )
    end
end

# ---------------------------------------------------------------------------

@testset "Parallel tempering checkpoint restart" begin
    # Three temperatures in a high-T regime so spins mix quickly.
    # MPIScheduler: one rank per temperature + one coordinator rank.
    # Strategy: run 50 sweeps (checkpoint), extend to 100 sweeps,
    # verify that all tasks reach their target sweep count.
    mktempdir() do tmpdir
        Ts = [2.0, 5.0, 10.0]

        function make_pt_job(sweeps)
            tm = JT.TaskMaker()
            tm.sweeps = sweeps
            tm.seed = 42
            tm.thermalization = 20
            tm.binsize = 10
            tm.xml_path = XML_2x2x2
            tm.parallel_tempering = (;
                mc = JPhiSpinMC,
                parameter = :T,
                values = Ts,
                interval = 5,
            )
            JT.task(tm)
            return JT.JobInfo(
                "$tmpdir/pt", Carlo.ParallelTemperingMC;
                tasks = JT.make_tasks(tm),
                checkpoint_time = "1:00",
                run_time = "5:00",
                ranks_per_run = length(Ts),
            )
        end

        n_ranks = length(Ts) + 1   # +1 coordinator for MPIScheduler

        # First half: run 50 sweeps → checkpoint saved on completion
        _run_mpi_job(make_pt_job(50); num_ranks = n_ranks)

        # Resume: extend target to 100 sweeps
        job_full = make_pt_job(100)
        _run_mpi_job(job_full; num_ranks = n_ranks)

        for t in JT.read_progress(job_full)
            @test t.sweeps >= t.target_sweeps
        end
    end
end
