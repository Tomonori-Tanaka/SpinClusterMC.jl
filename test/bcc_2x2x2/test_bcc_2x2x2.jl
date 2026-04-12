using SpinClusterMC.JPhiMagestyCarlo
using Carlo
import Carlo.JobTools as JT
import Carlo.ResultTools
using MPI
using Serialization
using Test
using Logging

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
