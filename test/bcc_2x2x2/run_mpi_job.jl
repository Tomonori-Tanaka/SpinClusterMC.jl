# Helper script for spawning Carlo jobs in a separate MPI process.
# Called by _run_mpi_job() in test_bcc_2x2x2.jl as:
#   mpiexec -n N julia --project=... run_mpi_job.jl <serialized_job_path>
using Serialization
using Carlo
import Carlo.JobTools as JT
using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo

job_path = ARGS[1]
job, scheduler = deserialize(job_path)
Carlo.start(scheduler, job)
