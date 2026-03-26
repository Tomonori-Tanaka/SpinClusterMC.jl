# Development Scripts

This directory is for developer-only helper scripts.

- These scripts may call `Pkg.activate(...)` and assume a local clone of this repository.
- End users should use files under `examples/`.

Current scripts:

- `example_job_dev.jl`: activates the repository root, then runs `examples/bccFe/mpi/example_job.jl`.
- `example_job_parallel_tempering_dev.jl`: activates the repository root, then runs `examples/bccFe/parallel_tempering/example_job.jl`.

Usage (Carlo CLI args are required):

- `... run`: continue from checkpoint if present
- `... run --restart`: delete prior history and start from scratch
- `julia scripts/dev/example_job_dev.jl run`
- `julia scripts/dev/example_job_dev.jl run --restart`
- `julia scripts/dev/example_job_parallel_tempering_dev.jl run`
- `julia scripts/dev/example_job_parallel_tempering_dev.jl run --restart`
