#!/usr/bin/env julia
# Development helper:
# - Activates this repository root so local source changes are reflected immediately.
# - For end users, prefer `examples/bccFe/parallel_tempering/example_job.jl`.

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

include(joinpath(@__DIR__, "../../examples/bccFe/parallel_tempering/example_job.jl"))
