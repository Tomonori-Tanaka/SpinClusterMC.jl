#!/usr/bin/env julia
#
# Run every benchmark in this directory back-to-back. Equivalent to:
#
#   julia --project=. benchmark/simple/bench_construction.jl
#   julia --project=. benchmark/simple/bench_energy.jl
#   julia --project=. benchmark/simple/bench_sweep.jl
#   julia --project=. benchmark/simple/bench_compare.jl
#
# Each is launched in a fresh Julia process so its own argument defaults
# apply cleanly (the per-script CLI option matrix is intentionally not
# unified — see each header for the per-bench knobs). Total wall time on
# a recent laptop is a couple of minutes with the defaults; pass
# `--fast` to skip the slowest fixtures.
#
# Usage:
#   julia --project=. benchmark/simple/runbench.jl
#   julia --project=. benchmark/simple/runbench.jl --fast

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

const JULIA_BIN = first(Base.julia_cmd().exec)
const PROJECT_ROOT = joinpath(@__DIR__, "..", "..")

fast_mode = "--fast" in ARGS

# Per-bench argument lists. `--fast` trims out ferh and shrinks eval counts
# so the runner finishes in well under a minute for quick smoke checks.
benches = if fast_mode
    [
        ("bench_construction.jl", ["--fixtures=bcc,fege", "--evals=2"]),
        ("bench_energy.jl",       ["--fixtures=bcc,fege", "--evals=20"]),
        ("bench_sweep.jl",        ["--fixtures=bcc",      "--sweeps=10"]),
        ("bench_compare.jl",      ["--fixtures=bcc",      "--evals=10"]),
    ]
else
    [
        ("bench_construction.jl", String[]),
        ("bench_energy.jl",       String[]),
        ("bench_sweep.jl",        String[]),
        ("bench_compare.jl",      String[]),
    ]
end

println("=" ^ 78)
println("benchmark/simple/runbench.jl ", fast_mode ? "(--fast)" : "")
println("=" ^ 78)

for (script, extra) in benches
    path = joinpath(@__DIR__, script)
    cmd = `$JULIA_BIN --project=$PROJECT_ROOT $path $extra`
    println()
    println("-" ^ 78)
    println("\$ ", cmd)
    println("-" ^ 78)
    run(cmd)
end

println()
println("=" ^ 78)
println("done")
println("=" ^ 78)
