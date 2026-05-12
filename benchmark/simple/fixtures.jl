# Shared helpers for benchmark/simple/*.jl. Not a public API.
#
# Each script in this directory `include`s this file to share:
#   - Fixture paths (bcc_2x2x2, fege_2x2x2, ferh_4x4x4)
#   - Argument parsing (matching benchmark/optimized/ style)
#   - Random unit-spin generator and a time formatter
#   - A thin `simple_bench(expr)` wrapper around BenchmarkTools that
#     prints min, median, allocations, and memory in a uniform way.
#
# The bench scripts use BenchmarkTools (not just `@elapsed`) because
# Simple is now the basis for performance work, and we need per-call
# allocation tracking and statistical timing to identify bottlenecks.
# See benchmark/Project.toml for the env this depends on.

using BenchmarkTools: BenchmarkTools, @benchmarkable, run, minimum, median
using Random: AbstractRNG, MersenneTwister
using LinearAlgebra: norm

const SIMPLE_FIXTURE_ROOT = abspath(joinpath(@__DIR__, "..", "..", "test"))

const SIMPLE_FIXTURES = (
    bcc  = joinpath(SIMPLE_FIXTURE_ROOT, "bcc_2x2x2",  "jphi.xml"),
    fege = joinpath(SIMPLE_FIXTURE_ROOT, "fege_2x2x2", "jphi.xml"),
    ferh = joinpath(SIMPLE_FIXTURE_ROOT, "ferh_4x4x4", "jphi.xml"),
)

"Parse `--key=value` CLI args into `Dict{String,String}`."
function simple_parse_args(args)
    opts = Dict{String, String}()
    for a in args
        startswith(a, "--") || error("unknown argument format: $a")
        kv = split(a[3:end], "="; limit = 2)
        length(kv) == 2 || error("argument must be --key=value, got: $a")
        opts[kv[1]] = kv[2]
    end
    return opts
end

"Parse a `n1,n2,n3` triple of positive integers."
function simple_parse_repeat(s::AbstractString)::NTuple{3, Int}
    parts = split(s, ",")
    length(parts) == 3 || error("repeat must be n1,n2,n3, got: $s")
    vals = parse.(Int, strip.(parts))
    all(>(0), vals) || error("repeat factors must be positive, got: $s")
    return (vals[1], vals[2], vals[3])
end

"Random `3 × n` matrix of unit-length spin directions."
function simple_random_spins(rng::AbstractRNG, n::Int)::Matrix{Float64}
    spins = randn(rng, 3, n)
    for i in 1:n
        spins[:, i] ./= norm(@view spins[:, i])
    end
    return spins
end

"Format a wall-clock duration (seconds) into a human-readable string."
function simple_fmt_time(s::Real)
    s < 1e-6 && return string(round(s * 1e9; digits = 1), " ns")
    s < 1e-3 && return string(round(s * 1e6; digits = 2), " µs")
    s < 1.0  && return string(round(s * 1e3; digits = 2), " ms")
    s < 60.0 && return string(round(s; digits = 2), " s")
    return string(round(s / 60; digits = 2), " min")
end

"Format a byte count into a human-readable string."
function simple_fmt_bytes(b::Real)
    b < 1024              && return string(round(Int, b), " B")
    b < 1024^2            && return string(round(b / 1024;     digits = 1), " KiB")
    b < 1024^3            && return string(round(b / 1024^2;   digits = 1), " MiB")
    return                   string(round(b / 1024^3;   digits = 2), " GiB")
end

"""
    BenchResult

Container for the subset of `BenchmarkTools.Trial` fields the bench scripts
report. Keeps min and median (in seconds) plus per-call allocations.
"""
struct BenchResult
    t_min::Float64       # seconds
    t_median::Float64    # seconds
    allocs::Int          # number of allocations per evaluation
    memory::Int          # bytes allocated per evaluation
    samples::Int         # number of timed samples BenchmarkTools collected
end

"""
    simple_bench(f; samples=..., seconds=...) -> BenchResult

Run `f` under `BenchmarkTools.@benchmarkable` and return a `BenchResult`.
`f` must be a zero-arg closure (typically `() -> work(args...)`); we wrap
it that way so the closure-specialization cost is paid once at definition
time rather than inside the timed loop.

`seconds` caps the total wall-clock budget (default 1.0s); large benches
should bump it so BenchmarkTools collects enough samples for a stable
median.
"""
function simple_bench(f; seconds::Real = 1.0, samples::Int = 10_000)
    # No `evals=` here; let `tune!` choose the per-sample eval count so
    # sub-µs ops don't floor at the timer resolution.
    bm = @benchmarkable ($f)() samples=samples seconds=seconds
    BenchmarkTools.tune!(bm)
    trial = run(bm)
    tmin  = minimum(trial)
    tmed  = median(trial)
    return BenchResult(
        BenchmarkTools.time(tmin)    / 1e9,
        BenchmarkTools.time(tmed)    / 1e9,
        BenchmarkTools.allocs(tmin),
        BenchmarkTools.memory(tmin),
        length(trial.times),
    )
end
