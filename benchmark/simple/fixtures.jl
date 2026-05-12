# Shared helpers for benchmark/simple/*.jl. Not a public API.
#
# Each script in this directory `include`s this file to share:
#   - Fixture paths (bcc_2x2x2, fege_2x2x2, ferh_4x4x4)
#   - Argument parsing (matching benchmark/optimized/ style)
#   - Random unit-spin generator and a time formatter

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
    s < 1e-3 && return string(round(s * 1e6; digits = 2), " µs")
    s < 1.0  && return string(round(s * 1e3; digits = 2), " ms")
    s < 60.0 && return string(round(s; digits = 2), " s")
    return string(round(s / 60; digits = 2), " min")
end

"""
Average wall-clock time over `n_eval` evaluations of `f()`. One untimed
warm-up call is issued first so closure-specialization / allocator-warmup
costs do not show up in the timed loop. Returns `(t_avg, checksum)`.
"""
function simple_avg_time(f::Function, n_eval::Int)
    f()  # warm-up; result discarded.
    checksum = 0.0
    t = @elapsed for _ in 1:n_eval
        checksum += f()
    end
    return t / n_eval, checksum
end
