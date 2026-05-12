#!/usr/bin/env julia
#
# Benchmark the four energy entry points exposed by the Simple submodule:
#   - total_energy(h, spins)             O(n_instances)
#   - local_energy(h, spins, i)          O(|atom_to_instance_indices[i]|)
#   - delta_local_energy(h, spins, i, S) same as local but evaluates twice
#   - gradient(h, spins, i)              same loop, returns SVector{3}
#
# All four call the same kernel under the hood; the per-call cost ratio is
# what is interesting (it tells you how much locality the
# `atom_to_instance_indices` table is buying you on each fixture).
#
# Usage:
#   julia benchmark/simple/bench_energy.jl
#   julia benchmark/simple/bench_energy.jl --fixtures=bcc,fege --evals=200

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Printf
using Random: MersenneTwister
using StaticArrays: SVector
using SpinClusterMC
using SpinClusterMC.Simple

include(joinpath(@__DIR__, "fixtures.jl"))

function bench_fixture(xml::AbstractString, repeat::NTuple{3, Int}, n_eval::Int, seed::Int)
    h = SpinClusterHamiltonian(xml; repeat = repeat)
    rng = MersenneTwister(seed)
    spins = simple_random_spins(rng, h.n_atoms)
    site = 1
    S_new = SVector{3, Float64}(0.0, 0.0, 1.0)

    # Warm-up
    total_energy(h, spins)
    local_energy(h, spins, site)
    delta_local_energy(h, spins, site, S_new)
    gradient(h, spins, site)

    t_total, _    = simple_avg_time(() -> total_energy(h, spins),               n_eval)
    t_local, _    = simple_avg_time(() -> local_energy(h, spins, site),         n_eval)
    t_delta, _    = simple_avg_time(() -> delta_local_energy(h, spins, site, S_new), n_eval)
    # gradient returns SVector; checksum its sum so the call isn't elided.
    # Wrap the loop call to share the warm-up convention with simple_avg_time.
    grad_scalar = () -> begin
        g = gradient(h, spins, site)
        g[1] + g[2] + g[3]
    end
    t_grad, _ = simple_avg_time(grad_scalar, n_eval)

    n_touch = length(h.atom_to_instance_indices[site])
    return (;
        xml,
        n_atoms     = h.n_atoms,
        n_instances = length(h.instances),
        n_touch,
        t_total_per = t_total,
        t_local_per = t_local,
        t_delta_per = t_delta,
        t_grad_per  = t_grad,
    )
end

function main()
    defaults = Dict(
        "fixtures" => "bcc,fege,ferh",
        "repeat"   => "1,1,1",
        "evals"    => "50",
        "seed"     => "42",
    )
    opts = merge(defaults, simple_parse_args(ARGS))

    names  = [Symbol(strip(s)) for s in split(opts["fixtures"], ",")]
    repeat = simple_parse_repeat(opts["repeat"])
    n_eval = parse(Int, opts["evals"])
    seed   = parse(Int, opts["seed"])
    n_eval > 0 || error("evals must be > 0, got: $n_eval")

    println("=== bench_energy (Simple) ===")
    println("fixtures = ", names)
    println("repeat   = ", repeat)
    println("evals    = ", n_eval)
    println("seed     = ", seed)
    println()

    results = []
    for name in names
        haskey(SIMPLE_FIXTURES, name) ||
            error("unknown fixture $(name); choose from $(keys(SIMPLE_FIXTURES))")
        xml = getproperty(SIMPLE_FIXTURES, name)
        print("$(rpad(string(name), 5)) ... ")
        flush(stdout)
        r = bench_fixture(xml, repeat, n_eval, seed)
        push!(results, (; name, r...))
        println("done")
    end

    println()
    @printf("%-6s %-8s %-12s %-9s %-14s %-14s %-14s %-14s\n",
        "fixture", "n_atoms", "n_instances", "n_touch",
        "total/call", "local/call", "delta/call", "gradient/call")
    println("-"^102)
    for r in results
        @printf("%-6s %-8d %-12d %-9d %-14s %-14s %-14s %-14s\n",
            string(r.name), r.n_atoms, r.n_instances, r.n_touch,
            simple_fmt_time(r.t_total_per),
            simple_fmt_time(r.t_local_per),
            simple_fmt_time(r.t_delta_per),
            simple_fmt_time(r.t_grad_per),
        )
    end
    println()
    println("Notes:")
    println("  n_touch = |atom_to_instance_indices[1]|, i.e. clusters touching site 1.")
    println("  local/total ratio ~ n_touch / n_instances says how much locality saves.")
    println("  delta ~ 2 × local (it evaluates local_energy at both old and new spin).")
end

main()
