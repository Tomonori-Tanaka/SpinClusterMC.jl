#!/usr/bin/env julia
#
# Benchmark the four energy entry points exposed by the Simple submodule:
#   - total_energy(h, spins)             O(n_instances)
#   - local_energy(h, spins, i)          O(|atom_to_instance_indices[i]|)
#   - delta_local_energy(h, spins, i, S) two local_energy evaluations
#   - gradient(h, spins, i)              same loop, returns SVector{3}
#
# Reports per-call wall time (min / median), allocation count, and bytes
# allocated. The local/total time ratio should roughly match
# n_touch / n_instances; large allocation counts on `local`/`delta` are
# the smoking gun for the SH-cache rebuild on every call.
#
# CLI options:
#   --fixtures=bcc,fege,ferh   Comma-separated subset.
#   --repeat=n1,n2,n3          Supercell repeat (default 1,1,1).
#   --seconds=1.0              BenchmarkTools per-bench wall-clock budget.
#                              BenchmarkTools collects samples until either
#                              this many seconds elapse or 10 000 samples
#                              are taken, then reports min/median over them.
#   --seed=42                  RNG seed for the spin configuration.
#
# Usage:
#   julia --project=benchmark benchmark/simple/bench_energy.jl
#   julia --project=benchmark benchmark/simple/bench_energy.jl --fixtures=bcc,fege

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using Random: MersenneTwister
using StaticArrays: SVector
using SpinClusterMC
using SpinClusterMC.Simple

include(joinpath(@__DIR__, "..", "bench_helpers.jl"))

function bench_fixture(
        xml::AbstractString, repeat::NTuple{3, Int}, seed::Int; seconds::Real,
    )
    h = SpinClusterHamiltonian(xml; repeat = repeat)
    rng = MersenneTwister(seed)
    spins = random_unit_spins(rng, h.n_atoms)
    site = 1
    S_new = SVector{3, Float64}(0.0, 0.0, 1.0)

    r_total = run_bench(() -> total_energy(h, spins);                    seconds = seconds)
    r_local = run_bench(() -> local_energy(h, spins, site);              seconds = seconds)
    r_delta = run_bench(() -> delta_local_energy(h, spins, site, S_new); seconds = seconds)
    r_grad  = run_bench(() -> gradient(h, spins, site);                  seconds = seconds)

    return (;
        xml,
        n_atoms     = h.n_atoms,
        n_instances = length(h.instances),
        n_touch     = length(h.atom_to_instance_indices[site]),
        r_total, r_local, r_delta, r_grad,
    )
end

function main()
    defaults = Dict(
        "fixtures" => "bcc,fege,ferh",
        "repeat"   => "1,1,1",
        "seconds"  => "1.0",
        "seed"     => "42",
    )
    opts = merge(defaults, parse_kv_args(ARGS))

    names   = [Symbol(strip(s)) for s in split(opts["fixtures"], ",")]
    repeat  = parse_repeat_csv(opts["repeat"])
    seconds = parse(Float64, opts["seconds"])
    seed    = parse(Int, opts["seed"])
    seconds > 0 || error("seconds must be > 0, got: $seconds")

    println("=== bench_energy (Simple) ===")
    println("fixtures = ", names)
    println("repeat   = ", repeat)
    println("budget   = ", seconds, " s/bench (BenchmarkTools wall-clock cap)")
    println("seed     = ", seed)
    println()

    results = []
    for name in names
        haskey(FIXTURES, name) ||
            error("unknown fixture $(name); choose from $(keys(FIXTURES))")
        xml = getproperty(FIXTURES, name)
        print("$(rpad(string(name), 5)) ... ")
        flush(stdout)
        r = bench_fixture(xml, repeat, seed; seconds = seconds)
        push!(results, (; name, r...))
        println("done")
    end

    println()
    @printf("%-6s %-7s %-12s %-9s\n",
        "fixture", "n_atoms", "n_instances", "n_touch")
    println("-"^40)
    for r in results
        @printf("%-6s %-7d %-12d %-9d\n",
            string(r.name), r.n_atoms, r.n_instances, r.n_touch)
    end
    println()

    @printf("%-6s %-9s %-12s %-12s %-10s %-10s\n",
        "fixture", "op", "t_min", "t_median", "allocs", "memory")
    println("-"^68)
    for r in results
        for (op, br) in (
            ("total",    r.r_total),
            ("local",    r.r_local),
            ("delta",    r.r_delta),
            ("gradient", r.r_grad),
        )
            @printf("%-6s %-9s %-12s %-12s %-10d %-10s\n",
                string(r.name), op,
                fmt_time(br.t_min),
                fmt_time(br.t_median),
                br.allocs,
                fmt_bytes(br.memory),
            )
        end
    end
    println()
    println("Notes:")
    println("  n_touch = |atom_to_instance_indices[1]|, i.e. clusters touching site 1.")
    println("  local/total ratio of t_min ~ n_touch / n_instances says how much locality saves.")
    println("  High allocs on local/delta/gradient = per-call SphericalHarmonics rebuild;")
    println("  it scales with n_atoms × (max_l+1)² and is the dominant Simple bottleneck.")
end

main()
