#!/usr/bin/env julia
#
# Benchmark `Carlo.sweep!` on an `SCEMC` Monte Carlo instance.
#
# A "sweep" tries `n_atoms` single-site Metropolis updates. We report
# per-sweep min/median time, allocations, and bytes — and the per-flip
# derived figures (t / n_atoms, allocs / n_atoms). T = 100 K and
# spin_theta_max = 0.3 rad fix the move statistics so per-fixture
# comparisons are meaningful.
#
# CLI options:
#   --fixtures=bcc,fege,ferh   Comma-separated subset (ferh excluded by default).
#   --repeat=n1,n2,n3          Supercell repeat (default 1,1,1).
#   --seconds=2.0              BenchmarkTools per-bench wall-clock budget.
#                              Sweeps are slow so we default higher than
#                              the energy benches; bump further on big fixtures.
#   --seed=42                  RNG seed.
#
# Usage:
#   julia --project=benchmark benchmark/simple/bench_sweep.jl
#   julia --project=benchmark benchmark/simple/bench_sweep.jl --fixtures=bcc,fege

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using Random: MersenneTwister
using Carlo
using SpinClusterMC
using SpinClusterMC.Simple

include(joinpath(@__DIR__, "..", "bench_helpers.jl"))

function bench_fixture(
        xml::AbstractString, repeat::NTuple{3, Int}, seed::Int; seconds::Real,
    )
    params = Dict{Symbol, Any}(
        :T              => 100.0,            # K
        :xml_path       => xml,
        :repeat         => repeat,
        :external       => nothing,
        :update_scheme  => :metropolis,
        :spin_theta_max => 0.3,
        :renorm_every   => 10_000,           # disable mid-bench renorm noise
        :thermalization => 0,
        :binsize        => 1,
        :seed           => seed,
        :initial_spins  => :random,
    )

    mc = SCEMC(params)
    ctx = Carlo.MCContext{MersenneTwister}(params)
    Carlo.init!(mc, ctx, params)

    r_sweep = run_bench(() -> Carlo.sweep!(mc, ctx); seconds = seconds)

    return (;
        xml,
        n_atoms     = mc.h.n_atoms,
        n_instances = length(mc.h.instances),
        r_sweep,
        final_energy_per_atom = mc.energy / mc.h.n_atoms,
    )
end

function main()
    defaults = Dict(
        "fixtures" => "bcc,fege",
        "repeat"   => "1,1,1",
        "seconds"  => "2.0",
        "seed"     => "42",
    )
    opts = merge(defaults, parse_kv_args(ARGS))

    names   = [Symbol(strip(s)) for s in split(opts["fixtures"], ",")]
    repeat  = parse_repeat_csv(opts["repeat"])
    seconds = parse(Float64, opts["seconds"])
    seed    = parse(Int, opts["seed"])
    seconds > 0 || error("seconds must be > 0, got: $seconds")

    println("=== bench_sweep (Simple) ===")
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
        println("done (E/atom = ", round(r.final_energy_per_atom; digits = 4), " eV)")
    end

    println()
    @printf("%-6s %-7s %-12s %-12s %-12s %-10s %-10s\n",
        "fixture", "n_atoms", "n_instances",
        "t_min/sweep", "t_med/sweep", "allocs", "memory")
    println("-"^76)
    for r in results
        @printf("%-6s %-7d %-12d %-12s %-12s %-10d %-10s\n",
            string(r.name), r.n_atoms, r.n_instances,
            fmt_time(r.r_sweep.t_min),
            fmt_time(r.r_sweep.t_median),
            r.r_sweep.allocs,
            fmt_bytes(r.r_sweep.memory),
        )
    end
    println()

    @printf("%-6s %-12s %-12s\n", "fixture", "t_min/flip", "allocs/flip")
    println("-"^36)
    for r in results
        n = r.n_atoms
        @printf("%-6s %-12s %-12.1f\n",
            string(r.name),
            fmt_time(r.r_sweep.t_min / n),
            r.r_sweep.allocs / n,
        )
    end
    println()
    println("Notes:")
    println("  per-flip values = per-sweep / n_atoms. allocs/flip > 1000 indicates")
    println("  the SH cache rebuild per delta_local_energy call dominates the sweep.")
    println("  ferh is excluded from defaults; --fixtures=ferh --seconds=30 to include.")
end

main()
