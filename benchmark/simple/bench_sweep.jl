#!/usr/bin/env julia
#
# Benchmark `Carlo.sweep!` on an `SCEMC` Monte Carlo instance.
#
# A "sweep" tries `n_atoms` single-site Metropolis updates. We report
# both ms/sweep and ms/flip so the per-fixture cost is comparable across
# different supercell sizes. The temperature is fixed at 100 K (deep in
# the ordered phase for the bundled fixtures), and the geodesic
# proposal half-width is fixed at 0.3 rad to get a moderate accept rate.
#
# Usage:
#   julia benchmark/simple/bench_sweep.jl
#   julia benchmark/simple/bench_sweep.jl --fixtures=bcc,fege --sweeps=30
#   julia benchmark/simple/bench_sweep.jl --fixtures=ferh --sweeps=2

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Printf
using Random: MersenneTwister
using Carlo
using SpinClusterMC
using SpinClusterMC.Simple

include(joinpath(@__DIR__, "fixtures.jl"))

function bench_fixture(xml::AbstractString, repeat::NTuple{3, Int}, n_sweeps::Int, seed::Int)
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

    # Warm-up: one sweep absorbs first-call compilation.
    Carlo.sweep!(mc, ctx)

    t = @elapsed for _ in 1:n_sweeps
        Carlo.sweep!(mc, ctx)
    end
    t_per_sweep = t / n_sweeps
    t_per_flip  = t_per_sweep / mc.h.n_atoms

    return (;
        xml,
        n_atoms     = mc.h.n_atoms,
        n_instances = length(mc.h.instances),
        n_sweeps,
        t_per_sweep,
        t_per_flip,
        final_energy_per_atom = mc.energy / mc.h.n_atoms,
    )
end

function main()
    defaults = Dict(
        "fixtures" => "bcc,fege",
        "repeat"   => "1,1,1",
        "sweeps"   => "20",
        "seed"     => "42",
    )
    opts = merge(defaults, simple_parse_args(ARGS))

    names    = [Symbol(strip(s)) for s in split(opts["fixtures"], ",")]
    repeat   = simple_parse_repeat(opts["repeat"])
    n_sweeps = parse(Int, opts["sweeps"])
    seed     = parse(Int, opts["seed"])
    n_sweeps > 0 || error("sweeps must be > 0, got: $n_sweeps")

    println("=== bench_sweep (Simple) ===")
    println("fixtures = ", names)
    println("repeat   = ", repeat)
    println("sweeps   = ", n_sweeps, " (after 1 warm-up)")
    println("seed     = ", seed)
    println()

    results = []
    for name in names
        haskey(SIMPLE_FIXTURES, name) ||
            error("unknown fixture $(name); choose from $(keys(SIMPLE_FIXTURES))")
        xml = getproperty(SIMPLE_FIXTURES, name)
        print("$(rpad(string(name), 5)) ... ")
        flush(stdout)
        r = bench_fixture(xml, repeat, n_sweeps, seed)
        push!(results, (; name, r...))
        println("done (E/atom = ", round(r.final_energy_per_atom; digits = 4), " eV)")
    end

    println()
    @printf("%-6s %-8s %-12s %-9s %-14s %-14s\n",
        "fixture", "n_atoms", "n_instances", "sweeps",
        "per_sweep", "per_flip")
    println("-"^74)
    for r in results
        @printf("%-6s %-8d %-12d %-9d %-14s %-14s\n",
            string(r.name), r.n_atoms, r.n_instances, r.n_sweeps,
            simple_fmt_time(r.t_per_sweep),
            simple_fmt_time(r.t_per_flip),
        )
    end
    println()
    println("Notes:")
    println("  per_sweep = average over n_sweeps; per_flip = per_sweep / n_atoms.")
    println("  ferh is excluded from the default fixture list; pass --fixtures=ferh --sweeps=2 to include it.")
end

main()
