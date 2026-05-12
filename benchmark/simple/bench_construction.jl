#!/usr/bin/env julia
#
# Benchmark Hamiltonian construction for the Simple submodule.
#
# Measures the wall-clock cost of `SpinClusterHamiltonian(xml; repeat)`, which
# internally does:
#   1. parse_jphi_xml (XML -> SALCs, basis, JPhi map)
#   2. _generate_instances (per-(SALC, translation, tile) ClusterInstance list)
#   3. build_cg_table (tesseral CG via Magesty.AngularMomentumCoupling)
#
# Usage:
#   julia benchmark/simple/bench_construction.jl
#   julia benchmark/simple/bench_construction.jl --fixtures=bcc,fege
#   julia benchmark/simple/bench_construction.jl --fixtures=bcc --repeat=2,2,2 --evals=5

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Printf
using SpinClusterMC
using SpinClusterMC.Simple

include(joinpath(@__DIR__, "fixtures.jl"))

function bench_fixture(xml::AbstractString, repeat::NTuple{3, Int}, n_eval::Int)
    # Warm-up to absorb first-call compilation.
    h0 = SpinClusterHamiltonian(xml; repeat = repeat)

    t_total = @elapsed for _ in 1:n_eval
        SpinClusterHamiltonian(xml; repeat = repeat)
    end
    t_parse = @elapsed for _ in 1:n_eval
        Simple.parse_jphi_xml(xml)
    end
    # CGTable build measured on the parsed SALC list to isolate it from XML I/O.
    data = Simple.parse_jphi_xml(xml)
    t_cg = @elapsed for _ in 1:n_eval
        Simple.build_cg_table(data.salcs)
    end

    return (;
        xml,
        repeat,
        n_atoms        = h0.n_atoms,
        n_instances    = length(h0.instances),
        n_salcs        = length(data.salcs),
        max_l          = h0.max_l,
        t_total_per    = t_total / n_eval,
        t_parse_per    = t_parse / n_eval,
        t_cg_per       = t_cg    / n_eval,
    )
end

function main()
    defaults = Dict(
        "fixtures" => "bcc,fege,ferh",
        "repeat"   => "1,1,1",
        "evals"    => "3",
    )
    opts = merge(defaults, simple_parse_args(ARGS))

    names  = [Symbol(strip(s)) for s in split(opts["fixtures"], ",")]
    repeat = simple_parse_repeat(opts["repeat"])
    n_eval = parse(Int, opts["evals"])
    n_eval > 0 || error("evals must be > 0, got: $n_eval")

    println("=== bench_construction (Simple) ===")
    println("fixtures = ", names)
    println("repeat   = ", repeat)
    println("evals    = ", n_eval)
    println()

    results = []
    for name in names
        haskey(SIMPLE_FIXTURES, name) ||
            error("unknown fixture $(name); choose from $(keys(SIMPLE_FIXTURES))")
        xml = getproperty(SIMPLE_FIXTURES, name)
        print("$(rpad(string(name), 5)) ... ")
        flush(stdout)
        r = bench_fixture(xml, repeat, n_eval)
        push!(results, (; name, r...))
        println("done")
    end

    println()
    @printf("%-6s %-8s %-12s %-7s %-6s %-14s %-14s %-14s\n",
        "fixture", "n_atoms", "n_instances", "n_salcs", "max_l",
        "total/build", "parse_xml", "cg_table")
    println("-"^92)
    for r in results
        @printf("%-6s %-8d %-12d %-7d %-6d %-14s %-14s %-14s\n",
            string(r.name), r.n_atoms, r.n_instances, r.n_salcs, r.max_l,
            simple_fmt_time(r.t_total_per),
            simple_fmt_time(r.t_parse_per),
            simple_fmt_time(r.t_cg_per),
        )
    end
end

main()
