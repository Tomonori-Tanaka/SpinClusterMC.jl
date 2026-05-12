#!/usr/bin/env julia
#
# Benchmark Hamiltonian construction for the Simple submodule.
#
# Measures the wall-clock cost of `SpinClusterHamiltonian(xml; repeat)`,
# which internally does:
#   1. parse_jphi_xml (XML -> SALCs, basis, JPhi map)
#   2. _generate_instances (per-(SALC, translation, tile) ClusterInstance list)
#   3. build_cg_table (tesseral CG via Magesty.AngularMomentumCoupling)
#
# CLI options:
#   --fixtures=bcc,fege,ferh   Comma-separated subset.
#   --repeat=n1,n2,n3          Supercell repeat (default 1,1,1).
#   --seconds=1.0              BenchmarkTools per-bench wall-clock budget.
#                              BenchmarkTools collects samples until either
#                              this many seconds elapse or 10 000 samples
#                              are taken, then reports min/median over them.
#
# Usage:
#   julia --project=benchmark benchmark/simple/bench_construction.jl
#   julia --project=benchmark benchmark/simple/bench_construction.jl --fixtures=bcc,fege
#   julia --project=benchmark benchmark/simple/bench_construction.jl --fixtures=bcc --repeat=2,2,2

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using SpinClusterMC
using SpinClusterMC.Simple

include(joinpath(@__DIR__, "fixtures.jl"))

function bench_fixture(xml::AbstractString, repeat::NTuple{3, Int}; seconds::Real)
    # Inspect a sample build so we can print shape info alongside timings.
    h0 = SpinClusterHamiltonian(xml; repeat = repeat)
    data = Simple.parse_jphi_xml(xml)

    r_total = simple_bench(() -> SpinClusterHamiltonian(xml; repeat = repeat); seconds = seconds)
    r_parse = simple_bench(() -> Simple.parse_jphi_xml(xml);                   seconds = seconds)
    r_cg    = simple_bench(() -> Simple.build_cg_table(data.salcs);            seconds = seconds)

    return (;
        xml,
        repeat,
        n_atoms     = h0.n_atoms,
        n_instances = length(h0.instances),
        n_salcs     = length(data.salcs),
        max_l       = h0.max_l,
        r_total, r_parse, r_cg,
    )
end

function main()
    defaults = Dict(
        "fixtures" => "bcc,fege,ferh",
        "repeat"   => "1,1,1",
        "seconds"  => "1.0",
    )
    opts = merge(defaults, simple_parse_args(ARGS))

    names   = [Symbol(strip(s)) for s in split(opts["fixtures"], ",")]
    repeat  = simple_parse_repeat(opts["repeat"])
    seconds = parse(Float64, opts["seconds"])
    seconds > 0 || error("seconds must be > 0, got: $seconds")

    println("=== bench_construction (Simple) ===")
    println("fixtures = ", names)
    println("repeat   = ", repeat)
    println("budget   = ", seconds, " s/bench (BenchmarkTools wall-clock cap)")
    println()

    results = []
    for name in names
        haskey(SIMPLE_FIXTURES, name) ||
            error("unknown fixture $(name); choose from $(keys(SIMPLE_FIXTURES))")
        xml = getproperty(SIMPLE_FIXTURES, name)
        print("$(rpad(string(name), 5)) ... ")
        flush(stdout)
        r = bench_fixture(xml, repeat; seconds = seconds)
        push!(results, (; name, r...))
        println("done")
    end

    println()
    @printf("%-6s %-7s %-12s %-7s %-6s\n",
        "fixture", "n_atoms", "n_instances", "n_salcs", "max_l")
    println("-"^45)
    for r in results
        @printf("%-6s %-7d %-12d %-7d %-6d\n",
            string(r.name), r.n_atoms, r.n_instances, r.n_salcs, r.max_l)
    end
    println()

    @printf("%-6s %-9s %-12s %-12s %-10s %-10s\n",
        "fixture", "stage", "t_min", "t_median", "allocs", "memory")
    println("-"^68)
    for r in results
        for (stage, br) in (
            ("build", r.r_total),
            ("parse", r.r_parse),
            ("cg",    r.r_cg),
        )
            @printf("%-6s %-9s %-12s %-12s %-10d %-10s\n",
                string(r.name), stage,
                simple_fmt_time(br.t_min),
                simple_fmt_time(br.t_median),
                br.allocs,
                simple_fmt_bytes(br.memory),
            )
        end
    end
end

main()
