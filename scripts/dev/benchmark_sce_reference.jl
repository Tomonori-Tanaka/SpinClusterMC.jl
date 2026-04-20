#!/usr/bin/env julia
#
# Benchmark sce_energy (reference / coupled_cluster_energy path) for ferh_4x4x4.
#
# Measures wall time for N evaluations of sce_energy and _energy_from_instances
# across multiple supercell sizes to quantify the O(n_salc × n_trans) cost scaling.
#
# Usage:
#   julia scripts/dev/benchmark_sce_reference.jl
#   julia scripts/dev/benchmark_sce_reference.jl --xml=test/ferh_4x4x4/jphi.xml --evals=10
#   julia scripts/dev/benchmark_sce_reference.jl --evals=5 --repeats=1x1x1,2x2x2

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Random
using LinearAlgebra
using Printf
using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo

const JMCC = JPhiMagestyCarlo

function parse_args(args)
    defaults = Dict(
        "xml"     => joinpath(@__DIR__, "../../test/ferh_4x4x4/jphi.xml"),
        "evals"   => "100",
        "seed"    => "42",
        "repeats" => "1x1x1,2x2x2",
    )
    opts = copy(defaults)
    for a in args
        startswith(a, "--") || error("unknown argument: $a")
        kv = split(a[3:end], "="; limit = 2)
        length(kv) == 2 || error("expected --key=value, got: $a")
        opts[kv[1]] = kv[2]
    end
    return opts
end

function parse_repeat(s::AbstractString)::NTuple{3,Int}
    p = parse.(Int, split(s, "x"))
    length(p) == 3 || error("repeat must be NxNxN, got: $s")
    return (p[1], p[2], p[3])
end

function rand_unit_spins(rng, n::Int)
    s = randn(rng, 3, n)
    for i in 1:n; s[:, i] ./= norm(s[:, i]); end
    return s
end

function fmt_time(s::Float64)
    s < 1.0   && return string(round(s * 1e3; digits = 2), " ms")
    s < 60.0  && return string(round(s; digits = 2), " s")
    return string(round(s / 60; digits = 2), " min")
end

function bench_repeat(xml, rep, n_eval, seed)
    h = load_sce_hamiltonian(xml; repeat = rep)
    cache = JMCC.build_local_energy_cache(h)
    rng = MersenneTwister(seed)
    spins = rand_unit_spins(rng, h.n_atoms)

    # warm up (compile + instruction cache)
    sce_energy(h, spins)
    JMCC._energy_from_instances(cache.instances, spins)

    checksum_ref = 0.0
    t_ref = @elapsed for _ in 1:n_eval
        checksum_ref += sce_energy(h, spins)
    end

    checksum_fast = 0.0
    t_fast = @elapsed for _ in 1:n_eval
        checksum_fast += JMCC._energy_from_instances(cache.instances, spins)
    end

    return (;
        rep,
        n_atoms      = h.n_atoms,
        n_instances  = length(cache.instances),
        n_salc       = length(h.salc_list),
        t_ref_total  = t_ref,
        t_ref_per    = t_ref / n_eval,
        t_fast_total = t_fast,
        t_fast_per   = t_fast / n_eval,
        speedup      = t_ref / t_fast,
        checksum_ref, checksum_fast,
    )
end

function main()
    opts  = parse_args(ARGS)
    xml   = abspath(opts["xml"])
    n_eval = parse(Int, opts["evals"])
    seed  = parse(Int, opts["seed"])
    reps  = [parse_repeat(s) for s in split(opts["repeats"], ",")]

    isfile(xml) || error("XML not found: $xml")

    println("=== benchmark_sce_reference ===")
    println("xml    = ", xml)
    println("evals  = ", n_eval, " per repeat")
    println("seed   = ", seed)
    println()

    results = []
    for rep in reps
        print("repeat=$(join(rep, "x")) ... loading + building cache ...")
        flush(stdout)
        r = bench_repeat(xml, rep, n_eval, seed)
        push!(results, r)
        println(" done")
    end

    println()
    println("repeat  n_atoms  n_salc    n_instances  sce_energy/call  fast/call   speedup  total($n_eval evals)")
    println("------  -------  ------    -----------  ---------------  ---------   -------  --------")
    for r in results
        rep_str   = join(r.rep, "x")
        @printf("%-7s %-8d %-9d %-12d %-16s %-11s %-8.1f %s\n",
            rep_str, r.n_atoms, r.n_salc, r.n_instances,
            fmt_time(r.t_ref_per), fmt_time(r.t_fast_per), r.speedup,
            fmt_time(r.t_ref_total),
        )
    end

    println()
    println("Notes:")
    println("  sce_energy      = coupled_cluster_energy reference path, O(n_salc × n_trans × cluster_size)")
    println("  fast path       = _energy_from_instances (cached Zlm tensor contraction)")
    println("  total($n_eval evals) = wall time to call sce_energy $n_eval times (no parallelism)")
end

main()
