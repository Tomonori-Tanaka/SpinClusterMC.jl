#!/usr/bin/env julia
#
# Benchmark sce_energy (reference path) vs _energy_from_instances (uncached) vs
# _energy_from_instances_cached (zlm-cache path) for ferh_4x4x4.
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
    derived = JMCC._get_or_build_derived(
        xml, rep, collect(eachindex(cache.body_list)), cache, h.n_atoms,
    )
    rng = MersenneTwister(seed)
    spins = rand_unit_spins(rng, h.n_atoms)
    active_instances = cache.instances[derived.active_instance_indices]

    # build zlm cache once (reused across calls in the cached path)
    zlm_cache = JMCC._build_zlm_cache(spins, derived.max_l)

    # warm-up (compile + instruction cache)
    sce_energy(h, spins)
    JMCC._energy_from_instances(active_instances, spins)
    JMCC._energy_from_instances_cached(active_instances, zlm_cache)

    # --- reference path ---
    checksum_ref = 0.0
    t_ref = @elapsed for _ in 1:n_eval
        checksum_ref += sce_energy(h, spins)
    end

    # --- uncached fast path (Ylm recomputed per instance) ---
    checksum_unc = 0.0
    t_unc = @elapsed for _ in 1:n_eval
        checksum_unc += JMCC._energy_from_instances(active_instances, spins)
    end

    # --- cached fast path (Ylm precomputed once per call, table-lookup per instance) ---
    checksum_cac = 0.0
    t_cac = @elapsed for _ in 1:n_eval
        # rebuild zlm_cache from spins (same cost as _rebuild_zlm_cache! in init!)
        zlm = JMCC._build_zlm_cache(spins, derived.max_l)
        checksum_cac += JMCC._energy_from_instances_cached(active_instances, zlm)
    end

    return (;
        rep,
        n_atoms         = h.n_atoms,
        n_instances     = length(active_instances),
        n_salc          = length(h.salc_list),
        t_ref_per       = t_ref / n_eval,
        t_unc_per       = t_unc / n_eval,
        t_cac_per       = t_cac / n_eval,
        speedup_unc     = t_ref / t_unc,
        speedup_cac     = t_ref / t_cac,
        checksum_ref, checksum_unc, checksum_cac,
    )
end

function main()
    opts   = parse_args(ARGS)
    xml    = abspath(opts["xml"])
    n_eval = parse(Int, opts["evals"])
    seed   = parse(Int, opts["seed"])
    reps   = [parse_repeat(s) for s in split(opts["repeats"], ",")]

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
    @printf("%-7s %-8s %-12s %-18s %-18s %-18s %-12s %-10s\n",
        "repeat", "n_atoms", "n_instances",
        "sce_energy/call", "uncached/call", "cached/call",
        "x vs sce", "x vs sce (cached)")
    println("-"^105)
    for r in results
        rep_str = join(r.rep, "x")
        @printf("%-7s %-8d %-12d %-18s %-18s %-18s %-12.1f %-10.1f\n",
            rep_str, r.n_atoms, r.n_instances,
            fmt_time(r.t_ref_per), fmt_time(r.t_unc_per), fmt_time(r.t_cac_per),
            r.speedup_unc, r.speedup_cac,
        )
        # checksum agreement
        δ_unc = abs(r.checksum_ref - r.checksum_unc) / (abs(r.checksum_ref) + 1e-300)
        δ_cac = abs(r.checksum_ref - r.checksum_cac) / (abs(r.checksum_ref) + 1e-300)
        @printf("        checksum vs sce: uncached rel-err=%.2e  cached rel-err=%.2e\n",
            δ_unc, δ_cac)
    end

    println()
    println("Notes:")
    println("  sce_energy  = coupled_cluster_energy reference path, O(n_salc × n_trans × cluster_size)")
    println("  uncached    = _energy_from_instances: recomputes Ylm for every instance (no cache)")
    println("  cached      = _energy_from_instances_cached: Ylm computed once per atom via _build_zlm_cache,")
    println("                then read from table per instance — same cost as Carlo.init!")
end

main()
