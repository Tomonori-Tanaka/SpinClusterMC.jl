#!/usr/bin/env julia
#
# Benchmark sce_energy (reference path) vs _energy_from_instances (uncached
# Ylm) vs _energy_from_instances_cached (zlm-cache path) on a JPhi
# fixture, sweeping over supercell repeats. Reports per-call min/median
# time + allocation count + bytes, plus the speedup ratios vs the
# reference path. Useful for sanity-checking the cached path's benefits
# at different supercell sizes.
#
# CLI options:
#   --xml=/path/to/jphi.xml    Input XML path (default: test/ferh_4x4x4/jphi.xml).
#   --repeats=1x1x1,2x2x2      Comma-separated list of NxNxN repeats.
#   --seed=42                  RNG seed for the spin configuration.
#   --seconds=2.0              BenchmarkTools per-bench wall-clock budget.
#
# Usage:
#   julia --project=benchmark benchmark/optimized/benchmark_sce_reference.jl
#   julia --project=benchmark benchmark/optimized/benchmark_sce_reference.jl --repeats=1x1x1

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using Random: MersenneTwister
using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo
const JMCC = JPhiMagestyCarlo

include(joinpath(@__DIR__, "..", "bench_helpers.jl"))

function parse_repeat_x(s::AbstractString)::NTuple{3, Int}
    parts = parse.(Int, split(s, "x"))
    length(parts) == 3 || error("repeat must be NxNxN, got: $s")
    return (parts[1], parts[2], parts[3])
end

function bench_repeat(xml::String, rep::NTuple{3, Int}, seed::Int, seconds::Real)
    h     = load_sce_hamiltonian(xml; repeat = rep)
    cache = JMCC.build_local_energy_cache(h)
    derived = JMCC._get_or_build_derived(
        xml, rep, collect(eachindex(cache.body_list)), cache, h.n_atoms,
    )
    active_instances = cache.instances[derived.active_instance_indices]
    max_l = derived.max_l

    rng   = MersenneTwister(seed)
    spins = random_unit_spins(rng, h.n_atoms)

    # The cached call rebuilds the zlm cache for the current spins (same
    # cost as inside Carlo.init! on a fresh config).
    cached_call() = let zlm = JMCC._build_zlm_cache(spins, max_l)
        JMCC._energy_from_instances_cached(active_instances, zlm)
    end

    e_ref = sce_energy(h, spins)
    e_unc = JMCC._energy_from_instances(active_instances, spins)
    e_cac = cached_call()
    δ_unc = abs(e_ref - e_unc) / (abs(e_ref) + 1e-300)
    δ_cac = abs(e_ref - e_cac) / (abs(e_ref) + 1e-300)

    r_ref = run_bench(() -> sce_energy(h, spins);                              seconds = seconds)
    r_unc = run_bench(() -> JMCC._energy_from_instances(active_instances, spins); seconds = seconds)
    r_cac = run_bench(cached_call;                                              seconds = seconds)

    return (;
        rep,
        n_atoms     = h.n_atoms,
        n_instances = length(active_instances),
        n_salc      = length(h.salc_list),
        r_ref, r_unc, r_cac,
        δ_unc, δ_cac,
    )
end

function main()
    defaults = Dict(
        "xml"     => FIXTURES.ferh,
        "repeats" => "1x1x1,2x2x2",
        "seed"    => "42",
        "seconds" => "2.0",
    )
    opts = merge(defaults, parse_kv_args(ARGS))

    xml     = abspath(opts["xml"])
    reps    = [parse_repeat_x(s) for s in split(opts["repeats"], ",")]
    seed    = parse(Int, opts["seed"])
    seconds = parse(Float64, opts["seconds"])

    isfile(xml) || error("XML not found: $xml")
    seconds > 0 || error("seconds must be > 0, got: $seconds")

    println("=== benchmark_sce_reference (Optimized) ===")
    println("xml     = ", xml)
    println("repeats = ", reps)
    println("seed    = ", seed)
    println("budget  = ", seconds, " s/bench (BenchmarkTools wall-clock cap)")
    println()

    results = []
    for rep in reps
        print("repeat=$(join(rep, "x")) ... loading + building cache ...")
        flush(stdout)
        r = bench_repeat(xml, rep, seed, seconds)
        push!(results, r)
        println(" done")
    end

    println()
    @printf("%-7s %-8s %-12s %-14s %-14s %-14s %-10s %-12s\n",
        "repeat", "n_atoms", "n_instances",
        "sce_energy/call", "uncached/call", "cached/call",
        "x vs ref", "x vs ref (cac)")
    println("-"^105)
    for r in results
        rep_str = join(r.rep, "x")
        @printf("%-7s %-8d %-12d %-14s %-14s %-14s %-10.1f %-12.1f\n",
            rep_str, r.n_atoms, r.n_instances,
            fmt_time(r.r_ref.t_min),
            fmt_time(r.r_unc.t_min),
            fmt_time(r.r_cac.t_min),
            r.r_ref.t_min / r.r_unc.t_min,
            r.r_ref.t_min / r.r_cac.t_min,
        )
    end

    println()
    println("--- allocations per call ---")
    @printf("%-7s %-22s %-22s %-22s\n", "repeat", "sce_energy", "uncached", "cached")
    println("-"^90)
    for r in results
        rep_str = join(r.rep, "x")
        @printf("%-7s %-22s %-22s %-22s\n",
            rep_str,
            string(r.r_ref.allocs, " / ", fmt_bytes(r.r_ref.memory)),
            string(r.r_unc.allocs, " / ", fmt_bytes(r.r_unc.memory)),
            string(r.r_cac.allocs, " / ", fmt_bytes(r.r_cac.memory)),
        )
    end

    println()
    println("--- parity ---")
    for r in results
        rep_str = join(r.rep, "x")
        @printf("  %-7s rel-err vs ref: uncached %.2e   cached %.2e\n",
            rep_str, r.δ_unc, r.δ_cac)
    end

    println()
    println("Notes:")
    println("  sce_energy  = coupled_cluster_energy reference path")
    println("  uncached    = _energy_from_instances (recomputes Ylm for every instance)")
    println("  cached      = _energy_from_instances_cached (Ylm computed once per call via")
    println("                _build_zlm_cache, then read from a table per instance —")
    println("                same cost as inside Carlo.init!)")
end

main()
