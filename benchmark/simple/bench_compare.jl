#!/usr/bin/env julia
#
# Cross-implementation comparison: simple vs optimized.
#
# On the same fixture and the same random spin configuration, we measure:
#   - Simple.total_energy(h_simple, spins)
#   - JPhiMagestyCarlo.sce_energy(h_opt, spins)            (reference loop)
#   - JPhiMagestyCarlo._energy_from_instances_cached(...)  (cached fast path,
#       Ylm precomputed once per call via _build_zlm_cache and read from a
#       table per instance — same cost as inside Carlo.init!)
#
# The simple path is the readable per-instance loop. The optimized fast path
# uses cached Ylm + body-list aggregation. Two ratios are reported:
#   x vs ref  = simple / sce_energy            (apples-to-apples on loop shape)
#   x vs fast = simple / _energy_from_instances_cached  (vs production kernel)
#
# Rel-err is compared end-to-end to flag any cross-implementation drift
# (parity tests under `test/parity/` are the authoritative check; this is a
# smoke version that runs at benchmark time).
#
# Usage:
#   julia benchmark/simple/bench_compare.jl
#   julia benchmark/simple/bench_compare.jl --fixtures=bcc,fege --evals=50

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using Printf
using Random: MersenneTwister
using SpinClusterMC
using SpinClusterMC.Simple
using SpinClusterMC.JPhiMagestyCarlo
const JMCC = JPhiMagestyCarlo

include(joinpath(@__DIR__, "fixtures.jl"))

function bench_fixture(xml::AbstractString, repeat::NTuple{3, Int}, n_eval::Int, seed::Int)
    # Build both implementations from the same XML / repeat.
    h_simple = SpinClusterHamiltonian(xml; repeat = repeat)
    h_opt    = load_sce_hamiltonian(xml; repeat = repeat)
    cache    = JMCC.build_local_energy_cache(h_opt)
    derived  = JMCC._get_or_build_derived(
        xml, repeat, collect(eachindex(cache.body_list)), cache, h_opt.n_atoms,
    )
    active_instances = cache.instances[derived.active_instance_indices]

    rng = MersenneTwister(seed)
    spins = simple_random_spins(rng, h_simple.n_atoms)

    # Each cached call rebuilds the zlm cache for the current spins (same
    # cost as inside Carlo.init! on a fresh config). Wrap that pattern.
    fast_call() = let zlm = JMCC._build_zlm_cache(spins, derived.max_l)
        JMCC._energy_from_instances_cached(active_instances, zlm)
    end

    # Warm-up.
    e_s = total_energy(h_simple, spins)
    e_r = sce_energy(h_opt, spins)
    e_f = fast_call()

    t_simple, _ = simple_avg_time(() -> total_energy(h_simple, spins), n_eval)
    t_ref,    _ = simple_avg_time(() -> sce_energy(h_opt, spins),      n_eval)
    t_fast,   _ = simple_avg_time(fast_call,                           n_eval)

    rel_err_ref  = abs(e_s - e_r) / (abs(e_r) + 1e-300)
    rel_err_fast = abs(e_s - e_f) / (abs(e_f) + 1e-300)

    return (;
        xml,
        n_atoms     = h_simple.n_atoms,
        n_inst_s    = length(h_simple.instances),
        n_inst_o    = length(cache.instances),
        t_simple,
        t_ref,
        t_fast,
        ratio_vs_ref  = t_simple / t_ref,
        ratio_vs_fast = t_simple / t_fast,
        rel_err_ref, rel_err_fast,
    )
end

function main()
    defaults = Dict(
        "fixtures" => "bcc,fege",
        "repeat"   => "1,1,1",
        "evals"    => "20",
        "seed"     => "42",
    )
    opts = merge(defaults, simple_parse_args(ARGS))

    names  = [Symbol(strip(s)) for s in split(opts["fixtures"], ",")]
    repeat = simple_parse_repeat(opts["repeat"])
    n_eval = parse(Int, opts["evals"])
    seed   = parse(Int, opts["seed"])
    n_eval > 0 || error("evals must be > 0, got: $n_eval")

    println("=== bench_compare (Simple vs Optimized) ===")
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
    @printf("%-6s %-8s %-12s %-14s %-14s %-14s %-10s %-10s\n",
        "fixture", "n_atoms", "n_inst",
        "simple/call", "opt_ref/call", "opt_fast/call",
        "x vs ref", "x vs fast")
    println("-"^104)
    for r in results
        @printf("%-6s %-8d %-12d %-14s %-14s %-14s %-10.1f %-10.1f\n",
            string(r.name), r.n_atoms, r.n_inst_s,
            simple_fmt_time(r.t_simple),
            simple_fmt_time(r.t_ref),
            simple_fmt_time(r.t_fast),
            r.ratio_vs_ref, r.ratio_vs_fast,
        )
        @printf("        rel-err vs ref = %.2e   rel-err vs fast = %.2e   (n_inst opt = %d)\n",
            r.rel_err_ref, r.rel_err_fast, r.n_inst_o)
    end
    println()
    println("Notes:")
    println("  opt_ref  = JPhiMagestyCarlo.sce_energy (reference loop, similar shape to Simple)")
    println("  opt_fast = _energy_from_instances_cached, including a fresh _build_zlm_cache per call")
    println("             (same pattern as Carlo.init! on a new config)")
    println("  Rel-err < ~1e-10 is expected; parity tests under test/parity/ are authoritative.")
end

main()
