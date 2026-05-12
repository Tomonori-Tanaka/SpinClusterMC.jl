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
# Two ratios are reported (both for time and allocation):
#   x vs ref  = simple / sce_energy            (apples-to-apples on loop shape)
#   x vs fast = simple / _energy_from_instances_cached  (vs production kernel)
#
# Allocation ratios usually tell the story more cleanly than time ratios:
# the cached fast path allocates the SH cache once per call, whereas
# Simple rebuilds it for every cluster instance. Rel-err is compared
# end-to-end as a smoke parity check (the authoritative parity tests
# live under test/parity/).
#
# CLI options:
#   --fixtures=bcc,fege,ferh   Comma-separated subset (ferh excluded by default).
#   --repeat=n1,n2,n3          Supercell repeat (default 1,1,1).
#   --seconds=2.0              BenchmarkTools per-bench wall-clock budget.
#   --seed=42                  RNG seed.
#
# Usage:
#   julia --project=benchmark benchmark/simple/bench_compare.jl
#   julia --project=benchmark benchmark/simple/bench_compare.jl --fixtures=bcc,fege

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using Random: MersenneTwister
using SpinClusterMC
using SpinClusterMC.Simple
using SpinClusterMC.JPhiMagestyCarlo
const JMCC = JPhiMagestyCarlo

include(joinpath(@__DIR__, "fixtures.jl"))

function bench_fixture(
        xml::AbstractString, repeat::NTuple{3, Int}, seed::Int; seconds::Real,
    )
    h_simple = SpinClusterHamiltonian(xml; repeat = repeat)
    h_opt    = load_sce_hamiltonian(xml; repeat = repeat)
    cache    = JMCC.build_local_energy_cache(h_opt)
    derived  = JMCC._get_or_build_derived(
        xml, repeat, collect(eachindex(cache.body_list)), cache, h_opt.n_atoms,
    )
    active_instances = cache.instances[derived.active_instance_indices]
    max_l = derived.max_l

    rng = MersenneTwister(seed)
    spins = simple_random_spins(rng, h_simple.n_atoms)

    # Each cached call rebuilds the zlm cache for the current spins (same
    # cost as inside Carlo.init! on a fresh config).
    fast_call() = let zlm = JMCC._build_zlm_cache(spins, max_l)
        JMCC._energy_from_instances_cached(active_instances, zlm)
    end

    e_s = total_energy(h_simple, spins)
    e_r = sce_energy(h_opt, spins)
    e_f = fast_call()

    rel_err_ref  = abs(e_s - e_r) / (abs(e_r) + 1e-300)
    rel_err_fast = abs(e_s - e_f) / (abs(e_f) + 1e-300)

    r_simple = simple_bench(() -> total_energy(h_simple, spins); seconds = seconds)
    r_ref    = simple_bench(() -> sce_energy(h_opt, spins);      seconds = seconds)
    r_fast   = simple_bench(fast_call;                            seconds = seconds)

    return (;
        xml,
        n_atoms     = h_simple.n_atoms,
        n_inst_s    = length(h_simple.instances),
        n_inst_o    = length(cache.instances),
        r_simple, r_ref, r_fast,
        rel_err_ref, rel_err_fast,
    )
end

ratio(a, b) = b == 0 ? NaN : a / b

"Print a ratio with adaptive precision (e.g. 0.35, 12.3, 1.77e+03)."
function fmt_ratio(x::Real)
    isnan(x) && return "NaN"
    ax = abs(x)
    ax < 10.0   && return @sprintf("%.2f", x)
    ax < 1000.0 && return @sprintf("%.1f", x)
    return @sprintf("%.2e", x)
end

function main()
    defaults = Dict(
        "fixtures" => "bcc,fege",
        "repeat"   => "1,1,1",
        "seconds"  => "2.0",
        "seed"     => "42",
    )
    opts = merge(defaults, simple_parse_args(ARGS))

    names   = [Symbol(strip(s)) for s in split(opts["fixtures"], ",")]
    repeat  = simple_parse_repeat(opts["repeat"])
    seconds = parse(Float64, opts["seconds"])
    seed    = parse(Int, opts["seed"])
    seconds > 0 || error("seconds must be > 0, got: $seconds")

    println("=== bench_compare (Simple vs Optimized) ===")
    println("fixtures = ", names)
    println("repeat   = ", repeat)
    println("budget   = ", seconds, " s/bench (BenchmarkTools wall-clock cap)")
    println("seed     = ", seed)
    println()

    results = []
    for name in names
        haskey(SIMPLE_FIXTURES, name) ||
            error("unknown fixture $(name); choose from $(keys(SIMPLE_FIXTURES))")
        xml = getproperty(SIMPLE_FIXTURES, name)
        print("$(rpad(string(name), 5)) ... ")
        flush(stdout)
        r = bench_fixture(xml, repeat, seed; seconds = seconds)
        push!(results, (; name, r...))
        println("done")
    end

    println()
    println("--- time (t_min per call) ---")
    @printf("%-6s %-12s %-12s %-12s %-10s %-10s\n",
        "fixture", "simple", "opt_ref", "opt_fast",
        "x vs ref", "x vs fast")
    println("-"^68)
    for r in results
        @printf("%-6s %-12s %-12s %-12s %-10s %-10s\n",
            string(r.name),
            simple_fmt_time(r.r_simple.t_min),
            simple_fmt_time(r.r_ref.t_min),
            simple_fmt_time(r.r_fast.t_min),
            fmt_ratio(ratio(r.r_simple.t_min, r.r_ref.t_min)),
            fmt_ratio(ratio(r.r_simple.t_min, r.r_fast.t_min)),
        )
    end

    println()
    println("--- allocations per call (count / bytes) ---")
    @printf("%-6s %-18s %-18s %-18s %-10s %-10s\n",
        "fixture", "simple", "opt_ref", "opt_fast",
        "x vs ref", "x vs fast")
    println("-"^94)
    for r in results
        s_s = @sprintf("%d / %s", r.r_simple.allocs, simple_fmt_bytes(r.r_simple.memory))
        s_r = @sprintf("%d / %s", r.r_ref.allocs,    simple_fmt_bytes(r.r_ref.memory))
        s_f = @sprintf("%d / %s", r.r_fast.allocs,   simple_fmt_bytes(r.r_fast.memory))
        @printf("%-6s %-18s %-18s %-18s %-10s %-10s\n",
            string(r.name), s_s, s_r, s_f,
            fmt_ratio(ratio(r.r_simple.allocs, r.r_ref.allocs)),
            fmt_ratio(ratio(r.r_simple.allocs, r.r_fast.allocs)),
        )
    end

    println()
    println("--- parity ---")
    @printf("%-6s %-12s %-12s %-12s\n",
        "fixture", "n_inst_s", "n_inst_o", "rel-err")
    println("-"^50)
    for r in results
        @printf("%-6s %-12d %-12d ref %.2e / fast %.2e\n",
            string(r.name), r.n_inst_s, r.n_inst_o,
            r.rel_err_ref, r.rel_err_fast,
        )
    end

    println()
    println("Notes:")
    println("  opt_ref  = JPhiMagestyCarlo.sce_energy (reference loop, similar shape to Simple)")
    println("  opt_fast = _energy_from_instances_cached including _build_zlm_cache per call")
    println("             (mirrors Carlo.init! on a new config)")
    println("  Allocation ratio is usually the cleanest 'why is Simple slow' signal.")
    println("  Rel-err < ~1e-10 is expected; parity tests under test/parity/ are authoritative.")
end

main()
