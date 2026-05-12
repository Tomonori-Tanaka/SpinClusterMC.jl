#!/usr/bin/env julia
#
# Benchmark the optimized JPhi MC engine on a single fixture: load the
# Hamiltonian, build the LocalEnergyCache, measure the reference vs
# fast-path energy, and time a Metropolis sweep.
#
# CLI options:
#   --xml=/path/to/jphi.xml    Input XML path (default: test/bcc_2x2x2/jphi.xml).
#   --repeat=n1,n2,n3          Supercell repeat (default 1,1,1).
#   --seed=42                  RNG seed for the spin configuration.
#   --T=0.02585                MC temperature [eV] (this is JPhiSpinMC's API; the
#                              optimized engine takes eV directly, unlike the
#                              Kelvin convention used by the Simple submodule).
#   --spin_theta_max=0.5       Geodesic proposal half-width [rad].
#   --seconds=2.0              BenchmarkTools per-bench wall-clock budget.
#                              BT collects samples until either this many seconds
#                              elapse or 10 000 samples are taken, then reports
#                              min/median over them.
#
# Usage:
#   julia --project=benchmark benchmark/optimized/benchmark_sce.jl
#   julia --project=benchmark benchmark/optimized/benchmark_sce.jl --repeat=2,2,2

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using Random: MersenneTwister
using Carlo
using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo
const JMCC = JPhiMagestyCarlo

include(joinpath(@__DIR__, "..", "bench_helpers.jl"))

function main()
    defaults = Dict(
        "xml"            => FIXTURES.bcc,
        "repeat"         => "1,1,1",
        "seed"           => "42",
        "T"              => "0.02585",
        "spin_theta_max" => "0.5",
        "seconds"        => "2.0",
    )
    opts = merge(defaults, parse_kv_args(ARGS))

    xml            = abspath(opts["xml"])
    repeat         = parse_repeat_csv(opts["repeat"])
    seed           = parse(Int, opts["seed"])
    T              = parse(Float64, opts["T"])
    spin_theta_max = parse(Float64, opts["spin_theta_max"])
    seconds        = parse(Float64, opts["seconds"])

    isfile(xml)       || error("xml file not found: $xml")
    T > 0             || error("T must be > 0, got: $T")
    spin_theta_max > 0 || error("spin_theta_max must be > 0, got: $spin_theta_max")
    seconds > 0       || error("seconds must be > 0, got: $seconds")

    println("=== benchmark_sce (Optimized) ===")
    println("xml            = ", xml)
    println("repeat         = ", repeat)
    println("seed           = ", seed)
    println("T              = ", T, " eV")
    println("spin_theta_max = ", spin_theta_max, " rad")
    println("budget         = ", seconds, " s/bench (BenchmarkTools wall-clock cap)")
    println()

    # ----- construction -----
    r_load = run_bench(() -> load_sce_hamiltonian(xml; repeat = repeat); seconds = seconds)
    h = load_sce_hamiltonian(xml; repeat = repeat)
    r_cache = run_bench(() -> JMCC.build_local_energy_cache(h); seconds = seconds)
    cache = JMCC.build_local_energy_cache(h)

    # ----- energy: reference vs uncached fast path -----
    rng = MersenneTwister(seed)
    spins = random_unit_spins(rng, h.n_atoms)
    e_ref  = sce_energy(h, spins)
    e_fast = JMCC._energy_from_instances(cache.instances, spins)
    diff = abs(e_ref - e_fast)

    r_ref  = run_bench(() -> sce_energy(h, spins);                          seconds = seconds)
    r_fast = run_bench(() -> JMCC._energy_from_instances(cache.instances, spins); seconds = seconds)

    # ----- MC sweep -----
    params = Dict{Symbol, Any}(
        :xml_path       => xml,
        :repeat         => repeat,
        :T              => T,
        :spin_theta_max => spin_theta_max,
    )
    mc = JPhiSpinMC(params)
    ctx = Carlo.MCContext(1, 0, MersenneTwister(seed), nothing)
    Carlo.init!(mc, ctx, params)
    r_sweep = run_bench(() -> Carlo.sweep!(mc, ctx); seconds = seconds)

    # ----- summary -----
    println("n_atoms                = ", h.n_atoms)
    println("instances              = ", length(cache.instances))
    println()

    @printf("%-32s %-12s %-12s %-10s %-10s\n",
        "stage", "t_min", "t_median", "allocs", "memory")
    println("-"^80)
    for (label, r) in (
        ("load_sce_hamiltonian",            r_load),
        ("build_local_energy_cache",        r_cache),
        ("sce_energy (reference)",          r_ref),
        ("_energy_from_instances (fast)",   r_fast),
        ("MC sweep (Carlo.sweep!)",         r_sweep),
    )
        @printf("%-32s %-12s %-12s %-10d %-10s\n",
            label,
            fmt_time(r.t_min),
            fmt_time(r.t_median),
            r.allocs,
            fmt_bytes(r.memory),
        )
    end

    println()
    println("speedup (reference / fast) : ", round(r_ref.t_min / r_fast.t_min; digits = 2), "x")
    println("abs(E_ref - E_fast)        : ", diff)
    println("MC final energy            : ", mc.energy)
end

main()
