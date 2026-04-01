#!/usr/bin/env julia
#
# Usage:
#   julia scripts/dev/benchmark_sce.jl
#
# Options (all optional; pass as --key=value):
#   --xml=/path/to/jphi.xml   Input XML path (default: examples/bccFe/metropolis/jphi.xml)
#   --repeat=n1,n2,n3         Supercell repeat (default: 1,1,1)
#   --evals=N                 Number of energy evaluations for averaging (default: 20)
#   --sweeps=N                Number of MC sweeps for averaging (default: 50)
#   --seed=S                  RNG seed for random spin initialization (default: 42)
#   --T=VALUE                 MC temperature in same unit as JPhi energy (default: 0.02585)
#   --spin_theta_max=VALUE    Geodesic proposal max angle [rad] (default: 0.5)
#
# Examples:
#   julia scripts/dev/benchmark_sce.jl --evals=100
#   julia scripts/dev/benchmark_sce.jl --repeat=2,2,2 --evals=50
#   julia scripts/dev/benchmark_sce.jl --xml=/tmp/jphi.xml --repeat=1,1,1 --seed=1

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Random
using LinearAlgebra
using Carlo
using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo

const JMCC = JPhiMagestyCarlo

function parse_repeat(s::AbstractString)::NTuple{3, Int}
    parts = split(s, ",")
    length(parts) == 3 || error("repeat must be n1,n2,n3 (e.g. 1,1,1), got: $s")
    vals = parse.(Int, strip.(parts))
    all(>(0), vals) || error("repeat must be positive integers, got: $s")
    return (vals[1], vals[2], vals[3])
end

function parse_args(args)
    opts = Dict{String, String}()
    for a in args
        startswith(a, "--") || error("unknown argument format: $a")
        kv = split(a[3:end], "="; limit = 2)
        length(kv) == 2 || error("argument must be --key=value, got: $a")
        opts[kv[1]] = kv[2]
    end
    return opts
end

function rand_unit_spins(rng, n::Int)
    spins = randn(rng, 3, n)
    for i in 1:n
        spins[:, i] ./= norm(spins[:, i])
    end
    return spins
end

function avg_eval_time_s(f::Function, n_eval::Int)
    checksum = 0.0
    t = @elapsed begin
        for _ in 1:n_eval
            checksum += f()
        end
    end
    return t / n_eval, checksum
end

function avg_sweep_time_s(mc, ctx, n_sweeps::Int)
    t = @elapsed begin
        for _ in 1:n_sweeps
            Carlo.sweep!(mc, ctx)
        end
    end
    return t / n_sweeps
end

function main()
    defaults = Dict(
        "xml" => joinpath(@__DIR__, "../../examples/bccFe/metropolis/jphi.xml"),
        "repeat" => "1,1,1",
        "evals" => "20",
        "sweeps" => "50",
        "seed" => "42",
        "T" => "0.02585",
        "spin_theta_max" => "0.5",
    )
    opts = merge(defaults, parse_args(ARGS))

    xml_path = abspath(opts["xml"])
    repeat = parse_repeat(opts["repeat"])
    n_eval = parse(Int, opts["evals"])
    n_sweeps = parse(Int, opts["sweeps"])
    seed = parse(Int, opts["seed"])
    T = parse(Float64, opts["T"])
    spin_theta_max = parse(Float64, opts["spin_theta_max"])

    isfile(xml_path) || error("xml file not found: $xml_path")
    n_eval > 0 || error("evals must be > 0, got: $n_eval")
    n_sweeps > 0 || error("sweeps must be > 0, got: $n_sweeps")
    T > 0 || error("T must be > 0, got: $T")
    spin_theta_max > 0 || error("spin_theta_max must be > 0, got: $spin_theta_max")

    println("=== benchmark_sce ===")
    println("xml    = ", xml_path)
    println("repeat = ", repeat)
    println("evals  = ", n_eval)
    println("sweeps = ", n_sweeps)
    println("seed   = ", seed)
    println("T      = ", T)
    println("spin_theta_max = ", spin_theta_max)
    println()

    t_load = @elapsed h = load_sce_hamiltonian(xml_path; repeat = repeat)
    t_cache = @elapsed cache = JMCC.build_local_energy_cache(h)
    println("load_sce_hamiltonian : ", round(1e3 * t_load; digits = 2), " ms")
    println("build_local_energy_cache : ", round(1e3 * t_cache; digits = 2), " ms")

    rng = MersenneTwister(seed)
    spins = rand_unit_spins(rng, h.n_atoms)

    # Warm-up to reduce first-call compilation effects in timing.
    _ = sce_energy(h, spins)
    _ = h.j0 + JMCC._energy_from_instances(cache.instances, spins)

    t_ref, sum_ref = avg_eval_time_s(() -> sce_energy(h, spins), n_eval)
    t_fast, sum_fast = avg_eval_time_s(() -> h.j0 + JMCC._energy_from_instances(cache.instances, spins), n_eval)

    e_ref = sce_energy(h, spins)
    e_fast = h.j0 + JMCC._energy_from_instances(cache.instances, spins)
    diff = abs(e_ref - e_fast)

    println()
    println("n_atoms = ", h.n_atoms)
    println("instances = ", length(cache.instances))
    println()
    println("sce_energy (reference) avg : ", round(1e3 * t_ref; digits = 3), " ms/eval")
    println("from_instances (fast) avg  : ", round(1e3 * t_fast; digits = 3), " ms/eval")
    println("speedup (reference/fast)   : ", round(t_ref / t_fast; digits = 3), "x")
    println("abs(E_ref - E_fast)        : ", diff)
    println("checksum ref/fast          : ", sum_ref, " / ", sum_fast)

    params = Dict{Symbol, Any}(
        :xml_path => xml_path,
        :repeat => repeat,
        :T => T,
        :spin_theta_max => spin_theta_max,
    )
    mc = JPhiSpinMC(params)
    ctx = Carlo.MCContext(n_sweeps, 0, MersenneTwister(seed), nothing)

    Carlo.init!(mc, ctx, params)
    Carlo.sweep!(mc, ctx) # warm-up
    t_sweep = avg_sweep_time_s(mc, ctx, n_sweeps)

    println()
    println("MC sweep avg               : ", round(1e3 * t_sweep; digits = 3), " ms/sweep")
    println("MC final energy            : ", mc.energy)
end

main()
