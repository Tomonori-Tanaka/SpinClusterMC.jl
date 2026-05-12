#!/usr/bin/env julia
#
# Benchmark the per-swap reconstruction costs incurred during parallel
# tempering (PT). During a Carlo PT checkpoint gather/scatter, each MPI
# rank:
#   1. Deserializes a JPhiSpinMC from a byte buffer (calls
#      _mpi_build_ham_and_cache + _rebuild_zlm_cache! internally).
#   2. Calls Carlo.read_checkpoint!, which calls _rebuild_zlm_cache!
#      after receiving the new spin configuration via MPI.scatter.
#
# This script isolates and times each component, reporting wall time +
# per-call allocations:
#   load_sce_hamiltonian       XML parse + SALC enumeration (startup)
#   build_local_energy_cache   Cluster instance build (startup)
#   _rebuild_zlm_cache!        Per-swap cost (every rank, every swap)
#   serialize/deserialize      Per-checkpoint cost on the coordinator
#
# CLI options:
#   --xml=/path/to/jphi.xml    Input XML (default: test/ferh_4x4x4/jphi.xml).
#   --seed=42                  RNG seed for the spin configuration.
#   --T=0.5                    JPhiSpinMC temperature [eV].
#   --seconds=2.0              BenchmarkTools per-bench wall-clock budget.
#
# Usage:
#   julia --project=benchmark benchmark/optimized/benchmark_pt_reconstruct.jl
#   julia --project=benchmark benchmark/optimized/benchmark_pt_reconstruct.jl --xml=test/ferh_4x4x4/jphi.xml

import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Printf
using Random: MersenneTwister
using Serialization
import Serialization as Ser
using SpheriCart: SphericalHarmonics
using StaticArrays: SVector
using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo
const JMCC = JPhiMagestyCarlo

include(joinpath(@__DIR__, "..", "bench_helpers.jl"))

function main()
    defaults = Dict(
        "xml"     => FIXTURES.ferh,
        "seed"    => "42",
        "T"       => "0.5",
        "seconds" => "2.0",
    )
    opts = merge(defaults, parse_kv_args(ARGS))

    xml     = abspath(opts["xml"])
    seed    = parse(Int, opts["seed"])
    T       = parse(Float64, opts["T"])
    seconds = parse(Float64, opts["seconds"])

    isfile(xml) || error("XML not found: $xml")
    seconds > 0 || error("seconds must be > 0, got: $seconds")

    println("=== benchmark_pt_reconstruct (Optimized) ===")
    println("xml     = ", xml)
    println("T       = ", T, " eV")
    println("budget  = ", seconds, " s/bench (BenchmarkTools wall-clock cap)")
    println()

    # --- 1. load_sce_hamiltonian -----------------------------------------
    r_load = run_bench(() -> load_sce_hamiltonian(xml); seconds = seconds)
    h = load_sce_hamiltonian(xml)

    # --- 2. build_local_energy_cache -------------------------------------
    r_cache = run_bench(() -> JMCC.build_local_energy_cache(h); seconds = seconds)
    cache = JMCC.build_local_energy_cache(h)

    # --- 3. _rebuild_zlm_cache! (per-swap cost) --------------------------
    rng    = MersenneTwister(seed)
    spins  = random_unit_spins(rng, h.n_atoms)
    max_l  = JMCC._max_l_in_instances(cache.instances)
    zlm    = JMCC._alloc_zlm_cache(h.n_atoms, max_l)
    sph    = SphericalHarmonics(max_l)

    rebuild_zlm = () -> begin
        for ia in 1:h.n_atoms
            JMCC._update_atom_zlm_cache!(zlm, ia, @view(spins[:, ia]), sph)
        end
    end
    r_zlm = run_bench(rebuild_zlm; seconds = seconds)

    # --- 4. serialize / deserialize round-trip ---------------------------
    params = Dict{Symbol, Any}(
        :xml_path       => xml,
        :T              => T,
        :thermalization => 0,
        :binsize        => 1,
        :seed           => seed,
    )
    mc = JPhiSpinMC(params)
    # mc.spins is a Vector{SVector{3,Float64}}; copy from the 3×N matrix.
    for i in 1:h.n_atoms
        mc.spins[i] = SVector{3, Float64}(spins[1, i], spins[2, i], spins[3, i])
    end
    mc.energy = JMCC._energy_from_instances(cache.instances, spins)

    ser_round_trip = () -> begin
        b = IOBuffer()
        Ser.serialize(b, mc)
        seekstart(b)
        Ser.deserialize(b)
    end
    r_ser = run_bench(ser_round_trip; seconds = seconds)

    # --- 5. Object sizes + serialized payload size -----------------------
    GC.gc(true)
    mem_h     = Base.summarysize(h)
    mem_cache = Base.summarysize(cache)
    mem_zlm   = Base.summarysize(zlm)
    mem_mc    = Base.summarysize(mc)
    buf = IOBuffer()
    Ser.serialize(buf, mc)
    ser_payload = position(buf)

    # ----- summary -----
    println("n_atoms     = ", h.n_atoms)
    println("n_instances = ", length(cache.instances))
    println("max_l       = ", max_l)
    println()

    @printf("%-32s %-12s %-12s %-10s %-10s\n",
        "stage", "t_min", "t_median", "allocs", "memory")
    println("-"^80)
    for (label, r) in (
        ("load_sce_hamiltonian",           r_load),
        ("build_local_energy_cache",       r_cache),
        ("_rebuild_zlm_cache! (n=$(h.n_atoms))", r_zlm),
        ("serialize+deserialize (hot)",    r_ser),
    )
        @printf("%-32s %-12s %-12s %-10d %-10s\n",
            label, fmt_time(r.t_min), fmt_time(r.t_median), r.allocs, fmt_bytes(r.memory))
    end

    println()
    println("--- object sizes (Base.summarysize) ---")
    @printf("  SCEHamiltonian                  : %s\n", fmt_bytes(mem_h))
    @printf("  LocalEnergyCache                : %s\n", fmt_bytes(mem_cache))
    @printf("  zlm_cache                       : %s\n", fmt_bytes(mem_zlm))
    @printf("  JPhiSpinMC (total)              : %s\n", fmt_bytes(mem_mc))
    @printf("  serialize payload (wire bytes)  : %s\n", fmt_bytes(ser_payload))

    println()
    println("Notes:")
    println("  _rebuild_zlm_cache! and serialize+deserialize fire on every PT swap;")
    println("  load_sce_hamiltonian and build_local_energy_cache fire once at startup.")
end

main()
