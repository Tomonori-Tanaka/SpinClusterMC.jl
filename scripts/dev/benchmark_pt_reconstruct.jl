#!/usr/bin/env julia
#
# Benchmark the per-swap reconstruction costs incurred during parallel tempering.
#
# During a Carlo PT checkpoint gather/scatter, each MPI rank:
#   1. Deserializes a JPhiSpinMC from a byte buffer (calls _mpi_build_ham_and_cache
#      + _rebuild_zlm_cache! internally).
#   2. Calls Carlo.read_checkpoint!, which calls _rebuild_zlm_cache! after
#      receiving the new spin configuration via MPI.scatter.
#
# This script isolates and times each component:
#   load_sce_hamiltonian    – full XML parse + SALC enumeration (one-time at startup)
#   build_local_energy_cache – cluster instance build (one-time at startup)
#   _rebuild_zlm_cache!     – per-swap cost (called every PT swap on every rank)
#   serialize/deserialize   – per-checkpoint cost on the coordinator rank
#
# Usage:
#   julia scripts/dev/benchmark_pt_reconstruct.jl
#   julia scripts/dev/benchmark_pt_reconstruct.jl --xml=test/ferh_4x4x4/jphi.xml
#   julia scripts/dev/benchmark_pt_reconstruct.jl --xml=test/ferh_4x4x4/jphi.xml --reps=20

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Random
using LinearAlgebra
using Serialization
import Serialization as Ser
using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo

const JMCC = JPhiMagestyCarlo

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

function parse_args(args)
    defaults = Dict(
        "xml"  => joinpath(@__DIR__, "../../test/ferh_4x4x4/jphi.xml"),
        "reps" => "20",
        "seed" => "42",
        "T"    => "0.5",
    )
    opts = copy(defaults)
    for a in args
        startswith(a, "--") || error("unknown argument format: $a")
        kv = split(a[3:end], "="; limit = 2)
        length(kv) == 2 || error("argument must be --key=value, got: $a")
        opts[kv[1]] = kv[2]
    end
    return opts
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function rand_unit_spins(rng, n::Int)
    s = randn(rng, 3, n)
    for i in 1:n; s[:, i] ./= norm(s[:, i]); end
    return s
end

function timed_median(f, reps)
    times = Vector{Float64}(undef, reps)
    result = nothing
    for i in 1:reps
        t0 = time_ns()
        result = f()
        times[i] = (time_ns() - t0) * 1e-6  # ms
    end
    return sort(times)[div(reps, 2) + 1], result
end

function print_timing(label, ms_median; width = 44)
    pad = max(1, width - length(label))
    println(label, " "^pad, ": ", round(ms_median; digits = 3), " ms")
end

function fmt_bytes(n::Integer)
    n < 1024      && return string(n, " B")
    n < 1024^2    && return string(round(n / 1024; digits = 1), " KiB")
    n < 1024^3    && return string(round(n / 1024^2; digits = 1), " MiB")
    return string(round(n / 1024^3; digits = 2), " GiB")
end

function print_mem(label, bytes; width = 44)
    pad = max(1, width - length(label))
    println(label, " "^pad, ": ", fmt_bytes(bytes))
end

# ---------------------------------------------------------------------------

function main()
    opts = parse_args(ARGS)
    xml  = abspath(opts["xml"])
    reps = parse(Int, opts["reps"])
    seed = parse(Int, opts["seed"])
    T    = parse(Float64, opts["T"])

    isfile(xml) || error("XML not found: $xml")

    println("=== benchmark_pt_reconstruct ===")
    println("xml  = ", xml)
    println("reps = ", reps)
    println("T    = ", T)
    println()

    # ------------------------------------------------------------------
    # 1. load_sce_hamiltonian  (XML parse + SALC build)
    # ------------------------------------------------------------------
    # Warm up once to compile, then measure.
    load_sce_hamiltonian(xml)  # compile
    t_load, h = timed_median(reps) do
        load_sce_hamiltonian(xml)
    end
    print_timing("load_sce_hamiltonian", t_load)

    # ------------------------------------------------------------------
    # 2. build_local_energy_cache  (cluster instance enumeration)
    # ------------------------------------------------------------------
    JMCC.build_local_energy_cache(h)  # compile
    t_cache, cache = timed_median(reps) do
        JMCC.build_local_energy_cache(h)
    end
    print_timing("build_local_energy_cache", t_cache)

    # ------------------------------------------------------------------
    # 3. _rebuild_zlm_cache!  (per-swap cost on every rank)
    # ------------------------------------------------------------------
    rng    = MersenneTwister(seed)
    spins  = rand_unit_spins(rng, h.n_atoms)
    max_l  = JMCC._max_l_in_instances(cache.instances)
    zlm    = JMCC._alloc_zlm_cache(h.n_atoms, max_l)

    # warm up
    for ia in 1:h.n_atoms
        JMCC._update_atom_zlm_cache!(zlm, ia, @view(spins[:, ia]), max_l)
    end

    t_zlm, _ = timed_median(reps) do
        for ia in 1:h.n_atoms
            JMCC._update_atom_zlm_cache!(zlm, ia, @view(spins[:, ia]), max_l)
        end
        nothing
    end
    print_timing("_rebuild_zlm_cache! (n=$(h.n_atoms))", t_zlm)

    # ------------------------------------------------------------------
    # 4. Serialize / deserialize round-trip
    #    This is what Carlo's coordinator rank does during PT checkpoint gather.
    #    The _HAM_CACHE is warm from step 1, so the deserialize cost is
    #    dominated by _rebuild_zlm_cache! rather than the XML parse.
    # ------------------------------------------------------------------
    params = Dict{Symbol,Any}(
        :xml_path       => xml,
        :T              => T,
        :thermalization => 0,
        :binsize        => 1,
        :seed           => seed,
    )
    mc = JPhiSpinMC(params)
    mc.spins .= spins
    mc.energy = h.j0 + JMCC._energy_from_instances(cache.instances, spins)

    # warm up
    buf = IOBuffer()
    Ser.serialize(buf, mc)
    seekstart(buf); Ser.deserialize(buf)

    t_ser, _ = timed_median(reps) do
        b = IOBuffer()
        Ser.serialize(b, mc)
        seekstart(b)
        Ser.deserialize(b)
        nothing
    end
    print_timing("serialize + deserialize (cache hot)", t_ser)

    # ------------------------------------------------------------------
    # 5. Memory: object sizes (deep, including all referenced heap objects)
    # ------------------------------------------------------------------
    GC.gc(true)
    mem_h     = Base.summarysize(h)
    mem_cache = Base.summarysize(cache)
    mem_zlm   = Base.summarysize(zlm)
    mem_mc    = Base.summarysize(mc)

    # Serialize payload size (how many bytes cross the wire per PT gather)
    buf_size = IOBuffer()
    Ser.serialize(buf_size, mc)
    ser_payload = position(buf_size)

    # Per-call heap allocation of each hot path
    alloc_zlm = @allocated begin
        for ia in 1:h.n_atoms
            JMCC._update_atom_zlm_cache!(zlm, ia, @view(spins[:, ia]), max_l)
        end
    end
    alloc_ser = @allocated begin
        b = IOBuffer()
        Ser.serialize(b, mc)
        seekstart(b)
        Ser.deserialize(b)
    end

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    println()
    println("n_atoms          : ", h.n_atoms)
    println("n_instances      : ", length(cache.instances))
    println("max_l            : ", max_l)

    println()
    println("--- object sizes (summarysize) ---")
    print_mem("  SCEHamiltonian", mem_h)
    print_mem("  LocalEnergyCache", mem_cache)
    print_mem("  zlm_cache", mem_zlm)
    print_mem("  JPhiSpinMC (total)", mem_mc)
    print_mem("  serialize payload (wire bytes)", ser_payload)

    println()
    println("--- per-PT-swap cost breakdown ---")
    println("  _rebuild_zlm_cache!       : ", round(t_zlm; digits = 3), " ms  |  alloc: ", fmt_bytes(alloc_zlm), "  (every swap, every rank)")
    println("  deserialize (cache hot)   : ", round(t_ser; digits = 3), " ms  |  alloc: ", fmt_bytes(alloc_ser), "  (every swap, coordinator)")

    println()
    println("--- one-time startup costs ---")
    println("  load_sce_hamiltonian      : ", round(t_load; digits = 1), " ms")
    println("  build_local_energy_cache  : ", round(t_cache; digits = 1), " ms")
end

main()
