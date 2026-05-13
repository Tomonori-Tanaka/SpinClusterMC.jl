# M10 Benchmark Baseline

Reference baseline taken at the close of the `simple-impl` spec
(M1–M10). The Simple submodule is now feature-complete with parity
gates green; this snapshot is what subsequent performance work
(SH-cache reuse, etc.) will be compared against.

## Run conditions

| Field | Value |
|---|---|
| Date | 2026-05-13 |
| Branch | `feature/simple-impl` |
| Commit | `cf25754` (`M10: ferh parity gate + completion verification`) |
| Host | `MacBook-Air-6.local` |
| CPU | Apple M4 (4 threads) |
| OS | Darwin arm64 (`arm64-apple-darwin24.0.0`) |
| Julia | 1.12.6 |
| BenchmarkTools | per-bench budget = 1.0–2.0 s wall (set by each script's default; see banners) |

All numbers below are produced by `benchmark/{simple,optimized}/*.jl`
using the shared `benchmark/Project.toml` env and `bench_helpers.jl`
helpers. `t_min` = fastest sample (treat as the headline); `t_median`
= median; `allocs` / `memory` = per-call allocation count / bytes.

Fixture sizes:

| Fixture | n_atoms | n_instances (1×1×1) |
|---|---|---|
| `bcc_2x2x2` | 16 | 88 |
| `fege_2x2x2` | 64 | 63 776 |
| `ferh_4x4x4` | 128 | 840 256 |

Only the optimized `benchmark_sce_reference.jl` was constrained to
`--repeats=1x1x1` (the default also runs `2x2x2`, but the
6.7 M-instance build on ferh × 2×2×2 is a scaling probe rather than a
baseline number). Every other script ran with shipped defaults.

---

## Simple submodule

### bench_construction (XML → CGTable → SpinClusterHamiltonian)

```
=== bench_construction (Simple) ===
fixtures = [:bcc, :fege, :ferh]
repeat   = (1, 1, 1)
budget   = 1.0 s/bench (BenchmarkTools wall-clock cap)

fixture n_atoms n_instances  n_salcs max_l
---------------------------------------------
bcc    16      88           2       1
fege   64      63776        734     2
ferh   128     840256       488     2

fixture stage     t_min        t_median     allocs     memory
--------------------------------------------------------------------
bcc    build     300.33 µs    338.62 µs    28052      1.1 MiB
bcc    parse     246.83 µs    279.12 µs    24561      963.1 KiB
bcc    cg        35.96 µs     43.33 µs     2077       110.4 KiB
fege   build     38.84 ms     41.18 ms     1279992    70.4 MiB
fege   parse     28.44 ms     30.32 ms     343963     16.8 MiB
fege   cg        347.92 µs    391.46 µs    29653      1.4 MiB
ferh   build     339.71 ms    379.99 ms    18695056   945.3 MiB
ferh   parse     79.44 ms     115.03 ms    6522264    251.1 MiB
ferh   cg        4.19 ms      4.68 ms      241728     13.0 MiB
```

### bench_energy (total / local / delta / gradient)

```
=== bench_energy (Simple) ===
fixtures = [:bcc, :fege, :ferh]
repeat   = (1, 1, 1)
budget   = 1.0 s/bench (BenchmarkTools wall-clock cap)
seed     = 42

fixture n_atoms n_instances  n_touch
----------------------------------------
bcc    16      88           11
fege   64      63776        1993
ferh   128     840256       19534

fixture op        t_min        t_median     allocs     memory
--------------------------------------------------------------------
bcc    total     179.46 µs    189.83 µs    8484       183.5 KiB
bcc    local     24.88 µs     26.21 µs     1092       24.6 KiB
bcc    delta     46.46 µs     49.38 µs     2157       48.1 KiB
bcc    gradient  26.5 µs      28.0 µs      1348       47.8 KiB
fege   total     1.89 s       1.89 s       84874798   1.76 GiB
fege   local     56.71 ms     59.84 ms     2652382    56.3 MiB
fege   delta     114.11 ms    117.67 ms    5304733    112.5 MiB
fege   gradient  60.89 ms     65.73 ms     3194336    137.2 MiB
ferh   total     12.07 s      12.07 s      455122222  9.56 GiB
ferh   local     276.07 ms    289.51 ms    10608460   228.3 MiB
ferh   delta     565.35 ms    572.11 ms    21216889   456.6 MiB
ferh   gradient  333.92 ms    334.27 ms    12455049   499.9 MiB
```

`n_touch = |atom_to_instance_indices[1]|`. The `t_min(local) /
t_min(total)` ratio (e.g. fege 0.030 vs `n_touch/n_instances` = 0.031)
shows the locality optimization works. The high `allocs` on every
`local` / `delta` / `gradient` call is the SphericalHarmonics rebuild
that scales with `n_atoms × (max_l + 1)²` — the dominant Simple
bottleneck.

### bench_sweep (Carlo.sweep! on SCEMC)

```
=== bench_sweep (Simple) ===
fixtures = [:bcc, :fege]
repeat   = (1, 1, 1)
budget   = 2.0 s/bench (BenchmarkTools wall-clock cap)
seed     = 42

fixture n_atoms n_instances  t_min/sweep  t_med/sweep  allocs     memory
----------------------------------------------------------------------------
bcc    16      88           715.42 µs    792.29 µs    34512      769.5 KiB
fege   64      63776        7.58 s       7.58 s       339502912  7.03 GiB

fixture t_min/flip   allocs/flip
------------------------------------
bcc    44.71 µs     2157.0
fege   118.47 ms    5304733.0
```

ferh is excluded from defaults (run `--fixtures=ferh --seconds=30` to
include). The per-flip alloc count (~2k on bcc, ~5M on fege) is the SH
rebuild firing once per `delta_local_energy` call.

### bench_compare (Simple vs JPhi reference vs JPhi cached fast path)

```
=== bench_compare (Simple vs Optimized) ===
fixtures = [:bcc, :fege]
repeat   = (1, 1, 1)
budget   = 2.0 s/bench (BenchmarkTools wall-clock cap)
seed     = 42

--- time (t_min per call) ---
fixture simple       opt_ref      opt_fast     x vs ref   x vs fast
--------------------------------------------------------------------
bcc    177.46 µs    555.38 µs    3.83 µs      0.32       46.4
fege   1.89 s       3.64 s       15.11 ms     0.52       124.9

--- allocations per call (count / bytes) ---
fixture simple             opt_ref            opt_fast           x vs ref   x vs fast
----------------------------------------------------------------------------------------------
bcc    8484 / 183.5 KiB   24446 / 751.6 KiB  38 / 2.4 KiB       0.35       223.3
fege   84874798 / 1.76 GiB 190922758 / 5.2 GiB 48 / 7.8 KiB       0.44       1.77e+06

--- parity ---
fixture n_inst_s     n_inst_o     rel-err
--------------------------------------------------
bcc    88           88           ref 0.00e+00 / fast 0.00e+00
fege   63776        63776        ref 2.90e-15 / fast 0.00e+00
```

- `opt_ref` = `JPhiMagestyCarlo.sce_energy` (similar shape to Simple — both walk every instance and recompute Ylm inside the loop). Simple is actually faster here, likely from SpheriCart batching and slightly tighter Julia code.
- `opt_fast` = `_energy_from_instances_cached` + `_build_zlm_cache` per call (mirrors `Carlo.init!`'s cost on a new config). This is the right comparison for an MC inner loop. **Simple is ~125× slower in wall time and ~1.77e+06× higher in allocations on fege**.

---

## Optimized submodule (JPhiMagestyCarlo)

### benchmark_sce (end-to-end on bcc fixture)

```
=== benchmark_sce (Optimized) ===
xml            = /Users/tomorin/Packages/SpinClusterMC.jl/test/bcc_2x2x2/jphi.xml
repeat         = (1, 1, 1)
seed           = 42
T              = 0.02585 eV
spin_theta_max = 0.5 rad
budget         = 2.0 s/bench (BenchmarkTools wall-clock cap)

n_atoms                = 16
instances              = 88

stage                            t_min        t_median     allocs     memory
--------------------------------------------------------------------------------
load_sce_hamiltonian             281.42 µs    295.5 µs     25096      989.6 KiB
build_local_energy_cache         24.54 µs     27.62 µs     1992       150.0 KiB
sce_energy (reference)           550.88 µs    573.58 µs    24446      751.6 KiB
_energy_from_instances (fast)    532.08 µs    551.62 µs    22704      684.8 KiB
MC sweep (Carlo.sweep!)          3.38 µs      3.56 µs      0          0 B

speedup (reference / fast) : 1.04x
abs(E_ref - E_fast)        : 0.0
MC final energy            : -59.16896182974548
```

The headline: **`Carlo.sweep!` on bcc is 3.38 µs / 0 bytes per sweep**.
That is the long-term target for Simple to approach.

### benchmark_sce_reference (sce_energy vs uncached vs cached on ferh)

```
=== benchmark_sce_reference (Optimized) ===
xml     = /Users/tomorin/Packages/SpinClusterMC.jl/test/ferh_4x4x4/jphi.xml
repeats = [(1, 1, 1)]
seed    = 42
budget  = 2.0 s/bench (BenchmarkTools wall-clock cap)

repeat  n_atoms  n_instances  sce_energy/call uncached/call  cached/call    x vs ref   x vs ref (cac)
---------------------------------------------------------------------------------------------------------
1x1x1   128      840256       22.92 s        22.52 s        105.13 ms      1.0        218.0

--- allocations per call ---
repeat  sce_energy             uncached               cached
------------------------------------------------------------------------------------------
1x1x1   1094128832 / 31.37 GiB 1078870656 / 30.81 GiB 49 / 13.8 KiB

--- parity ---
  1x1x1   rel-err vs ref: uncached 3.14e-14   cached 3.14e-14
```

On the largest fixture, even the optimized reference path takes ~23 s
and 31 GiB to evaluate one full-cell energy. The cached fast path
(same SH cache reuse trick Simple lacks) gets that to 105 ms with 14
KiB — **218× faster, ~10⁶× fewer allocations**. The Simple
implementation today is in the same regime as `uncached` here.

### benchmark_pt_reconstruct (per-PT-swap reconstruction cost on ferh)

```
=== benchmark_pt_reconstruct (Optimized) ===
xml     = /Users/tomorin/Packages/SpinClusterMC.jl/test/ferh_4x4x4/jphi.xml
T       = 0.5 eV
budget  = 2.0 s/bench (BenchmarkTools wall-clock cap)

n_atoms     = 128
n_instances = 840256
max_l       = 2

stage                            t_min        t_median     allocs     memory
--------------------------------------------------------------------------------
load_sce_hamiltonian             127.34 ms    140.89 ms    6906947    276.8 MiB
build_local_energy_cache         459.54 ms    536.05 ms    12924842   997.6 MiB
_rebuild_zlm_cache! (n=128)      833.9 ns     858.3 ns     0          0 B
serialize+deserialize (hot)      436.15 ms    519.6 ms     17035006   1.38 GiB

--- object sizes (Base.summarysize) ---
  SCEHamiltonian                  : 9.7 MiB
  LocalEnergyCache                : 191.3 MiB
  zlm_cache                       : 9.0 KiB
  JPhiSpinMC (total)              : 354.6 MiB
  serialize payload (wire bytes)  : 3.3 KiB
```

For PT, the per-swap cost on optimized ferh is dominated by
serialize/deserialize (~436 ms), not by `_rebuild_zlm_cache!` (834 ns
/ 0 allocs). The wire payload is only 3.3 KiB (just spin
configurations) — the rest of the deserialize time is rebuilding the
local-energy cache on the receiving rank.

---

## Headline takeaways

1. **Simple is now ~125× slower than the optimized fast path** on the
   reference fege workload (1.89 s vs 15.11 ms per `total_energy`),
   with a **~1.77 × 10⁶ allocation gap**. The root cause is the
   per-call SphericalHarmonics rebuild (`(l_max + 1)² × n_atoms`
   allocations per call) — the same trick the optimized cached path
   uses via `_build_zlm_cache`.
2. **`Carlo.sweep!` on Simple bcc is 715 µs / 34 k allocs**, vs the
   optimized 3.38 µs / 0 allocs. That ~210× gap on the smallest
   fixture is the cleanest target for any future Simple perf work.
3. **`Carlo.sweep!` (optimized) is alloc-free per sweep**. Any future
   Simple optimization should aim for the same: zero per-sweep
   allocations once the warm caches are in place.
4. **PT reconstruction cost is dominated by deserialize**, not by
   `_rebuild_zlm_cache!` (834 ns). If PT is ever in the hot path,
   look at serialize layout, not Ylm caching.
5. The requirements.md target of `simple / optimized ≈ 10–30×` is
   **not met** (we are at 46–125× wall time, 1.77 × 10⁶× allocs).
   The bottleneck is unambiguously identified (SH-cache reuse), so
   this is left as the first future-work item; the next baseline
   should be taken on the same machine immediately after that
   change lands so the delta is uncontaminated by host drift.

## Reproducing

```bash
# Simple (defaults exercise all 3 fixtures for construction/energy)
julia --project=benchmark benchmark/simple/bench_construction.jl
julia --project=benchmark benchmark/simple/bench_energy.jl
julia --project=benchmark benchmark/simple/bench_sweep.jl
julia --project=benchmark benchmark/simple/bench_compare.jl

# Optimized
julia --project=benchmark benchmark/optimized/benchmark_sce.jl
julia --project=benchmark benchmark/optimized/benchmark_sce_reference.jl --repeats=1x1x1
julia --project=benchmark benchmark/optimized/benchmark_pt_reconstruct.jl
```

This baseline is tracked in git via `git add -f` despite the global
`benchmark/results/*.md` gitignore — the ignore exists so day-to-day
runs do not clutter `git status`, but milestone snapshots are kept
explicitly so the perf trend can be reconstructed later.
