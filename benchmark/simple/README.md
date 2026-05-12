# benchmark/simple

Developer benchmarks for the `Simple` reference implementation. Since
Simple is the basis for ongoing performance work, the scripts here use
**`BenchmarkTools`** (per-call min/median + allocation count + bytes) so
the bottleneck is visible at a glance, not just total wall time.

These benchmarks live in their own Pkg environment
([`../Project.toml`](../Project.toml)) so `BenchmarkTools` does not
leak into the main package's dependency graph. Run every script as:

```bash
julia --project=benchmark benchmark/simple/<script>.jl
```

Or run all of them in sequence:

```bash
julia --project=benchmark benchmark/simple/runbench.jl          # full defaults
julia --project=benchmark benchmark/simple/runbench.jl --fast   # smoke
```

The first invocation will resolve the environment and pull
`BenchmarkTools`. After that the bench scripts launch in seconds.

## What each script measures

| Script | What | Notes |
|---|---|---|
| `bench_construction.jl` | XML parse → CGTable → `SpinClusterHamiltonian` | Per-stage min/median + alloc / memory. |
| `bench_energy.jl` | `total_energy`, `local_energy`, `delta_local_energy`, `gradient` | Allocation count per call surfaces the per-instance SH cache rebuild. |
| `bench_sweep.jl` | `Carlo.sweep!` on `SCEMC` | Per-sweep and per-flip wall time + allocations. |
| `bench_compare.jl` | Simple vs JPhi `sce_energy` (ref) vs `_energy_from_instances_cached` (production fast path) | Reports time **and allocation** ratios; rel-err is a parity smoke check. |

## Common CLI options

```
--fixtures=bcc,fege,ferh   Comma-separated subset of {bcc, fege, ferh}.
                           Defaults vary; ferh is excluded from sweep/compare
                           defaults because of its 840 k cluster instances.
--repeat=n1,n2,n3          Supercell repeat (default 1,1,1).
--seconds=1.0              BenchmarkTools per-bench wall-clock budget.
                           BT collects samples until either this many seconds
                           elapse or 10 000 samples are taken, then reports
                           min/median over them.
--seed=42                  RNG seed for the spin configuration.
```

## Reading the output

`simple_bench` returns a `BenchResult` with four numbers:

| Field | Meaning |
|---|---|
| `t_min` | Fastest observed timing, in seconds. **Use this as the headline number** — it filters out GC and OS noise. |
| `t_median` | Median timing. A `t_median ≫ t_min` indicates noisy or GC-heavy workloads. |
| `allocs` | Allocation count per call. |
| `memory` | Bytes allocated per call. |

For Simple specifically, **the allocation count is usually the loudest
signal** — the dominant bottleneck right now is that
`delta_local_energy` rebuilds the SphericalHarmonics cache on every
call, which scales as `n_atoms × (max_l+1)²` allocations per evaluation.
`bench_compare.jl` shows this directly: the cached fast path allocates
~50 bytes per call against the Simple path's 10⁸ bytes on `fege_2x2x2`.

## Fixture sizes

| Fixture | n_atoms | n_instances (1×1×1) | Notes |
|---|---|---|---|
| `bcc_2x2x2` | 16 | 88 | Smallest; sub-millisecond `total_energy`. |
| `fege_2x2x2` | 64 | 63 776 | Anisotropy via `Lf > 0`; second-long `total_energy` on the Simple path. |
| `ferh_4x4x4` | 128 | 839 936 | Largest; minutes per `total_energy`. Always pass `--fixtures=ferh` explicitly. |

## Relation to the optimized benchmarks

See [`../README.md`](../README.md) for the bigger picture (when to run
which tree). The optimized tree still uses plain `@elapsed` and the root
Pkg env; we may unify the two later but for now they live separately.
