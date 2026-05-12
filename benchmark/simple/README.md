# benchmark/simple

Developer benchmarks for the `Simple` reference implementation. They share
the optimized-side style (`@elapsed` + `--key=value` CLI args, no
`BenchmarkTools` dependency) so the two trees can be read side by side.

Run any single script from the repository root — each one activates the
project for you:

```bash
julia benchmark/simple/bench_construction.jl
julia benchmark/simple/bench_energy.jl
julia benchmark/simple/bench_sweep.jl
julia benchmark/simple/bench_compare.jl
```

Or run them all in sequence:

```bash
julia benchmark/simple/runbench.jl          # full default sizes
julia benchmark/simple/runbench.jl --fast   # smoke (bcc/fege only, small counts)
```

## What each script measures

| Script | What | Notes |
|---|---|---|
| `bench_construction.jl` | XML parse → CGTable → `SpinClusterHamiltonian` build | Reports parse, CG, total build time per fixture. |
| `bench_energy.jl` | `total_energy`, `local_energy`, `delta_local_energy`, `gradient` | All four call the same SCE kernel; ratios show how much locality the `atom_to_instance_indices` table saves. |
| `bench_sweep.jl` | `Carlo.sweep!` on an `SCEMC` instance | Reports ms/sweep and ms/flip; `T = 100 K`, `spin_theta_max = 0.3 rad`. |
| `bench_compare.jl` | Simple `total_energy` vs optimized `sce_energy` (reference) vs `_energy_from_instances_cached` (production fast path) | Same XML / spin config; rel-err is a parity smoke check. |

## Common CLI options

Every script accepts the same shared subset:

```
--fixtures=bcc,fege,ferh   Comma-separated subset of {bcc, fege, ferh}.
                           Defaults vary; ferh is excluded from sweep/compare
                           defaults because of its 840 k cluster instances.
--repeat=n1,n2,n3          Supercell repeat (default: 1,1,1).
--seed=42                  RNG seed.
```

Plus per-script knobs (`--evals`, `--sweeps`) — see the script header.

## Fixture sizes

| Fixture | n_atoms | n_instances (1×1×1) | Notes |
|---|---|---|---|
| `bcc_2x2x2` | 16 | 88 | Smallest; under-1 ms `total_energy`. |
| `fege_2x2x2` | 64 | 63 776 | Anisotropy via `Lf > 0`; second-long `total_energy`. |
| `ferh_4x4x4` | 128 | 839 936 | Largest; minutes per `total_energy` on the Simple path. Always pass `--fixtures=ferh` explicitly. |

## Why no BenchmarkTools

The optimized benchmarks deliberately use plain `@elapsed` so they have
no extra dependency footprint. The simple benchmarks follow the same
convention. Add `BenchmarkTools` ad-hoc from an REPL if you need
per-call statistics; we do not depend on it.

## Relation to the optimized benchmarks

See [`../README.md`](../README.md) for the bigger picture (when to run
which tree).
