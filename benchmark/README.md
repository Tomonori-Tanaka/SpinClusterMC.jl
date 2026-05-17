# benchmark/

Developer benchmarks for SpinClusterMC.jl. Two parallel trees, one per
implementation:

```
benchmark/
├── Project.toml      # Standalone Pkg env: BenchmarkTools + Carlo + SpinClusterMC (path) + ...
├── bench_helpers.jl  # Shared helpers: FIXTURES, run_bench, fmt_time, fmt_bytes, ...
├── optimized/        # JPhiMagestyCarlo (production path; cached Ylm, body-list aggregation)
└── simple/           # Simple submodule (readable reference; per-instance loop)
```

Both trees:

- Activate the same Pkg env: `julia --project=benchmark <script>.jl`,
  or inside a script `Pkg.activate(joinpath(@__DIR__, ".."))`.
- Use **BenchmarkTools** for per-call min/median time + allocation
  count + bytes. Allocation tracking is usually the clearest "why is
  this slow" signal in Julia, so it gets equal billing with wall time
  in every output table.
- Share `bench_helpers.jl` (fixtures, CLI parsing, `run_bench`,
  formatters). Subtree-specific helpers stay inline in each script.

## When to use which

| You are … | Run |
|---|---|
| profiling the production MC path | `benchmark/optimized/*` |
| understanding why the reference (Simple) implementation is slow | `benchmark/simple/*` |
| comparing the reference and production paths on the same fixture | `benchmark/simple/bench_compare.jl` |
| measuring per-PT-swap reconstruction cost | `benchmark/optimized/benchmark_pt_reconstruct.jl` |

The two trees share fixtures (`test/bcc_2x2x2`, `test/fege_2x2x2`,
`test/ferh_4x4x4`) and the `--fixtures=` / `--seconds=` CLI shape, so
numbers are directly comparable across trees.

## What is *not* benchmarked here

These scripts are for **developer inspection**, not regression gating.
There is no recorded baseline, no CI integration, and the timings
fluctuate with the host machine. If you need to defend a performance
change in a PR, run the relevant scripts before and after locally and
paste both outputs into the PR description.

For numerical regression gating, use the test suite instead
(`make test`, `make test-slow`).

## Per-tree details

- [`benchmark/optimized/README.md`](optimized/README.md)
- [`benchmark/simple/README.md`](simple/README.md)
