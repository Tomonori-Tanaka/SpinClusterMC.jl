# benchmark/

Developer benchmarks for SpinClusterMC.jl. Two parallel trees, one per
implementation:

```
benchmark/
├── optimized/   # JPhiMagestyCarlo (production path; cached Ylm, body-list aggregation)
└── simple/      # Simple submodule (readable reference; per-instance loop)
```

The two trees use different timing tools today:

- **`optimized/`** uses plain `@elapsed` against the root project env;
  no extra deps.
- **`simple/`** uses `BenchmarkTools` (per-call min/median + allocation
  count + bytes) against [`./Project.toml`](Project.toml), a standalone
  Pkg environment that pins `BenchmarkTools` (and any future profiling
  tooling) without touching the main package's dependency graph.

The split exists because Simple is now the basis for ongoing
performance work — we need per-call allocation tracking to identify
bottlenecks — whereas the optimized scripts are more about coarse
production-path timing. Both still share fixtures (`test/bcc_2x2x2`,
`test/fege_2x2x2`, `test/ferh_4x4x4`) and a `--fixtures=` style CLI so
numbers are directly comparable across trees.

Pick the tree by which implementation you are profiling.

## When to use which

| You are … | Run |
|---|---|
| profiling the production MC path | `benchmark/optimized/*` |
| comparing the reference and production paths on the same fixture | `benchmark/simple/bench_compare.jl` |
| sanity-checking that the Simple submodule is not regressing in absolute terms | `benchmark/simple/*` (other scripts) |
| documenting "what does the optimized path actually buy us?" | `benchmark/simple/bench_compare.jl` |

The two trees share fixtures (`test/bcc_2x2x2`, `test/fege_2x2x2`,
`test/ferh_4x4x4`) so numbers are directly comparable across trees.

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
