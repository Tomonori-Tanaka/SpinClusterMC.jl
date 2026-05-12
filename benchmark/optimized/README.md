# benchmark/optimized

Developer benchmarks for the optimized SCE path (`JPhiMagestyCarlo`).

Run from the repository root:

```bash
julia benchmark/optimized/benchmark_sce.jl
julia benchmark/optimized/benchmark_sce_reference.jl
julia benchmark/optimized/benchmark_pt_reconstruct.jl
```

Each script activates the repository project, so no extra `--project` flag is
needed. Pass `--xml=...`, `--repeat=...`, `--evals=...`, etc. — see the script
header comments for available options.

The companion `benchmark/simple/` (TBD) will mirror this layout for the simple
reference implementation, so the two implementations can be compared on the
same fixtures.
