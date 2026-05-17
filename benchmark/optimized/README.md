# benchmark/optimized

Developer benchmarks for the optimized SCE path (`JPhiMagestyCarlo`).

Run from the repository root with the shared benchmark env:

```bash
julia --project=benchmark benchmark/optimized/benchmark_sce.jl
julia --project=benchmark benchmark/optimized/benchmark_sce_reference.jl
julia --project=benchmark benchmark/optimized/benchmark_pt_reconstruct.jl
```

Each script accepts `--key=value` CLI options — see the header comments
for the full per-script list. All scripts share the BenchmarkTools-based
output (per-call min/median time + allocation count + bytes) and the
[`../bench_helpers.jl`](../bench_helpers.jl) helpers.

## What each script measures

| Script | What |
|---|---|
| `benchmark_sce.jl` | End-to-end optimized run on one fixture: load + cache build + `sce_energy` (reference) + `_energy_from_instances` (uncached fast) + `Carlo.sweep!`. |
| `benchmark_sce_reference.jl` | `sce_energy` vs `_energy_from_instances` (uncached) vs `_energy_from_instances_cached` (production fast path) at one or more supercell repeats. |
| `benchmark_pt_reconstruct.jl` | Per-PT-swap reconstruction cost: `load_sce_hamiltonian`, `build_local_energy_cache`, `_rebuild_zlm_cache!`, and `serialize`/`deserialize` round-trip. |

## Common CLI options

```
--xml=/path/to/jphi.xml    Input XML (default per script).
--repeat=n1,n2,n3          Supercell repeat (benchmark_sce.jl).
--repeats=1x1x1,2x2x2      Repeat sweep (benchmark_sce_reference.jl).
--seed=42                  RNG seed.
--seconds=2.0              BenchmarkTools per-bench wall-clock budget.
                           BT collects samples until either this many seconds
                           elapse or 10 000 samples are taken, then reports
                           min/median over them.
```

Plus per-script knobs (`--T`, `--spin_theta_max`) — see headers.

## Temperature unit (optimized vs Simple)

`JPhiSpinMC` reads `params[:T]` in **eV**, not Kelvin. The `Simple.SCEMC`
constructor takes Kelvin and converts internally — see `CLAUDE.md`
§ "物理規約" for the wider convention. The `--T=` CLI flag on
`benchmark_sce.jl` and `benchmark_pt_reconstruct.jl` takes eV directly
to match JPhiSpinMC's API.

## Relation to the simple benchmarks

See [`../README.md`](../README.md) for the bigger picture.
