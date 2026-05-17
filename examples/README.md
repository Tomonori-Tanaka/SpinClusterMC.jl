# SpinClusterMC — Examples

Runnable example scripts for the `Simple` submodule. Each `0N_*.jl` file is
self-contained and started with `julia --project=. examples/0N_*.jl`.

The fixtures (`bcc_2x2x2`, `fege_2x2x2`, `ferh_4x4x4`) live under `test/` and
are pointed at by `joinpath(@__DIR__, "..", "test", "...", "jphi.xml")`.

---

## 30-second quickstart

```bash
julia --project=. examples/01_quickstart.jl
```

This loads the `bcc_2x2x2` Hamiltonian, builds an `SCEMC` at 100 K, runs 500
Metropolis sweeps, and prints the initial / final energy and magnetization.
No file output. ~5 s on a recent laptop.

---

## 30-minute reading order

Each example is ~100 lines and includes a top-of-file note distinguishing
**pedagogical** code (manual MC loops, raw-sample collection, CSV output)
from the **production** path (Carlo's job runner, binned `measure!`,
HDF5 results). Read in numeric order:

| # | File | What it shows |
|---|------|---------------|
| 1 | `01_quickstart.jl` | Minimum viable run: `SCEMC` constructor, `Carlo.init!`/`sweep!`, reading `mc.energy` / `mc.spins`. |
| 2 | `02_cooling_run.jl` | Simulated annealing: build `SCEMC` once at high T, walk down a temperature ladder by mutating `mc.T`, carry the spin state across steps. Writes `cooling_results.csv` via `DelimitedFiles.writedlm`. |
| 3 | `03_anisotropy_demo.jl` | Direct energy evaluation (no MC) on `fege_2x2x2` to surface the cubic anisotropy from `Lf > 0` SALCs. Compares E(+x̂), E(+ŷ), E(+ẑ), E(diagonal). |
| 4 | `04_initial_spin_presets.jl` | Every form `init_spins(spec, n_atoms, base_n_atoms; rng)` accepts — `Symbol`, `Tuple`, `AbstractVector`, `AbstractMatrix` (`3 × base_n_atoms` or `3 × n_atoms`), `AbstractDict`. Bypasses `SCEMC` and just reports `total_energy` for each. |
| 5 | `05_custom_observable.jl` | `params[:extra_measure]` callback in action: per-sublattice magnetization for Fe (atoms 1..64) and Rh (atoms 65..128) on `ferh_4x4x4`. Shows the indirect call chain `Carlo.measure!(mc, ctx) -> mc.extra_measure(mc, ctx)`. |

---

## Conventions used across examples

### Temperature

`params[:T]` is in **Kelvin**. The `SCEMC` constructor converts to eV
internally via `BOLTZMANN_EV_PER_KELVIN ≈ 8.6173e-5 eV/K`; everything
downstream (Metropolis acceptance, `register_evaluables`) runs in eV. See
`CLAUDE.md` § "物理規約".

### Magnetization observable

`m = (1/n) Σ_i S_i`, where `S_i` is the **unit spin direction**. `|m| ∈
[0, 1]` is a dimensionless order parameter and **does not** include
per-site moment magnitudes (those live in `MomentModel`). To compute
SI-unit magnetization users multiply by the moment magnitudes — see the
final block of `05_custom_observable.jl`.

### RNG / reproducibility

Each example sets `params[:seed] = 42` and threads a deterministic RNG
through `Carlo.MCContext{MersenneTwister}(params)`. `MersenneTwister`
streams are stable across Julia minor versions; `Random.default_rng()`
(Xoshiro) is not. See the `init_spins` docstring under "Reproducibility".

### Pedagogical vs production

Each script's header lists the places where it deviates from how a real
simulation is usually structured (manual loops vs Carlo's job runner,
raw `mc.energy` access vs `Carlo.measure!`, CSV via `writedlm` vs Carlo's
HDF5 results). The deviations always favor *readability* over operational
realism.

---

## Output files

`02_cooling_run.jl` produces `cooling_results.csv` next to the script.
This file is `.gitignore`d (the example regenerates it on each run).
Other examples print to stdout only.

---

## CI smoke test

The fast examples (01, 03, 04) are run automatically in CI via the
`examples-smoke` Makefile target:

```bash
make examples-smoke
```

`02` (longer cooling run) and `05` (ferh + 840k cluster instances) are
excluded from the smoke list because of their wall time on the current
Simple implementation. They are still expected to run from a checkout —
see the per-file headers for the recommended workflow.
