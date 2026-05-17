"""
Single-spin Metropolis sweep for `SCEMC`.

One sweep performs `n_atoms` independent flip attempts: each step picks a
site uniformly at random, proposes a new spin direction by a geodesic step
around the current spin (`_propose_spin_geodesic`), evaluates the
incremental energy change including any registered external term, and
accepts or rejects according to the Metropolis criterion `exp(-ΔE / T)`
with `T` in eV (set by the `SCEMC` constructor from the user-supplied
Kelvin input).

Additional update schemes (overrelaxation, heatbath, HMC, Wolff, …) live in
their own files in this directory and plug into `Carlo.sweep!` via the
dispatch site in `mc.jl`. They are not implemented in v1.
"""

using LinearAlgebra: norm
using Random: rand
using StaticArrays: SVector

"""
    metropolis_sweep!(mc::SCEMC, ctx)

Perform `mc.h.n_atoms` Metropolis flip attempts on `mc.spins`. On each
accepted move `mc.energy` is incremented by `ΔE` so the running total stays
in sync without a full recomputation; the periodic drift check in
`Carlo.sweep!` reconciles round-off accumulation.
"""
function metropolis_sweep!(mc::SCEMC, ctx::Carlo.MCContext)
    n = mc.h.n_atoms
    theta_max = mc.theta_max
    rng = ctx.rng
    @inbounds for _ in 1:n
        i = rand(rng, 1:n)
        S_old = SVector{3, Float64}(mc.spins[1, i], mc.spins[2, i], mc.spins[3, i])
        S_new = _propose_spin_geodesic(rng, S_old, theta_max)

        # Energy delta: SCE + (optional) external term. The dispatch helper
        # `_external_delta_local` (defined in mc.jl) avoids the `isnothing`
        # branch in this hot loop and keeps JET happy on the Union-typed
        # `mc.external` field.
        ΔE = delta_local_energy(mc.h, mc.spins, i, S_new) +
             _external_delta_local(mc.external, mc.spins, i, S_new)

        # Metropolis criterion. `ΔE ≤ 0` short-circuits the exponential.
        if ΔE ≤ 0.0 || rand(rng) < exp(-ΔE / mc.T)
            mc.spins[1, i] = S_new[1]
            mc.spins[2, i] = S_new[2]
            mc.spins[3, i] = S_new[3]
            mc.energy += ΔE
        end
    end
    return nothing
end
