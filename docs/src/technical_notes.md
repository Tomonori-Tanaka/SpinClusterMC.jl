# Technical Notes

## SCE Hamiltonian

The SCE Hamiltonian energy for a spin configuration $\{\hat{e}_i\}$ is

$$E = J_0 + \sum_\nu J_\nu \,\Phi_\nu\!\left(\{\hat{e}_i\}\right),$$

where $J_0$ (`h.j0`) is the reference energy, $J_\nu$ (`h.jphi[ν]`) are the SCE coefficients,
and $\Phi_\nu$ are the symmetry-adapted design features computed from the basis stored in `jphi.xml`.

For the definition of $\Phi_\nu$ and the conversion to conventional exchange parameters
($J$, DM vector $\vec{D}$, anisotropic exchange $\boldsymbol{\Gamma}$),
see the [Magesty.jl Technical Notes](https://Tomonori-Tanaka.github.io/Magesty.jl/technical_notes/).

## Metropolis algorithm

`Carlo.sweep!` performs one sweep of $N_\text{atoms}$ single-spin Metropolis updates:

1. Pick a random atom $i$.
2. Propose a new spin direction $\hat{e}_i'$ (see [Spin proposals](#spin-proposals) below).
3. Compute the local energy change $\Delta E = E_\text{new} - E_\text{old}$ from the instances that involve atom $i$.
4. Accept with probability $\min(1, e^{-\Delta E / T})$.

The temperature $T$ is in the same energy unit as `JPhi@unit` in `jphi.xml` (typically eV).
To convert from Kelvin: $T = k_B T_\text{K}$ with $k_B = 8.617333262 \times 10^{-5}\ \text{eV/K}$.

## Spin proposals

Two proposal types are supported, selected by `params[:spin_theta_max]`:

### Uniform (default)

When `spin_theta_max` is absent, the new spin $\hat{e}'$ is drawn uniformly on $S^2$:

$$\hat{e}' \sim \text{Uniform}(S^2)$$

This is ergodic at any temperature but yields low acceptance at low $T$ because most proposals
are far from the current spin.

### Geodesic (local)

When `spin_theta_max = θ_\text{max} > 0`, a random unit tangent $\hat{t}$ at the current spin $\hat{e}$
is drawn (Gram–Schmidt orthogonalization of a Gaussian random vector), and the new spin is

$$\hat{e}' = \cos\theta\, \hat{e} + \sin\theta\, \hat{t}, \qquad \theta \sim \text{Uniform}[-\theta_\text{max}, \theta_\text{max}].$$

This keeps the proposed spin within a geodesic cone of half-angle $\theta_\text{max}$, yielding
much higher acceptance at low temperature.
The proposal is symmetric ($q(\hat{e}'|\hat{e}) = q(\hat{e}|\hat{e}')$), so detailed balance holds.

**Choosing $\theta_\text{max}$**: a value around $0.3$–$0.5$ rad typically gives acceptance rates
of 50–80 % near room temperature for typical SCE models. For high-temperature runs
($T \gg |J|$), uniform proposals are equally effective.

## Energy evaluation

### Cluster instances

At construction time, `build_local_energy_cache` enumerates all unique translated cluster
instances by applying each translation in `map_sym` to every cluster in the XML basis,
deduplicating by sorted atom-index tuple.
Each `ClusterInstance` stores:

- `atoms` — supercell atom indices for this translated cluster
- `cbc` — coupled-basis data (`coeff_tensor`, `coefficient`, `ls`, `multiplicity`)
- `prefactor` — pre-multiplied scalar $J_\nu \times \text{multiplicity} \times (4\pi)^{N/2}$
- `dims`, `strides` — precomputed per-site tensor dimensions and strides for the hot path

### Cached spherical harmonics

All real (tesseral) spherical harmonics $Z_{lm}(\hat{e}_i)$ are cached in a
$N_\text{atoms} \times (l_\text{max}+1)^2$ matrix `zlm_cache`.
The cache is rebuilt once after initialization and after each renormalization (every
`renorm_every` sweeps, default 1000).

### Delta-energy update

For a Metropolis move on atom $i$, only the cluster instances that contain $i$ need
to be recontracted.
`_tensor_contract_instance_cached_changed!` factorizes the tensor sum over the changed site's
$m$-indices into an inner loop, keeping the other-sites product fixed in the outer loop.
Preallocated integer buffers (`contract_other_sites`, `contract_cart_idx`) of length
$N_\text{max\_sites}$ avoid any heap allocation in the hot path.

## Supercell tiling

Given a base cell with $N_0$ atoms and repeat $(n_1, n_2, n_3)$, the total supercell has
$N = N_0 n_1 n_2 n_3$ atoms.
Atom $b$ in tile $(t_i, t_j, t_k)$ is mapped to supercell index

$$\text{ia} = b + N_0 (t_i + n_1 t_j + n_1 n_2 t_k).$$

The supercell lattice columns are $[n_1 \mathbf{a}_1,\; n_2 \mathbf{a}_2,\; n_3 \mathbf{a}_3]$.
Each cluster in the base cell is replicated on every tile using the same translation map
`map_sym` from the XML, extending the Magesty periodicity convention naturally.

## Measured observables

Each Carlo `measure!` call records the following from the current spin state:

| Observable | Formula |
|:-----------|:--------|
| `Energy` | $E / N$ (energy per atom) |
| `Energy2` | $(E / N)^2$ |
| `Magnetization` | $\|\bar{\mathbf{m}}\|$ where $\bar{\mathbf{m}} = \frac{1}{N}\sum_i \hat{e}_i$ |
| `AbsMagnetization` | same as `Magnetization` |
| `Magnetization2` | $\|\bar{\mathbf{m}}\|^2$ |
| `Magnetization4` | $\|\bar{\mathbf{m}}\|^4$ |

Derived quantities (registered via `Carlo.register_evaluables`):

| Observable | Formula |
|:-----------|:--------|
| `SpecificHeat` | $N (\langle E^2 \rangle - \langle E \rangle^2) / T^2$ |
| `BinderRatio` | $\langle m^2 \rangle^2 / \langle m^4 \rangle$ |
| `Susceptibility` | $N \langle m^2 \rangle / T$ |

## Floating-point stability

Spin vectors can drift from unit length due to floating-point accumulation.
`renorm_every` (default: 1000) controls how often all spins are renormalized and
the $Z_{lm}$ cache is rebuilt.
Set `renorm_every = 0` to disable renormalization entirely.

## Checkpointing

Carlo.jl writes HDF5 checkpoints at intervals of `checkpoint_time`.
SpinClusterMC stores `spins` (the full $3 \times N$ spin matrix) and the scalar `energy`
in each checkpoint.
The $Z_{lm}$ cache is rebuilt from spins on resume, so no additional state needs saving.
