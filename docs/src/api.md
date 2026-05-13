# API Reference

```@meta
CurrentModule = SpinClusterMC.JPhiMagestyCarlo
```

## Main Types

### SCEHamiltonian
```@docs
SCEHamiltonian
```

### JPhiSpinMC
```@docs
JPhiSpinMC
```

## Loading and Energy Evaluation

### load_sce_hamiltonian
```@docs
load_sce_hamiltonian
```

### sce_energy
```@docs
sce_energy
```

### coupled_cluster_energy
```@docs
coupled_cluster_energy
```

## Supercell Utilities

### supercell_atom_index
```@docs
supercell_atom_index
```

## Internal Types

### ClusterInstance
```@docs
ClusterInstance
```

# Simple Module

```@meta
CurrentModule = SpinClusterMC.Simple
```

The `Simple` submodule is a readable reference implementation of the SCE
Hamiltonian and a `Carlo.AbstractMC` glue type. It mirrors the production
`JPhiMagestyCarlo` path numerically but trades performance for clarity:
per-instance loops, no body-list aggregation, no Zlm cache reuse across
calls. Use it as a parity reference or a starting point for new
algorithms.

## Hamiltonian and Types

### SpinClusterHamiltonian
```@docs
SpinClusterHamiltonian
```

### ClusterInstance
```@docs
ClusterInstance
```

### CGTable
```@docs
CGTable
```

## Energy API

```@docs
total_energy
local_energy
delta_local_energy
gradient
```

## External Field

```@docs
ExternalTerm
Zeeman
MomentModel
UniformMoment
PerSiteMoment
```

## Spin Initialization

```@docs
init_spins
```

## Monte Carlo

```@docs
SCEMC
```
