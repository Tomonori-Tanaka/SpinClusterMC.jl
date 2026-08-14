"""
    Simple

Reference implementation of the spin cluster expansion (SCE) Monte Carlo engine.

Prioritizes readability and extensibility over performance. Maintained in
parallel with `JPhiMagestyCarlo` and kept numerically consistent via parity
tests under `test/parity/`. See `docs/specs/260512-simple-impl/` for the design.
"""
module Simple

include("xml_io.jl")
include("supercell.jl")
include("types.jl")
include("cg.jl")
include("energy.jl")
include("external.jl")
include("spin_proposal.jl")
include("mc.jl")
include("updates/metropolis.jl")

# Public surface. Internal helpers (`parse_jphi_xml`, `_propose_spin_geodesic`,
# `_compute_zlm_all`, etc.) remain accessible via qualified `Simple.foo` for
# inspection / debugging but are not part of the stable API.
export SpinClusterHamiltonian, ClusterInstance, CGTable
export total_energy, local_energy, delta_local_energy, gradient
export ExternalTerm, Zeeman, MomentModel, UniformMoment, PerSiteMoment
export init_spins
export SCEMC
export acceptance_rate, reset_acceptance!

end
