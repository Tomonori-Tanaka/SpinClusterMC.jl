"""
    Simple

Reference implementation of the spin cluster expansion (SCE) Monte Carlo engine.

Prioritizes readability and extensibility over performance. Maintained in
parallel with `JPhiMagestyCarlo` and kept numerically consistent via parity
tests under `test/parity/`. See `docs/specs/260512-simple-impl/` for the design.
"""
module Simple

include("xml_io.jl")
include("types.jl")
include("cg.jl")
include("energy.jl")
include("external.jl")
include("spin_proposal.jl")
include("mc.jl")
include("updates/metropolis.jl")

end
