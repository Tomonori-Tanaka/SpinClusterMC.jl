module SpinClusterMC

include("supercell_common.jl")
include("JPhiMagestyCarlo.jl")
include("simple/Simple.jl")

using .JPhiMagestyCarlo
using .Simple

export JPhiMagestyCarlo, Simple

end
