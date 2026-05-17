module SpinClusterMC

include("JPhiMagestyCarlo.jl")
include("simple/Simple.jl")

using .JPhiMagestyCarlo
using .Simple

export JPhiMagestyCarlo, Simple

end
