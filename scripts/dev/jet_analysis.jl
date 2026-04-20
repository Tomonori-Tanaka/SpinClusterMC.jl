#!/usr/bin/env julia
#
# Run JET.jl static analysis on SpinClusterMC.JPhiMagestyCarlo.
#
# Activates a temporary environment so it is self-contained and does not
# modify the project's Manifest.toml.
#
# Usage:
#   julia scripts/dev/jet_analysis.jl          # default: report_package
#   julia scripts/dev/jet_analysis.jl --fail   # exit 1 if any reports found
#
# The analysis is scoped to our own modules (SpinClusterMC and
# JPhiMagestyCarlo) so that issues inside Magesty, Carlo, EzXML, etc.
# are not surfaced here.

import Pkg

# Use the dedicated jet sub-project (scripts/jet/Project.toml).
# This pins JET to 0.9.x which is compatible with Julia 1.12,
# avoiding the Revise.SigInfo breakage in JET 0.11.
Pkg.activate(joinpath(@__DIR__, "../jet"))
Pkg.instantiate()

using JET
using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo

fail_on_reports = "--fail" in ARGS

println("=== JET analysis: SpinClusterMC ===")
println()

result = report_package(
    "SpinClusterMC";
    # Only report issues originating in our own code.
    target_modules = (SpinClusterMC, SpinClusterMC.JPhiMagestyCarlo),
)

show(result)
println()

n = length(JET.get_reports(result))
println()
println("Total reports: ", n)

if fail_on_reports && n > 0
    exit(1)
end
