import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "..")))
using SpinClusterMC
using SpinClusterMC.JPhiMagestyCarlo
using Documenter

DocMeta.setdocmeta!(
    SpinClusterMC.JPhiMagestyCarlo,
    :DocTestSetup,
    :(using SpinClusterMC.JPhiMagestyCarlo);
    recursive = true,
)

makedocs(
    sitename = "SpinClusterMC.jl",
    modules = [SpinClusterMC.JPhiMagestyCarlo],
    remotes = nothing,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://Tomonori-Tanaka.github.io/SpinClusterMC.jl",
        mathengine = Documenter.MathJax3(),
        edit_link = "main",
        repolink = "https://github.com/Tomonori-Tanaka/SpinClusterMC.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
        "Technical Notes" => "technical_notes.md",
    ],
    warnonly = true,
    checkdocs = :none,
    doctest = false,
)

deploydocs(
    repo = "github.com/Tomonori-Tanaka/SpinClusterMC.jl",
    devbranch = "main",
)
