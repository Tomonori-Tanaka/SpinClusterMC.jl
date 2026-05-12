# 04_initial_spin_presets.jl
#
# Walk through every form `init_spins(spec, n_atoms, base_n_atoms; rng)`
# accepts as `spec`, building each one against the `bcc_2x2x2` fixture
# and reporting its total energy. The point is to make the dispatch
# table from the `init_spins` docstring concrete and runnable.
#
#     julia --project=. examples/04_initial_spin_presets.jl
#
# Pedagogical vs production notes
# -------------------------------
# * `init_spins` is normally invoked indirectly through `Carlo.init!`,
#   which passes `params[:initial_spins]` straight through. This example
#   calls it directly so the input → output mapping is visible.
# * Each call passes an explicit `rng = MersenneTwister(42)` so the
#   `:random` case is reproducible. In a Carlo job the rng is `ctx.rng`;
#   see the `init_spins` docstring for the reproducibility notes.

using Random: MersenneTwister
using StaticArrays: SVector

using SpinClusterMC
using SpinClusterMC.Simple

const XML = joinpath(@__DIR__, "..", "test", "bcc_2x2x2", "jphi.xml")

h = SpinClusterHamiltonian(XML)
n_atoms = h.n_atoms
base_n = h.base_n_atoms
println("Loaded $(n_atoms) atoms (base_n_atoms=$(base_n))")
println()

# Per-atom mean-spin magnitude |m| (same convention as examples 01 and 02).
function mean_magnetization(spins::AbstractMatrix{<:Real})::Float64
    n = size(spins, 2)
    mx = sum(@view spins[1, :]) / n
    my = sum(@view spins[2, :]) / n
    mz = sum(@view spins[3, :]) / n
    return sqrt(mx * mx + my * my + mz * mz)
end

function report(label, spins)
    E = total_energy(h, spins)
    println(
        rpad(label, 38) *
        " E = $(round(E; digits = 6)) eV  " *
        "|m| = $(round(mean_magnetization(spins); digits = 4))",
    )
end

# 1) Symbol :random — i.i.d. uniform spins on S². Reproducible via rng.
report(":random", init_spins(:random, n_atoms, base_n; rng = MersenneTwister(42)))

# 2) Symbol :ferromagnetic — all spins along +ẑ (by convention).
report(":ferromagnetic", init_spins(:ferromagnetic, n_atoms, base_n))

# 3) Tuple direction — every atom aligned with (sx, sy, sz). Magnitudes
#    do not have to be unit; init_spins normalizes each column.
report("Tuple (1, 0, 0) → +x̂", init_spins((1.0, 0.0, 0.0), n_atoms, base_n))
report("Tuple (1, 1, 1) → diagonal", init_spins((1.0, 1.0, 1.0), n_atoms, base_n))

# 4) AbstractVector{<:Real} of length 3 — same as Tuple, just a
#    different Julia container.
report("Vector [0, 1, 0] → +ŷ", init_spins([0.0, 1.0, 0.0], n_atoms, base_n))

# 5) SVector{3} direction — same again, stack-allocated.
report(
    "SVector{3} (0, 0, 1) → +ẑ",
    init_spins(SVector{3,Float64}(0, 0, 1), n_atoms, base_n),
)

# 6) AbstractMatrix `(3, base_n_atoms)` — per-base-atom direction, tiled
#    across the supercell. For bcc_2x2x2 base_n = 16 (here repeat=(1,1,1)
#    so the tile is the full supercell already).
#
#    Here we just alternate ±ẑ by *base-atom index*. This is not a
#    physically meaningful antiferromagnetic order for bcc (which has
#    several inequivalent AFM types — G-type, A-type, etc. — defined by
#    the actual atomic positions in the XML); we use it only to make
#    the 3×base_n input path produce a clearly non-uniform pattern.
base_matrix = zeros(3, base_n)
for ib = 1:base_n
    base_matrix[3, ib] = iseven(ib) ? 1.0 : -1.0
end
report("Matrix 3×base_n (alternating ±ẑ)", init_spins(base_matrix, n_atoms, base_n))

# 7) AbstractMatrix `(3, n_atoms)` — supercell-shaped, used as-is after
#    column normalization. Built here from a deterministic angular sweep
#    around +ẑ so the result is reproducible without an rng.
super_matrix = zeros(3, n_atoms)
for ia = 1:n_atoms
    θ = 2π * (ia - 1) / n_atoms
    super_matrix[1, ia] = cos(θ)
    super_matrix[2, ia] = sin(θ)
end
report("Matrix 3×n_atoms (in-plane fan)", init_spins(super_matrix, n_atoms, base_n))

# 8) AbstractDict — reads `:initial_spins` from the dict. This is the
#    path used by Carlo.init! when SCEMC forwards `params` to init_spins.
report(
    "Dict :initial_spins => :ferromagnetic",
    init_spins(Dict{Symbol,Any}(:initial_spins => :ferromagnetic), n_atoms, base_n),
)
