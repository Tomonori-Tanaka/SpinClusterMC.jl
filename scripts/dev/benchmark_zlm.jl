#!/usr/bin/env julia

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Random
using Magesty.MySphericalHarmonics: Zₗₘ, Zₗₘ_unsafe

function rand_unit_spin!(rng, u)
    z = 2.0 * rand(rng) - 1.0
    ϕ = 2π * rand(rng)
    r = sqrt(max(0.0, 1.0 - z^2))
    u[1] = r * cos(ϕ)
    u[2] = r * sin(ϕ)
    u[3] = z
    return u
end

function main()
    rng = MersenneTwister(42)
    n_samples = 200_000
    lmax = 6

    us = Matrix{Float64}(undef, 3, n_samples)
    u = zeros(3)
    for i in 1:n_samples
        rand_unit_spin!(rng, u)
        us[:, i] .= u
    end

    # Warmup
    warm = 0.0
    for i in 1:1000
        ui = @view us[:, i]
        warm += Zₗₘ(3, 1, ui)
        warm += Zₗₘ_unsafe(3, 1, ui)
    end
    println("warmup checksum = ", warm)

    n_calls = n_samples * sum(2l + 1 for l in 0:lmax)

    acc_safe = 0.0
    t_safe = @elapsed begin
        for i in 1:n_samples
            ui = @view us[:, i]
            for l in 0:lmax, m in -l:l
                acc_safe += Zₗₘ(l, m, ui)
            end
        end
    end

    acc_unsafe = 0.0
    t_unsafe = @elapsed begin
        for i in 1:n_samples
            ui = @view us[:, i]
            for l in 0:lmax, m in -l:l
                acc_unsafe += Zₗₘ_unsafe(l, m, ui)
            end
        end
    end

    ns_per_call_safe = 1e9 * t_safe / n_calls
    ns_per_call_unsafe = 1e9 * t_unsafe / n_calls
    speedup = t_safe / t_unsafe
    diff = abs(acc_safe - acc_unsafe)

    println("samples      = ", n_samples)
    println("lmax         = ", lmax)
    println("total calls  = ", n_calls)
    println("safe   time  = ", round(t_safe; digits = 4), " s")
    println("unsafe time  = ", round(t_unsafe; digits = 4), " s")
    println("safe   ns/call  = ", round(ns_per_call_safe; digits = 2))
    println("unsafe ns/call  = ", round(ns_per_call_unsafe; digits = 2))
    println("speedup (safe/unsafe) = ", round(speedup; digits = 3), "x")
    println("abs(checksum diff)    = ", diff)
end

main()
