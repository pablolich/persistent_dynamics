# given a set of examples that failed, load parameters, integrate,
# and reclassify dynamics using permanence tests.
ENV["GKSwstype"] = "100"   # offscreen PNG mode (no X11 window)

using Random
using Printf
using DataFrames
using DifferentialEquations
using LinearAlgebra
using SciMLBase
using CairoMakie
using Colors
using Plots
gr()

include("src/feasibility.jl")
include("src/necessary_conditions.jl")
include("src/boundary_equilibria.jl")
include("src/solve_lp.jl")
include("src/stability.jl")
include("src/check_b_matrix.jl")

# ------------------------
# GLV dynamics
# ------------------------
function glv!(dx, x, p, t)
    A, r = p
    dx .= x .* (r .+ A * x)
end

# ------------------------
# I/O helpers
# ------------------------
"""
    load_instance(path)

Read a saved model instance produced by `save_model_instance`. Returns
`(attempt, A, r)`.
"""
function load_instance(path::AbstractString)
    rows = [parse.(Float32, split(strip(line))) for line in eachline(path) if !isempty(strip(line))]
    n = length(rows)
    ncols = length(rows[1])
    @assert ncols == n + 2 "Expected first column = attempt, then n columns of A, then r"

    attempt = Int(round(rows[1][1]))
    A = zeros(Float32, n, n)
    r = zeros(Float32, n)
    for i in 1:n
        row = rows[i]
        A[i, :] .= row[2:n+1]
        r[i] = row[end]
    end
    return (attempt = attempt, A = A, r = r)
end

"""
    parse_metadata(path)

Extract metadata encoded in filenames created by `save_model_instance`.
"""
function parse_metadata(path::AbstractString)
    base = splitext(basename(path))[1]
    parts = split(base, "_")
    model = parts[1]
    seed_idx = findfirst(p -> startswith(p, "seed"), parts)
    attempt_idx = findfirst(p -> startswith(p, "attempt"), parts)
    bflag_old = parts[end-2]
    persistence_old = parts[end-1]
    stability_old = parts[end]
    seed  = seed_idx === nothing ? -1 : parse(Int, replace(parts[seed_idx], "seed" => ""))
    attempt = attempt_idx === nothing ? -1 : parse(Int, replace(parts[attempt_idx], "attempt" => ""))
    params = seed_idx === nothing ? join(parts[2:end-3], "_") : join(parts[2:seed_idx-1], "_")
    return (model = model,
            params = params,
            seed = seed,
            attempt = attempt,
            bflag_old = bflag_old,
            persistence_old = persistence_old,
            stability_old = stability_old)
end

# ------------------------
# classification helpers
# ------------------------
"""
    classify_instance(A, r; tol=1f-7, ϵ=1e-9)

Recompute feasibility, boundary persistence type, stability, and bflag.
"""
function classify_instance(A::Matrix{Float32}, r::Vector{Float32};
                           tol::Float32 = 1f-7, ϵ::Float32 = 1f-5)
    feas, xstar = is_feasible_fast(A, r, tol)
    if !feas
        return (feasible = false, persistence = "infeasible",
                stable = false, bflag = nothing, xstar = xstar, λmax = NaN)
    end

    J = build_jacobian_at_equilibrium(A, xstar)
    λmax = maximum(real.(eigen(J).values))
    nec_ok = necessary_conditions_persistence(A, J)
    if !nec_ok
        return (feasible = true, persistence = "fails_necessary",
                stable = false, bflag = nothing, xstar = xstar, λmax = λmax)
    end

    onevec = ones(eltype(A), size(A, 1))
    out = boundary_equilibria(A, onevec; tol = tol)
    persistence = determine_persistence_type(out.lhs, out.rhs; ϵ = ϵ)
    st = is_stable(J)
    bflag = out.bflag === nothing ? check_b_matrix(A) : out.bflag
    return (feasible = true, persistence = persistence, stable = st, bflag = bflag, xstar = xstar, λmax = λmax)
end

# ------------------------
# integration
# ------------------------
function integrate_instance(A::Matrix{Float32}, r::Vector{Float32};
                            N_ic::Int = 5, T::Float64 = 50.0,
                            x0_scale::Float64 = .5, seed::Int = 1,
                            x_center::Union{Nothing,AbstractVector}=nothing)
    n = length(r)
    rng = MersenneTwister(seed)
    tspan = (0.0, T)
    endpoints = Vector{Vector{Float64}}()
    sols = Vector{Any}()
    for k in 1:N_ic
        if x_center === nothing
            x0 = x0_scale .* rand(rng, n) .+ 0.1
        else
            x0 = x_center .+ x0_scale .* randn(rng, n)
            x0 .= max.(x0, 1e-6)
        end
        prob = ODEProblem(glv!, x0, tspan, (A, r))
        sol = DifferentialEquations.solve(prob; reltol = 1e-8, abstol = 1e-10, saveat = 0.1)
        if SciMLBase.successful_retcode(sol)
            push!(endpoints, Vector{Float64}(sol.u[end]))
            push!(sols, sol)
        end
    end
    return endpoints, sols
end

# ------------------------
# plotting
# ------------------------
"""
    plot_trajectories(integration_results;
                      outfile = "integration_comparison.png")

Plot trajectories for two systems (baseline and reclassified) if both are
provided. Expects a NamedTuple or Dict with keys `:system1` and `:system2`,
each storing a Vector of endpoints per IC (as returned by `integrate_instance`).
Produces side-by-side plots of x_i(t) with different colors per IC and
linetype per species, saving to `outfile`.
"""
function plot_trajectories(results;
                           outfile::AbstractString = "integration_comparison.png")
    haskey(results, :system1) || haskey(results, :system2) || haskey(results, :system) || error("Need :system1, :system2, or :system trajectories")

    data = Dict{Symbol, Vector{Tuple{Vector{Float64}, Matrix{Float64}}}}()
    n = nothing
    for key in (:system1, :system2, :system)
        if haskey(results, key)
            traces = results[key]
            isempty(traces) && continue
            data[key] = traces
            n = n === nothing ? size(traces[1][2], 1) : n
        end
    end
    n === nothing && error("No trajectories found")

    species_colors = distinguishable_colors(n)
    nplots = length(data)
    fig = Figure(resolution = (400 * nplots, 400))

    for (col, key) in enumerate(keys(data))
        traces = data[key]
        isempty(traces) && continue
        ax = Axis(fig[1, col], title = "$(String(key)) trajectories", xlabel = "t", ylabel = "x(t)")

        # only first IC
        t, X = traces[1]
        for i in 1:n
            lines!(ax, t, X[i, :]; color = species_colors[i], linewidth = 1.5, label = "x_$i")
        end
    end

    save(outfile, fig)
    return fig
end

# ------------------------
# driver
# ------------------------
function analyze_failures(paths::Vector{String};
                          N_ic::Int = 5,
                          T::Float64 = 500.0,
                          tol::Float32 = 1f-7,
                          ϵ::Float32 = 1f-9,
                          plot::Bool = true,
                          plot_dir::AbstractString = "integration_plots")
    df = DataFrame(path = String[], model = String[], params = String[],
                   seed = Int[], attempt = Int[],
                   old_bflag = String[], old_persistence = String[], old_stability = String[],
                   feasible = Bool[], persistence = String[], stable = Bool[], bflag = Any[], λmax = Float64[],
                   endpoints = Vector{Vector{Vector{Float64}}}(), plot_path = Union{String,Missing}[])

    plot && !isdir(plot_dir) && mkpath(plot_dir)

    for path in paths
        meta = parse_metadata(path)
        inst = load_instance(path)
        class = classify_instance(inst.A, inst.r; tol = tol, ϵ = ϵ)
        ends, sol_traces = integrate_instance(inst.A, inst.r;
                                      N_ic = N_ic, T = T,
                                    seed = meta.seed,
                                    x_center = nothing)#class.xstar)

        plot_path = missing
        if plot && !isempty(sol_traces)
            traces = [(collect(sol.t), hcat(sol.u...)) for sol in sol_traces]
            prefix = replace(splitext(basename(path))[1], "." => "_")
            outfile = joinpath(plot_dir, "$(prefix)_trajectories.png")
            plot_trajectories(Dict(:system1 => traces); outfile = outfile)
            plot_path = outfile
        end

        push!(df, (path, meta.model, meta.params, meta.seed, meta.attempt,
                   meta.bflag_old, meta.persistence_old, meta.stability_old,
                   class.feasible, class.persistence, class.stable, class.bflag, class.λmax,
                   ends, plot_path))
    end
    return df
end

function find_failure_files(root::AbstractString = "results")
    files = String[]
    for (dir, _, fs) in walkdir(root)
        for f in fs
            push!(files, joinpath(dir, f))
        end
    end
    return sort(filter(f -> endswith(f, ".txt"), files))
end
paths = ["results/normal_f32/mu=0.0_sigma=3.5_rho=0.0/normal_f32_mu=0.0_sigma=3.5_rho=0.0_seed1_attempt88803_unphysical_strong_stable.txt",
         "results/normal_f32/mu=0.0_sigma=3.5_rho=0.0/normal_f32_mu=0.0_sigma=3.5_rho=0.0_seed1_attempt73301_unphysical_strong_stable.txt"]



df = analyze_failures(paths; N_ic = 5, T = 100.0)
println(df)
