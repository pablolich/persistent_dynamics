using DataFrames
using CairoMakie
using LinearSolve
using LinearSolveAutotune

function luflop(m, n = m; innerflop = 2)
    sum(1:min(m, n)) do k
        invflop = 1
        scaleflop = isempty((k + 1):m) ? 0 : sum((k + 1):m)
        updateflop = isempty((k + 1):n) ? 0 :
                     sum((k + 1):n) do j
            isempty((k + 1):m) ? 0 : sum((k + 1):m) do i
                innerflop
            end
        end
        invflop + scaleflop + updateflop
    end
end

function plot_lu_efficiency_faceted(df::DataFrame)
    eltypes = sort(unique(df.eltype))

    # only successful runs
    df_success = filter(row -> row.success, df)

    # all algorithms that appear anywhere (for consistent colors)
    algorithms_all = sort(unique(df_success.algorithm))
    n_el  = length(eltypes)
    n_alg = length(algorithms_all)

    # global y-limits
    _, y_max = extrema(df_success.gflops)

    # panel geometry: wider than tall (height/width ≈ 3/4, tweak factor as you like)
    panel_width  = 360
    panel_height = Int(round(0.75 * panel_width))   # use 0.8 if you prefer

    fig = Figure(resolution = (panel_width * n_el + 140, panel_height + 140))

    # colors per algorithm (consistent across all panels)
    color_vec = Makie.categorical_colors(:tab10, n_alg)
    colors = Dict(algorithms_all[i] => color_vec[i] for i in eachindex(algorithms_all))

    handles = Dict{String, AbstractPlot}()

    for (i, el) in enumerate(eltypes)
        df_use = filter(row -> row.success && row.eltype == el, df)
        # algorithms actually present for this eltype
        algorithms_here = sort(unique(df_use.algorithm))

        ax = Axis(fig[1, i];
                  width  = panel_width,
                  height = panel_height,
                  xlabel = "matrix size n",
                  ylabel = (i == 1 ? "GFLOP/s" : ""),
                  title  = string(el))

        ylims!(ax, 0, y_max * 1.1)

        for alg in algorithms_here
            sub = filter(row -> row.algorithm == alg, df_use)
            isempty(sub) && continue
            sort!(sub, :size)

            plt = lines!(ax, sub.size, sub.gflops; color = colors[alg])
            scatter!(ax, sub.size, sub.gflops; color = colors[alg])

            # record one handle per algorithm (first time we see it)
            if !haskey(handles, alg)
                handles[alg] = plt
            end
        end

        # visually shared y-axis: hide y on all but first panel
        if i > 1
            ax.ylabelvisible      = false
            ax.yticklabelsvisible = false
            ax.yticksvisible      = false
        end
    end

    # legend: only algorithms that actually appeared
    alg_order = [alg for alg in algorithms_all if haskey(handles, alg)]

    Legend(fig[2, 1:n_el],
           [handles[alg] for alg in alg_order],
           alg_order;
           orientation = :horizontal,
           nbanks      = 2)

    rowgap!(fig.layout, 10)

    fig
end

using DataFrames
using BenchmarkTools
using LinearAlgebra
using Random


"""
    append_solve_results!(df; rng=Random.GLOBAL_RNG, nruns=10)

For each (size, eltype) combination present in `df`,
benchmark the dense solve `A \\ b` and append rows with
operation = "solve_Ab".
"""
function append_solve_results!(df::DataFrame;
                               rng = Random.GLOBAL_RNG,
                               nruns::Int = 10)

    # only successful rows define the grid
    df_success = filter(row -> row.success, df)

    eltypes = sort(unique(df_success.eltype))
    sizes   = sort(unique(df_success.size))

    new_rows = NamedTuple[]

    for el in eltypes
        for n in sizes
            # skip combinations that never existed
            any(r -> r.eltype == el && r.size == n, eachrow(df_success)) || continue

            # --- build A, b of the requested eltype ---
            # assumes eltype stored as "Float32", "Float64", etc.
            T = eval(Meta.parse(el))
            A = rand(rng, T, n, n)
            b = rand(rng, T, n)

            # total time for nruns solves
            t_total = @elapsed begin
                for _ in 1:nruns
                    A \ b
                end
            end
            t_avg = t_total / nruns  # seconds per solve

            # FLOP estimate using same formula as benchmark_algorithms
            flops  = luflop(n, n)               # your helper
            gflops = flops / (t_avg * 1e9)      # GFLOP/s

            push!(new_rows, (
                size      = n,
                algorithm = "solve_Ab",
                eltype    = el,
                gflops    = gflops,
                success   = true,
                error     = "",
            ))
        end
    end

    append!(df, DataFrame(new_rows))
    return df
end

results = autotune_setup(
           samples = 1000,
           seconds = 100.0,
           sizes = [:tiny],
           eltypes = (Float16 , Float32, Float64)
       )

# usage
full_results = append_solve_results!(results.results_df)

# example
fig = plot_lu_efficiency_faceted(full_results)
save("lu_efficiency_all.png", fig)


