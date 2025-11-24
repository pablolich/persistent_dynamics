using CSV, DataFrames
using CairoMakie          # do not `using Makie`
using Colors              # for PolyElement/colors used in your fn defaults

include("sample_persistent_track_logs.jl")

using CairoMakie
using Colors
using DataFrames

# assumes you already have: p1p2_by_nsigma(df)

"""
    plot_p1p2_by_n_facets_sigma!(fig, row, probs; sigmas=:every_other, colors=...)

Stacked-bar panels with **x = n** and **one panel per σ**.
- `probs` must have columns: :n, :sigma, :p1, :p2 (e.g., from `p1p2_by_nsigma`).
- `sigmas = :every_other` keeps every other σ (odd indices in sorted unique list).
- Returns `(axes, σlist)`.
"""
function plot_p1p2_by_n_facets_sigma!(fig::Figure, row::Int, probs::DataFrame;
                                      sigmas = :every_other,
                                      colors = (colorant"steelblue", colorant"lightskyblue",
                                                colorant"tomato",    colorant"peachpuff"))
    # choose σ panels
    σuniq = sort(unique(Float64.(probs.sigma)))
    σlist = sigmas === :every_other ? σuniq[1:2:end] :
            sort(unique(Float64.(sigmas)))

    axes = Axis[]
    for (j, σ) in enumerate(σlist)
        sub = filter(:sigma => v -> isapprox(v, σ; atol=1e-8), probs) |> df -> sort(df, :n)
        isempty(sub) && continue

        nvals = Int.(sub.n)
        p1, p2 = sub.p1, sub.p2
        a = p1 .* p2
        b = p1 .* (1 .- p2)
        c = (1 .- p1) .* p2
        d = (1 .- p1) .* (1 .- p2)

        heights = vcat(hcat(a, b, c, d)'...)         # stack order matches legend
        cat     = repeat(1:length(nvals), inner=4)   # bar positions
        grp     = repeat(1:4, outer=length(nvals))   # stack ids 1..4
        colv    = getindex.(Ref(colors), grp)

        ax = Axis(fig[row, j],
                  xlabel = "n",
                  ylabel = (j == 1 ? "probability" : ""),
                  title  = "σ = $(round(σ, digits=2))",
                  xticks = (collect(1:length(nvals)), string.(nvals)))
        barplot!(ax, cat, heights; stack=grp, color=colv, width=1.0, gap=0.0)
        push!(axes, ax)
    end

    # explicit legend (fixed order/colors)
    legend_elems  = [PolyElement(color=colors[i]) for i in 1:4]
    legend_labels = ["Stable & non-B", "Stable & B", "Unstable & non-B", "Unstable & B"]
    Legend(fig[row + 1, 1:length(σlist)], legend_elems, legend_labels; orientation = :horizontal)

    return axes, σlist
end

# --- load & normalize columns ---
path = "cluster_sims/merged_persistent_mu_0_sigma1p0to2p8_allseeds.csv"


df = CSV.read(path, DataFrame)


# --- example usage ---
probs = p1p2_by_nsigma(df)  # df has columns :n, :sigma, :stable, :bflag
fig = Figure(size = (360*ceil(Int, length(unique(probs.sigma))/2), 360))
plot_p1p2_by_n_facets_sigma!(fig, 1, probs; sigmas = :every_other)
save("mu_0_by_n_panels_sigma.png", fig)



# ---- compute & plot ----
probs = p1p2_by_nsigma(df)
ns = sort(unique(probs.n))

fig = Figure(size = (360*length(ns), 360))
legend_elems = nothing; legend_labels = nothing
for (i, nval) in enumerate(ns)
    sub = filter(:n => ==(nval), probs)
    ax, elems, labels = plot_p1p2_single!(fig, (1, i), sub; title = "n = $nval")
    legend_elems = elems; legend_labels = labels
end
# Build legend entries explicitly (do not rely on loop side-effects)
legend_colors = (colorant"steelblue", colorant"lightskyblue",
                 colorant"tomato",    colorant"peachpuff")
legend_elems  = [PolyElement(color=c) for c in legend_colors]
legend_labels = ["Stable & non-B", "Stable & B", "Unstable & non-B", "Unstable & B"]

Legend(fig[2, 1:length(ns)], legend_elems, legend_labels; orientation = :horizontal)

save("mu_0_p1p2_by_n_sigma.png", fig)
save("mu_0_p1p2_by_n_sigma.pdf", fig)
fig