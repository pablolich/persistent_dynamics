include("sample_persistent_track_logs.jl")
include("efficient_persistence_classification.jl")

using DataFrames, CairoMakie, Statistics, Makie

# --- tiny timing helper ---
const _NS2S = 1e-9
timed(f) = (local t=time_ns(); local v=f(); (v, (time_ns()-t)*_NS2S))

mutable struct StepTimers
    t_sample::Float64
    t_feasible::Float64
    t_jacobian::Float64
    t_necessary::Float64
    t_boundary::Float64
    t_buildC::Float64
    t_LP::Float64
    t_stable::Float64
end
StepTimers() = StepTimers(0,0,0,0,0,0,0,0)

"""
    run_grid_times(n_vals, sigma_vals; K, mu=-1.0, rho=0.0, ϵ=1e-9, tol=1e-7,
                   trials_per_success=2_000_000, progress=true)

Run the same search as `run_grid_simple`, but **do not** plot.  
Returns a pair `(df_min, df_times)`:

- `df_min`: minimal rows for your p1/p2 bars (columns: `:n, :sigma, :stable, :bflag`).
- `df_times`: one row per `(n, σ)` cell with seconds spent in each step, plus totals:
  `:n, :sigma, :attempts, :found, :t_sample, :t_feasible, :t_jacobian, :t_necessary,
   :t_boundary, :t_buildC, :t_LP, :t_stable, :t_total`.
"""
function run_grid_times(n_vals, sigma_vals;
                        K::Int,
                        mu::Float64 = -1.0,
                        rho::Float64 = 0.0,
                        ϵ::Float64 = 1e-9,
                        tol::Float64 = 1e-7,
                        trials_per_success::Int = 2_000_000,
                        progress::Bool = true)

    df_min   = DataFrame(n=Int[], sigma=Float64[], stable=Bool[])
    df_times = DataFrame(n=Int[], sigma=Float64[], attempts=Int[], found=Int[],
                         t_sample=Float64[], t_feasible=Float64[], t_jacobian=Float64[],
                         t_necessary=Float64[], t_boundary=Float64[], t_buildC=Float64[],
                         t_LP=Float64[], t_stable=Float64[], t_total=Float64[])

    for n in n_vals
        onevec = ones(Float64, n)
        for σ in sigma_vals
            found    = 0
            attempts = 0
            attempts_limit = K * trials_per_success
            T = StepTimers()

            while found < K && attempts < attempts_limit
                attempts += 1
                seed = 1_000_000*n + 10_000*round(Int, 100*Float64(σ)) + attempts

                # 1) sample
                (A, dt) = timed(() -> sampler_normal_allatonce(n, seed; mu=mu, sigma=σ, rho=rho))
                T.t_sample += dt

                # 2) feasible
                ((feas, xstar), dt) = timed(() -> is_feasible(A, onevec))
                T.t_feasible += dt
                feas || continue

                # 3) jacobian
                (J, dt) = timed(() -> build_jacobian_at_equilibrium(A, xstar))
                T.t_jacobian += dt

                # 4) necessary conditions
                (nec_ok, dt) = timed(() -> necessary_conditions_persistence(A, J))
                T.t_necessary += dt
                nec_ok || continue

                # 5) boundary enumeration (+ bflag, early-exit on necessary fail inside)
                (out, dt) = timed(() -> boundary_equilibria_b_flag_efficient(A, onevec; tol=tol))
                T.t_boundary += dt
                isempty(out.lhs) && continue

                # 6) build C
                (C, dt) = timed(() -> build_C(out.lhs, out.rhs))
                T.t_buildC += dt

                # 7) LP
                (res, dt) = timed(() -> find_p_maxmargin(C; ϵ=ϵ))
                T.t_LP += dt
                (res === nothing || res.δ <= 0) && continue

                # 8) stability
                (st, dt) = timed(() -> is_stable(J))
                T.t_stable += dt

                push!(df_min, (n=n, sigma=Float64(σ), stable=st))
                found += 1
            end

            t_total = T.t_sample + T.t_feasible + T.t_jacobian + T.t_necessary +
                      T.t_boundary + T.t_buildC + T.t_LP + T.t_stable

            push!(df_times, (n=n, sigma=Float64(σ), attempts=attempts, found=found,
                             t_sample=T.t_sample, t_feasible=T.t_feasible, t_jacobian=T.t_jacobian,
                             t_necessary=T.t_necessary, t_boundary=T.t_boundary, t_buildC=T.t_buildC,
                             t_LP=T.t_LP, t_stable=T.t_stable, t_total=t_total))

            progress && println((found == K ?
                "✓" : "×"), " n=$n, σ=$(Float64(σ))  found=$found / K=$K  attempts=$attempts / $attempts_limit  total_time=$(round(t_total,digits=3))s")
        end
    end

    return df_min, df_times
end

# Build 2×2 panel of heatmaps for the four timing categories in one figure.
function heatmaps_2x2_times(df_times::DataFrame;
                            steps = (:t_sample, :t_feasible, :t_boundary, :t_LP),
                            use_log::Bool = true,
                            cmap = :plasma,
                            size = (1200, 800),
                            out_path::AbstractString = "timing_plots/heatmaps_2x2.png")

    mkpath(dirname(out_path))
    ns   = sort(unique(df_times.n))
    sigs = sort(unique(df_times.sigma))

    # assemble Z matrices per step
    Zs = Dict{Symbol, Matrix{Float64}}()
    for step in steps
        Z = Matrix{Float64}(undef, length(ns), length(sigs))
        for (i, n) in enumerate(ns), (j, σ) in enumerate(sigs)
            rows = df_times[(df_times.n .== n) .& (df_times.sigma .== σ), :]
            Z[i, j] = nrow(rows) == 0 ? NaN : Float64(first(rows[!, step]))
        end
        if use_log
            @inbounds for k in eachindex(Z)
                Z[k] = (isfinite(Z[k]) && Z[k] > 0) ? log10(Z[k]) : NaN
            end
        end
        Zs[step] = Z
    end

    # shared color range across panels (ignore NaNs)
    allvals = reduce(vcat, [vec(Zs[s]) for s in steps])
    finite_vals = filter(isfinite, allvals)
    vmin = isempty(finite_vals) ? 0.0 : minimum(finite_vals)
    vmax = isempty(finite_vals) ? 1.0 : maximum(finite_vals)

    fig = Figure(size = size)
    axes = Tuple{Axis,Any}[]

    for (idx, step) in enumerate(steps)
        row = ((idx - 1) ÷ 2) + 1
        col = ((idx - 1) % 2) + 1
        ax = Axis(fig[row, col],
                  xlabel = "σ",
                  ylabel = "n",
                  title  = string(step, use_log ? " (log10 s)" : " (s)"))
        hm = heatmap!(ax, sigs, ns, Zs[step];
                      colormap = cmap, colorrange = (vmin, vmax))
        push!(axes, (ax, hm))
    end

    # one shared colorbar
    Colorbar(fig[:, 3], axes[end][2],
             label = use_log ? "log10(seconds)" : "seconds")

    save(out_path, fig)
    return fig
end

# map a Symbol (e.g., :t_sample) to the actual column key in df (String or Symbol)
@inline function _key(df::DataFrame, s::Symbol)
    ns = names(df)
    s ∈ ns         && return s
    string(s) ∈ ns && return string(s)
    return nothing
end

"""
Stacked, normalized time shares vs σ, one panel per n.
Top `topk` time categories are colored; the rest are merged into "other" (grey).
Works whether df_times column names are Strings or Symbols.

Expected timing columns (any subset present is OK):
:t_sample, :t_feasible, :t_jacobian, :t_necessary, :t_boundary, :t_buildC, :t_LP, :t_stable
"""
@inline _key(df::DataFrame, s::Symbol) = (s ∈ names(df)) ? s :
                                         (string(s) ∈ names(df) ? string(s) : nothing)

function stacked_normalized_times_by_sigma(df_times::DataFrame;
        topk::Int = 4,
        size = (1200, 900),
        colors = (:steelblue, :tomato, :goldenrod, :seagreen),
        other_color = :gray80,
        out_path::AbstractString = "timing_plots/stacked_times_2x2.png")

    mkpath(dirname(out_path))

    expected = Symbol[:t_sample, :t_feasible, :t_jacobian, :t_necessary,
                      :t_boundary, :t_buildC, :t_LP, :t_stable]
    keymap = Dict{Symbol,Union{Symbol,String}}()
    for s in expected
        k = _key(df_times, s); isnothing(k) || (keymap[s] = k)
    end
    steps_all = collect(keys(keymap))
    @assert !isempty(steps_all) "No timing columns found."

    totals = Dict(s => sum(skipmissing(df_times[!, keymap[s]])) for s in steps_all)
    steps_sorted = sort(steps_all; by = s -> -totals[s])
    topcats   = steps_sorted[1:min(topk, length(steps_sorted))]
    othercats = setdiff(steps_all, topcats)

    ns   = sort(unique(df_times.n))
    sigs = sort(unique(df_times.sigma))

    fig = Figure(size=size)
    labels = vcat(string.(topcats), "other")
    elems  = vcat([PolyElement(color=colors[i]) for i in 1:length(topcats)],
                  PolyElement(color=other_color))

    nrows = min(4, length(ns))
    ncols = cld(length(ns), nrows)

    for (k, n) in enumerate(ns)
        row = fld(k-1, ncols) + 1
        col = mod(k-1, ncols) + 1
        sub = df_times[df_times.n .== n, :]

        # Build normalized shares per σ
        S = length(sigs)
        Kc = length(topcats) + 1       # +1 for “other”
        shares = Matrix{Float64}(undef, S, Kc)
        for (j, σ) in enumerate(sigs)
            rows = sub[sub.sigma .== σ, :]
            if nrow(rows) == 0
                shares[j, :] .= NaN
                continue
            end
            r = rows[1, :]
            T = sum(Float64(r[keymap[s]]) for s in steps_all)
            if !(isfinite(T) && T > 0)
                shares[j, :] .= NaN
                continue
            end
            for (i, s) in enumerate(topcats)
                shares[j, i] = Float64(r[keymap[s]]) / T
            end
            shares[j, end] = sum(Float64(r[keymap[s]]) for s in othercats) / T
        end

        # ---- single barplot! call with stacking ----
        x = collect(sigs)
        X = Vector{Float64}(); H = Vector{Float64}(); G = Vector{Int}()
        reserve = length(x) * Kc
        sizehint!(X, reserve); sizehint!(H, reserve); sizehint!(G, reserve)
        for i in 1:Kc
            append!(X, x)
            append!(H, view(shares, :, i))
            append!(G, fill(i, length(x)))
        end

        ax = Axis(fig[row, col]; title="n = $n", xlabel="σ", ylabel="normalized time")
        # width = σ spacing; no gap; no stroke
        Δσ = (length(x) > 1) ? minimum(diff(x)) : 0.2
        colvec = vcat(Tuple(colors)[1:length(topcats)]..., other_color)

        barplot!(ax, X, H;
                stack = G,
                color = repeat(colvec, inner=length(x)),
                width = Δσ,       # makes adjacent bars touch
                gap = 0.0,
                strokewidth = 0)
        colvec = vcat(Tuple(colors)[1:length(topcats)]..., other_color)
        ylims!(ax, 0, 1)
    end

    fig[nrows+1, 1:ncols] = Legend(fig, elems, labels; orientation=:horizontal, framevisible=false)
    save(out_path, fig)
    return fig
end

# Map Symbol -> actual df column (String or Symbol)
@inline _key(df::DataFrame, s::Symbol) = (s ∈ names(df)) ? s :
                                         (string(s) ∈ names(df) ? string(s) : nothing)

"""
lines_abs_times_by_sigma(df_times; topk=4, y_min=1e-6, size=(1200,900),
                         colors=(...), other_color=:gray80,
                         out_path="timing_plots/lines_abs_times_2x2.png")

Per-n panels. X = σ. Y = absolute time (seconds, log scale).
Plots one line+dots per category for the globally top-`topk` timing columns; all
remaining timing columns are summed into "other" (grey). Colors match earlier plots.
"""
function lines_abs_times_by_sigma(df_times::DataFrame;
        topk::Int = 4,
        y_min::Float64 = 1e-4,
        size = (1200, 900),
        colors = (:steelblue, :tomato, :goldenrod, :seagreen),
        other_color = :gray80,
        out_path::AbstractString = "timing_plots/lines_abs_times_2x2.png")

    mkpath(dirname(out_path))

    # discover timing columns (accept String/Symbol names)
    expected = Symbol[:t_sample, :t_feasible, :t_jacobian, :t_necessary,
                      :t_boundary, :t_buildC, :t_LP, :t_stable]
    keymap = Dict{Symbol,Union{Symbol,String}}()
    for s in expected
        k = _key(df_times, s); isnothing(k) || (keymap[s] = k)
    end
    steps_all = collect(keys(keymap))
    @assert !isempty(steps_all) "No timing columns found."

    # pick global top-k by total time
    totals = Dict(s => sum(skipmissing(df_times[!, keymap[s]])) for s in steps_all)
    steps_sorted = sort(steps_all; by = s -> -totals[s])
    topcats   = steps_sorted[1:min(topk, length(steps_sorted))]
    othercats = setdiff(steps_all, topcats)

    ns   = sort(unique(df_times.n))
    sigs = sort(unique(df_times.sigma))

    fig = Figure(size=size)
    nrows = min(4, length(ns))
    ncols = cld(length(ns), nrows)

    # legend (shared)
    labels = vcat(string.(topcats), "other")
    colvec = vcat(Tuple(colors)[1:length(topcats)]..., other_color)
    leg_elems = [LineElement(color=c) for c in colvec]

    for (k, n) in enumerate(ns)
        row = fld(k-1, ncols) + 1
        col = mod(k-1, ncols) + 1
        sub = df_times[df_times.n .== n, :]

        ax = Axis(fig[row, col];
                  title="n = $n", xlabel="σ", ylabel="seconds",
                  yscale=log10)
        x = collect(sigs)

        # series for top categories
        for (i, s) in enumerate(topcats)
            y = similar(x, Float64)
            for (j, σ) in enumerate(sigs)
                rows = sub[sub.sigma .== σ, :]
                if nrow(rows) == 0
                    y[j] = NaN
                else
                    v = Float64(rows[1, keymap[s]])
                    y[j] = (v > 0) ? v : NaN
                end
            end
            lines!(ax, x, y, color=colvec[i])
            scatter!(ax, x, y, color=colvec[i])
        end

        # series for "other"
        y_other = similar(x, Float64)
        for (j, σ) in enumerate(sigs)
            rows = sub[sub.sigma .== σ, :]
            if nrow(rows) == 0
                y_other[j] = NaN
            else
                r = rows[1, :]
                s = sum(Float64(r[keymap[sc]]) for sc in othercats; init=0.0)
                y_other[j] = (s > 0) ? s : NaN
            end
        end
        lines!(ax, x, y_other, color=other_color)
        scatter!(ax, x, y_other, color=other_color)

        ylims!(ax, y_min, nothing)
    end

    fig[nrows+1, 1:ncols] = Legend(fig, leg_elems, labels;
                                    orientation=:horizontal, framevisible=false)
    save(out_path, fig)
    return fig
end

# ---- RUN + PLOT ----
# ranges requested: n = 4:1:11, σ = 1:0.2:3
n_vals     = [13]
sigma_vals = 1.0:0.2:2.8

# # choose K and effort per success (keeps effort constant as K grows)
K  = 1
TPS = 1_030_000_000

#run timed sweep (uses your instrumented run_grid_times; no bar plots produced)
df_min, df_times = run_grid_times(n_vals, sigma_vals;
    K=K, mu=0.0, rho=0.0, ϵ=1e-9, tol=1e-7,
    trials_per_success=TPS, progress=true)

# --- example usage after you produced `df_times` with run_grid_times ---
#fig = heatmaps_2x2_times(df_times; out_path="timing_plots/heatmaps_all_at_once.png")

# usage:
fig = stacked_normalized_times_by_sigma(df_times; topk=4, out_path="timing_plots/stacked.png")

# usage (after computing df_times with `run_grid_times`):
fig = lines_abs_times_by_sigma(df_times; topk=4, out_path="timing_plots/lines_abs_time.png")
