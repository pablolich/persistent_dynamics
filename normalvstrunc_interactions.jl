# normalvstrunc_interactions.jl
# Two samplers:
#  (1) A = -(Z + μ·11ᵀ), where Z_ij ~ Normal(0, σ), diag reset to -1
#  (2) A = -G, where G_ij ~ Gamma(mean=μ, sd=σ), diag reset to -1
# Then run the same pipeline as classify_persistent_grid, for n=10.

using Random, Distributions, LinearAlgebra, DataFrames, Statistics
using CairoMakie

include("src.jl")  # expects: sample_growth_rates, is_feasible, is_persistent_glv,
                   # build_jacobian_at_equilibrium, is_stable, p1p2_by_nsigma,
                   # plot_p1p2_stacks_by_sigma, MU_SCALE,N_SCALE,REP_SCALE,SIGMA_SCALE

# ---------------------- samplers ----------------------

# (1) Normal(0,σ) + μ·11ᵀ, then flip signs; enforce diag=-1.
using Random, LinearAlgebra, Distributions

# --- Normal (with 1/n scaling) ---
# Off-diagonals: jointly normal for (A_ij, A_ji) with
#   mean = (-μ/n, -μ/n),  Var = (σ^2/n) * [1  ρ; ρ  1]
# Diagonal fixed to -1.
function sampler_normal_rank1(n::Int, seed::Int; mu::Float64, sigma::Float64, rho::Float64=0.0)
    @assert sigma > 0
    @assert mu > 0
    @assert -1.0 < rho < 1.0
    Random.seed!(seed)

    A = Matrix{Float64}(undef, n, n)
    m = fill(mu/n, 2)                       # mean before sign flip
    Σ = (sigma^2/n) * [1.0 rho; rho 1.0]
    d = MvNormal(m, Matrix(Σ))

    @inbounds for i in 1:n-1, j in i+1:n
        pair = rand(d)
        A[i, j] = -pair[1]
        A[j, i] = -pair[2]
    end
    @inbounds for i in 1:n
        A[i, i] = -1.0
    end
    return A
end

# ---- helper: Gamma(mean=μ, sd=σ) → shape k, scale θ ----
@inline function gamma_shape_scale_from_mean_sd(μ::Float64, σ::Float64)
    @assert μ > 0 "Gamma mean must be > 0"
    @assert σ > 0 "Gamma sd must be > 0"
    k = (μ/σ)^2
    θ = (σ^2)/μ
    return k, θ
end

# --- Gamma (with 1/n scaling) ---
# Target off-diagonal moments: mean = μ/n, sd = σ/√n.
# Achieved by Γ(k', θ) with k' = k/n and θ' = θ (keeps Var = (k θ^2)/n).
# Entries are then flipped in sign; diagonal fixed to -1.
function sampler_gamma(n::Int, seed::Int; mu::Float64, sigma::Float64, rho::Float64=0.0)
    @assert sigma > 0
    @assert mu > 0
    Random.seed!(seed)

    k, θ = gamma_shape_scale_from_mean_sd(mu, sigma)
    k′ = k / n
    @assert k′ > 0
    d = Gamma(k′, θ)

    A = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        A[i, j] = -rand(d)
    end
    @inbounds for i in 1:n
        A[i, i] = -1.0
    end
    return A
end

# --- helper: LogNormal(mean=m, sd=s) → (μ_LN, σ_LN) params ---
@inline function lognormal_params_from_mean_sd(m::Float64, s::Float64)
    @assert m > 0 "Lognormal mean must be > 0"
    @assert s > 0 "Lognormal sd must be > 0"
    σ2 = log(1 + (s*s)/(m*m))
    μ  = log(m) - 0.5*σ2
    return μ, sqrt(σ2)
end

# --- Lognormal (with 1/n scaling) ---
# Target off-diagonal moments: mean = μ/n, sd = σ/√n.
# Draw Y ~ LogNormal(μ_LN, σ_LN) with those moments, then set A_ij = -Y; diag = -1.
function sampler_lognormal(n::Int, seed::Int; mu::Float64, sigma::Float64, rho::Float64=0.0)
    @assert mu > 0
    @assert sigma > 0
    Random.seed!(seed)

    m = mu / n
    s = sigma / sqrt(n)
    μ_LN, σ_LN = lognormal_params_from_mean_sd(m, s)
    d = LogNormal(μ_LN, σ_LN)

    A = Matrix{Float64}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        A[i, j] = -rand(d)
    end
    @inbounds for i in 1:n
        A[i, i] = -1.0
    end
    return A
end

# ---------------------- classifier (same structure as src.classify_persistent_grid) ----------------------

function classify_persistent_grid_with_sampler(sampler::Function;
        nsim::Int=10,
        mu_vals = 1.0:1.0:1.0,
        sigma_vals = 1.0:0.1:3.0,
        n_vals = [10],
        base_seed::Int=1,
        max_tries::Int=2_000_000,
        ϵ::Float64=1e-9,
        debug_every::Int=0)

    df = DataFrame(seed=Int[], n=Int[], mu=Float64[], sigma=Float64[],
                   stable=Bool[], bflag=Int[])

    for rep in 1:nsim
        for mu in mu_vals
            for sigma in sigma_vals
                for n in n_vals
                    if n == 12 && sigma > 3.0
                        continue
                    end
                    println("rep=$rep, n=$n, μ=$(float(mu)), σ=$(float(sigma))")
                    found = false
                    n_feas_fail = 0
                    n_persist_fail = 0

                    for t in 1:max_tries
                        seed = base_seed +
                               MU_SCALE    * round(Int, float(mu)    * 100) +
                               N_SCALE     * n +
                               REP_SCALE   * rep +
                               SIGMA_SCALE * round(Int, float(sigma) * 100) +
                               t

                        A = sampler(n, seed; mu=float(mu), sigma=float(sigma), rho=0.0)
                        r = sample_growth_rates(n, seed, :ones)

                        feas, xeq = is_feasible(A, r)
                        if !feas
                            n_feas_fail += 1
                            if debug_every > 0 && (t % debug_every == 0)
                                println("    t=$t  fails → feasibility=$n_feas_fail, persistence=$n_persist_fail")
                            end
                            continue
                        end

                        ok, _, δ, bf = is_persistent_glv(A, r, xeq; ϵ=ϵ)
                        if !ok
                            n_persist_fail += 1
                            if debug_every > 0 && (t % debug_every == 0)
                                println("    t=$t  fails → feasibility=$n_feas_fail, persistence=$n_persist_fail")
                            end
                            continue
                        end

                        J  = build_jacobian_at_equilibrium(A, xeq)
                        st = is_stable(J)

                        push!(df, (seed, n, float(mu), float(sigma), st, bf))
                        println("  → found persistent system after $t tries " *
                                "(feas fails=$n_feas_fail, persist fails=$n_persist_fail)")
                        found = true
                        break
                    end

                    if !found
                        println("  → gave up after $max_tries tries " *
                                "(feas fails=$n_feas_fail, persist fails=$n_persist_fail)")
                    end
                end
            end
        end

        # # optional live update (aggregates over μ)
        # try
        #     df_probs = p1p2_by_nsigma(df)
        #     plot_p1p2_stacks_by_sigma(df_probs; size=(650, 900))
        # catch err
        #     @warn "Live plot update failed" err
        # end
    end
    return df
end

# ---------------------- plotting (two stacked panels) ----------------------

# Input: DataFrame with columns :sigma, :p1 (P(stable)), :p2 (P(nonB))
# Output: Axis with stacked bars
function plot_p1p2_single!(fig, cell, df_probs::DataFrame; 
                          title::AbstractString,
                          colors = (:steelblue, :lightskyblue, :tomato, :peachpuff))
    sub = sort(df_probs[:, [:sigma, :p1, :p2]], :sigma)
    T   = nrow(sub)
    t   = collect(sub.sigma)           # sigma values (x-ticks)
    p1  = collect(sub.p1)
    p2  = collect(sub.p2)

    # --- same height calculations as original ---
    a = p1 .* p2                    # Stable & non-B
    b = p1 .* (1 .- p2)             # Stable & B
    c = (1 .- p1) .* p2             # Unstable & non-B
    d = (1 .- p1) .* (1 .- p2)      # Unstable & B

    tbl = (
        cat    = repeat(collect(1:T), inner=4),
        height = vcat(hcat(a, b, c, d)'...),
        grp    = repeat(collect(1:4), outer=T) # 1..4 = bottom→top
    )
    colv = getindex.(Ref(colors), tbl.grp)

    ax = Axis(fig[cell...];
              xlabel = "σ",
              ylabel = "probability",
              xticks = ((1:T) .+ 2, string.(t)),
              title  = title)

    # keep the same bar semantics (offset +2, width=1, gap=0)
    barplot!(ax, tbl.cat .+ 2, tbl.height;
             stack = tbl.grp,
             color = colv,
             width = 1.0, gap = 0.0)

    labels = ["Stable & non-B", "Stable & B", "Unstable & non-B", "Unstable & B"]
    elems  = [PolyElement(color=colors[i]) for i in 1:4]
    
    return ax, elems, labels
end

# ---------------------- run ----------------------
# ---------------------- sweep runner (per-rep full sweep + save plot) ----------------------

const DIST_SCALE = 10_000_000  # to decorrelate seeds across distributions

# fixed order of distributions → columns: Normal | Gamma | Lognormal
const DISTS = [("Normal(0,σ)+μ·11ᵀ", sampler_normal_rank1),
               ("Gamma(mean=μ, sd=σ)", sampler_gamma),
               ("Lognormal(mean=μ, sd=σ)", sampler_lognormal)]

function save_three_panel_plot!(df::DataFrame; n_vals, mu_val::Float64, out_path::String)
    nrows = length(n_vals)
    ncols = length(DISTS)
    panel_w, panel_h = 600, 260
    legend_h = 70

    fig = Figure(resolution = (panel_w * ncols, panel_h * nrows + legend_h))

    for (row_idx, n) in enumerate(n_vals)
        for (col_idx, (dname, _)) in enumerate(DISTS)
            sub = df[(df.n .== n) .& (df.dist .== dname), :]
            if nrow(sub) == 0
                Axis(fig[(row_idx, col_idx)]; title = "$dname, n=$n (μ=$mu_val)")
                continue
            end
            dfp = p1p2_by_nsigma(sub)
            plot_p1p2_single!(fig, (row_idx, col_idx), dfp;
                              title = "$dname, n=$n (μ=$mu_val)")
        end
    end

    legend_colors = (:steelblue, :lightskyblue, :tomato, :peachpuff)
    legend_labels = ["Stable & non-B", "Stable & B", "Unstable & non-B", "Unstable & B"]
    legend_elems  = [PolyElement(color = c) for c in legend_colors]
    fig[nrows + 1, 1:ncols] = Legend(fig, legend_elems, legend_labels;
        orientation = :horizontal, framevisible = false, tellwidth = false)

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 8)
    save(out_path, fig)
end

function run_full_sweeps(; nsim::Int               = 700,
                          mu_vals                   = [1.0],
                          sigma_vals                = 0.1:0.1:3.5,
                          n_vals                    = [2, 4, 6, 10, 12],
                          base_seed::Int            = 1,
                          max_tries::Int            = 2_000_000,
                          ϵ::Float64                = 1e-9,
                          debug_every::Int          = 0,
                          out_dir::String           = "plots/three_dists",
                          out_prefix::String        = "normal_gamma_lognormal_p1p2_scaled")

    df = DataFrame(rep=Int[], dist=String[], seed=Int[], n=Int[],
                   mu=Float64[], sigma=Float64[], stable=Bool[], bflag=Int[])

    @assert length(mu_vals) == 1 "Current plot titles assume a single μ; got $(mu_vals)."
    μ0 = float(mu_vals[1])

    for rep in 1:nsim
        println("\n===== SWEEP rep=$rep =====")

        # loop over all distributions, then all σ, then all n (full sweep)
        for (dist_idx, (dname, sampler)) in enumerate(DISTS)
            println("  • $dname")
            for sigma in sigma_vals
                for n in n_vals
                    found = false
                    n_feas_fail = 0
                    n_persist_fail = 0

                    for t in 1:max_tries
                        seed = base_seed +
                               DIST_SCALE * dist_idx +
                               MU_SCALE    * round(Int, μ0 * 100) +
                               N_SCALE     * n +
                               REP_SCALE   * rep +
                               SIGMA_SCALE * round(Int, float(sigma) * 100) +
                               t

                        A = sampler(n, seed; mu=μ0, sigma=float(sigma), rho=0.0)
                        r = sample_growth_rates(n, seed, :ones)

                        feas, xeq = is_feasible(A, r)
                        if !feas
                            n_feas_fail += 1
                            if debug_every > 0 && (t % debug_every == 0)
                                @info "rep=$rep $dname n=$n σ=$sigma  t=$t  feas_fail=$n_feas_fail persist_fail=$n_persist_fail"
                            end
                            continue
                        end

                        ok, _, δ, bf = is_persistent_glv(A, r, xeq; ϵ=ϵ)
                        if !ok
                            n_persist_fail += 1
                            if debug_every > 0 && (t % debug_every == 0)
                                @info "rep=$rep $dname n=$n σ=$sigma  t=$t  feas_fail=$n_feas_fail persist_fail=$n_persist_fail"
                            end
                            continue
                        end

                        J  = build_jacobian_at_equilibrium(A, xeq)
                        st = is_stable(J)

                        push!(df, (rep, dname, seed, n, μ0, float(sigma), st, bf))
                        println("    n=$n σ=$(round(sigma, digits=2)) → found after $t tries " *
                                "(feas fails=$n_feas_fail, persist fails=$n_persist_fail)")
                        found = true
                        break
                    end

                    if !found
                        println("    n=$n σ=$(round(sigma, digits=2)) → gave up after $max_tries tries " *
                                "(feas fails=$n_feas_fail, persist fails=$n_persist_fail)")
                    end
                end
            end
        end

        # save plot after completing the full sweep (all dists×σ×n) for this rep
        out_path = "$(out_prefix)_sweep$(lpad(string(rep),3,'0')).png"
        try
            save_three_panel_plot!(df; n_vals=n_vals, mu_val=μ0, out_path=out_path)
            println("Saved: $out_path")
        catch err
            @warn "Plot save failed on sweep $rep" err
        end
    end
    return df
end

# ---------------------- call the sweep runner ----------------------

df_all = run_full_sweeps(; nsim=1000,
                         mu_vals=[1.0],
                         sigma_vals=1.:0.1:3.5,
                         n_vals=[2,4,6,10,12],
                         out_dir="out/three_dists",
                         out_prefix="p1p2_scaled")
