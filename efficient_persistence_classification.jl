include("sample_persistent_track_logs.jl")

function boundary_equilibria_b_flag_efficient(A::AbstractMatrix, r::AbstractVector; tol::Float64=1e-7)

    n = size(A,1); @assert size(A,2) == n; @assert length(r) == n
    A64 = Matrix{Float64}(A)
    r64 = Vector{Float64}(r)

    M  = zeros(Float64, n+1, n+1)
    @views M[1:n,1:n] .= A64
    @views M[1:n,n+1] .= r64
    M2 = M .+ ones(n+1, n+1)

    lhs = Vector{Vector{Float64}}()
    rhs = Vector{Float64}()

    nec_fail  = false
    found_nonB = false

    nM = n + 1
    for k in 1:nM-1
        onesk = ones(Float64, k)
        for S in combinations(1:nM, k)
            nec_fail && break

            Mk = @view M2[S,S]

            colsum = vec(sum(Mk; dims=1))
            maximum(colsum) <= 0.0 && continue
            badrow = false
            @inbounds for i in 1:k
                if maximum(@view Mk[i,:]) <= 0.0; badrow = true; break; end
            end
            badrow && continue

            F = lu(Mk)
            any(abs.(diag(F.U)) .< tol) && continue
            xk = F \ onesk
            make_positive_if_same_sign!(xk, tol) || continue

            p = zeros(Float64, nM); p[S] = xk
            s = sum(p); s <= tol && continue
            p ./= s

            g = M * p
            φ = sum(p .* g)
            y = @view g[1:n]

            # necessary condition: max(Ax_off) > φ
            inS = falses(n)
            @inbounds for i in S
                i <= n && (inS[i] = true)
            end
            if !all(inS)                     # there is at least one off-support species
                off = findall(!, inS)        # positions, not a BitVector mask
                offmax = maximum(@view y[off])
                if !(offmax > φ + tol)
                    nec_fail = true
                    break
                end
            end

            push!(lhs, g)
            push!(rhs, φ)

            # B-flag reuse on pure-A faces S ⊆ {1..n}
            if !found_nonB && all(i -> i <= n, S)
                AS = @view A64[S,S]
                v  = AS * xk
                α  = mean(v)
                if abs(α) > tol && maximum(abs.(v .- α)) <= 10*tol
                    xA = xk ./ α
                    all(xA .> tol) && (found_nonB = true)
                end
            end
        end
        nec_fail && break
    end

    return (lhs = nec_fail ? Vector{Vector{Float64}}() : lhs,
            rhs = nec_fail ? Vector{Float64}()          : rhs,
            bflag = (found_nonB ? 1 : -1))
end

# ---------------------------------------------------------------------------
# Minimal sampler for p1/p2 bars: returns only what p1p2_by_nsigma() needs.
# Columns: :n, :sigma, :stable (Bool), :bflag (Int; 1=non-B, -1=B)
# ---------------------------------------------------------------------------
# Simple, legible loop:
# - Walk (n, σ) cell-by-cell.
# - For each cell, attempt until you either collect K successes or hit max_attempts.
# - Print status when a cell is completed or abandoned.
# - Optionally save interim barplots every `save_every` total successes into `out_dir`.

# Same logic as before, but attempts budget scales with K:
# per-success budget = trials_per_success (default 2e6) → per-cell limit = K * trials_per_success.
# Same logic as before, but attempts budget scales with K:
# per-success budget = trials_per_success (default 2e6) → per-cell limit = K * trials_per_success.

function run_grid_simple(n_vals, sigma_vals;
                         K::Int,
                         mu::Float64 = -1.0,
                         rho::Float64 = 0.0,
                         ϵ::Float64 = 1e-9,
                         tol::Float64 = 1e-7,
                         trials_per_success::Int = 2_000_000,   # <-- new
                         save_every::Union{Int,Nothing} = 5,
                         out_dir::AbstractString = "plots_p1p2_simple")

    mkpath(out_dir)
    df = DataFrame(n=Int[], sigma=Float64[], stable=Bool[], bflag=Int[])

    total_found = 0
    for n in n_vals
        onevec = ones(Float64, n)
        for σ in sigma_vals
            found    = 0
            attempts = 0
            attempts_limit = K * trials_per_success

            while found < K && attempts < attempts_limit
                attempts += 1
                seed = 1_000_000*n + 10_000*round(Int, 100*Float64(σ)) + attempts

                A = sampler_normal_rank1(n, seed; mu=abs(mu), sigma=σ, rho=rho)
                feas, xstar = is_feasible(A, onevec); feas || continue

                J = build_jacobian_at_equilibrium(A, xstar)
                necessary_conditions_persistence(A, J) || continue

                out = boundary_equilibria_b_flag_efficient(A, onevec; tol=tol)
                isempty(out.lhs) && continue

                C   = build_C(out.lhs, out.rhs)
                res = find_p_maxmargin(C; ϵ=ϵ)
                (res === nothing || res.δ <= 0) && continue

                st = is_stable(J)
                push!(df, (n=n, sigma=Float64(σ), stable=st, bflag=out.bflag))
                found += 1
                total_found += 1

                if !isnothing(save_every) && total_found % save_every == 0
                    df_probs = p1p2_by_nsigma(df)
                    fig = plot_p1p2_stacks_by_sigma(df_probs; size=(1200, 900))
                    CairoMakie.save(joinpath(out_dir, "p1p2_stacks_step_$(lpad(total_found,6,'0')).png"), fig)
                end
            end

            if found == K
                println("✓ n=$n, σ=$(Float64(σ)) → K=$K collected (attempts=$attempts / $attempts_limit)")
            else
                println("× n=$n, σ=$(Float64(σ)) → gave up at attempts=$attempts / $attempts_limit (found=$found < K=$K)")
            end
        end
    end

    df_probs = p1p2_by_nsigma(df)
    fig = plot_p1p2_stacks_by_sigma(df_probs; size=(1200, 900))
    CairoMakie.save(joinpath(out_dir, "p1p2_stacks_FINAL.png"), fig)
    return df
end

# Assumes df has columns: :n, :sigma, :stable (Bool), :bflag (Int; 1 = non-B)
function p1p2_by_nsigma(df::DataFrame)
    out = DataFrame(n=Int[], sigma=Float64[], p1=Float64[], p2=Float64[])
    for sub in groupby(df, [:n, :sigma])
        stab = collect(skipmissing(sub.stable))
        bflg = collect(skipmissing(sub.bflag))
        isempty(stab) && continue
        push!(out, (
            first(sub.n),
            first(sub.sigma),
            mean(Bool.(stab)),       # P(stable)
            mean(bflg .== 1),        # P(nonB) (bflag == 1)
        ))
    end
    sort!(out, [:n, :sigma])
    return out
end

# Input: DataFrame with columns :n, :sigma, :p1 (P(stable)), :p2 (P(nonB))
# Output: Figure with panels by n; x-axis = sigma; stacked bars with same heights as before.
function plot_p1p2_stacks_by_sigma(df::DataFrame;
                                   size=(1200, 800),
                                   colors = (:steelblue, :lightskyblue, :tomato, :peachpuff))
    ns = sort(unique(df.n))
    nrows = min(5, length(ns))
    ncols = cld(length(ns), nrows)

    fig = Figure(size=size)

    # legend entries (match p1=stable, p2=nonB)
    labels = ["Stable & non-B", "Stable & B", "Unstable & non-B", "Unstable & B"]
    elems  = [PolyElement(color=colors[i]) for i in 1:4]

    for (k, n) in enumerate(ns)
        row = fld(k-1, ncols) + 1
        col = mod(k-1, ncols) + 1

        sub = sort(df[df.n .== n, [:sigma, :p1, :p2]], :sigma)
        T   = nrow(sub)
        t   = collect(sub.sigma)           # sigma values (x-ticks)
        p1  = collect(sub.p1)
        p2  = collect(sub.p2)

        # --- same height calculations as original ---
        a = p1 .* p2
        b = p1 .* (1 .- p2)
        c = (1 .- p1) .* p2
        d = (1 .- p1) .* (1 .- p2)

        tbl = (
            cat    = repeat(collect(1:T), inner=4),
            height = vcat(hcat(a, b, c, d)'...),
            grp    = repeat(collect(1:4), outer=T) # 1..4 = bottom→top
        )
        colv = getindex.(Ref(colors), tbl.grp)

        ax = Axis(fig[row, col];
                  xlabel = "σ",
                  ylabel = "probability",
                  xticks = ((1:T) .+ 2, string.(t)),
                  title  = "n = $n")

        # keep the same bar semantics (offset +2, width=1, gap=0)
        barplot!(ax, tbl.cat .+ 2, tbl.height;
                 stack = tbl.grp,
                 color = colv,
                 width = 1.0, gap = 0.0)
    end

    fig[nrows+1, 1:ncols] = Legend(fig, elems, labels; orientation=:horizontal, framevisible=false)
    return fig
end

# --- call: collect minimal data for p1/p2 bars and plot ---

using DataFrames, CairoMakie

# Example:
# n_vals     = [13]#4:2:12
# sigma_vals = 1.0:0.2:3.0
# K = 100
# df = run_grid_simple(n_vals, sigma_vals; K=K, trials_per_success=2_000_000,
#                      mu=-1.0, rho=0.0, ϵ=1e-9, tol=1e-7,
#                      save_every=K, out_dir="plots_p1p2_mu_m1_simple")
