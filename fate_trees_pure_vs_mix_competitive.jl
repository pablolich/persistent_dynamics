# combined_persistence_fate_trees.jl  (updated)
# n = 8, mu = -1, sigma = 1.0:0.2:2.8, K = 100 LP successes per σ
# Distributions: Normal / Gamma / Lognormal (all 1/n-scaled, matched mean/sd)

using Random, LinearAlgebra, Statistics
using Distributions
using DataFrames
using Printf
using Combinatorics

# ---- project utilities you already have ----
include("sample_persistent_track_logs.jl")

# ========= 1/n-scaled samplers (matched statistics) =========

function sampler_normal_rank1(n::Int, seed::Int; mu::Float64, sigma::Float64, rho::Float64=0.0)
    @assert sigma > 0
    @assert mu > 0
    @assert -1.0 < rho < 1.0
    Random.seed!(seed)

    A = Matrix{Float64}(undef, n, n)
    m = fill(mu/n, 2)
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

@inline function gamma_shape_scale_from_mean_sd(μ::Float64, σ::Float64)
    @assert μ > 0; @assert σ > 0
    k = (μ/σ)^2
    θ = (σ^2)/μ
    return k, θ
end

function sampler_gamma(n::Int, seed::Int; mu::Float64, sigma::Float64, rho::Float64=0.0)
    @assert mu > 0; @assert sigma > 0
    Random.seed!(seed)
    k, θ = gamma_shape_scale_from_mean_sd(mu, sigma)
    k′ = k / n
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

@inline function lognormal_params_from_mean_sd(m::Float64, s::Float64)
    @assert m > 0; @assert s > 0
    σ2 = log(1 + (s*s)/(m*m))
    μ  = log(m) - 0.5*σ2
    return μ, sqrt(σ2)
end

function sampler_lognormal(n::Int, seed::Int; mu::Float64, sigma::Float64, rho::Float64=0.0)
    @assert mu > 0; @assert sigma > 0
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

# ========= B-matrix flag (physical/unphysical) =========
# Return -1 ⇒ B-matrix (physical); +1 ⇒ non-B (unphysical).
if !isdefined(@__MODULE__, :b_matrix_flag)
    function b_matrix_flag(A::AbstractMatrix{T}; tol::T=T(1e-9)) where {T<:Real}
        n = size(A,1); @assert size(A,2) == n
        for k in 1:n
            for S in combinations(1:n, k)
                Mk = @view A[S,S]
                # cheap row screen: skip if a row has no positive entry
                badrow = any(maximum(@view Mk[i,:]) <= T(0) for i in 1:k)
                badrow && continue
                F = lu(Mk)
                any(abs.(diag(F.U)) .< tol) && continue
                x = F \ ones(T, k)
                if all(x .> tol)
                    return 1   # non-B ⇒ unphysical
                end
            end
        end
        return -1              # B-matrix ⇒ physical
    end
end

# ========= persistence sampling (generalized) =========

function sample_K_persistent_per_sigma(; K::Int=100, n::Int=8, mu::Float64=-1.0,
    sigma_vals = 1.0:0.2:2.8, rho::Float64=0.0, ϵ::Float64=1e-9, tol::Float64=1e-7,
    max_cycles_per_cell::Int=10_000_000, sampler::Function)

    onevec = ones(Float64, n)

    df_persist = DataFrame(
        sigma=Float64[], seed=Int[], delta=Float64[], p=Vector{Float64}[],
        A=Matrix{Float64}[], xstar=Vector{Float64}[], necessary_screenAx=Union{Bool,Nothing}[],
        n_boundary_points=Int[], sufficient_fail_trials=Int[], stable=Bool[], bflag=Int[]
    )

    df_necpass = DataFrame(
        sigma=Float64[], seed=Int[], A=Matrix{Float64}[], xstar=Vector{Float64}[],
        n_boundary_points=Int[], necessary_screenAx=Bool[], lp_success=Bool[],
        delta=Union{Float64,Missing}[], p=Union{Vector{Float64},Missing}[],
        sufficient_fails_since_last_success=Int[], stable=Union{Bool,Missing}[]
    )

    df_counts  = DataFrame(
        sigma=Float64[], found=Int[], attempts=Int[], infeasible_trials=Int[],
        hs_fail_trials=Int[], necAx_passes=Int[], necAx_fails=Int[], sufficient_fails_total=Int[]
    )

    for σ in sigma_vals
        found = 0; attempts = 0
        infeasible_trials = 0; hs_fail_trials = 0
        necAx_passes = 0; necAx_fails = 0
        sufficient_fails_total = 0
        sufficient_fails_since_last_success = 0

        while found < K && attempts < max_cycles_per_cell
            attempts += 1
            seed = 10_000 * round(Int, 100*σ) + attempts

            A = sampler(n, seed; mu=abs(mu), sigma=σ, rho=rho)
            r = onevec
            feas, xstar = is_feasible(A, r)
            if !feas
                infeasible_trials += 1
                continue
            end

            J = build_jacobian_at_equilibrium(A, xstar)
            if !necessary_conditions_persistence(A, J)
                hs_fail_trials += 1
                continue
            end
            st = is_stable(J)

            out = boundary_equilibria(A, r; tol=tol)
            necAx = out.necessary_maxAx
            necAx === true  && (necAx_passes += 1)
            necAx === false && (necAx_fails  += 1)
            isempty(out.lhs) && continue

            if necAx === true
                C   = build_C(out.lhs, out.rhs)
                res = find_p_maxmargin(C; ϵ=ϵ)
                lp_ok = (res !== nothing) && (res.δ > 0)

                push!(df_necpass, (
                    sigma=Float64(σ), seed=seed, A=A, xstar=xstar,
                    n_boundary_points=length(out.lhs), necessary_screenAx=true,
                    lp_success=lp_ok,
                    delta = lp_ok ? res.δ : missing,
                    p     = lp_ok ? res.p  : missing,
                    sufficient_fails_since_last_success = sufficient_fails_since_last_success,
                    stable = lp_ok ? st : missing
                ))

                if lp_ok
                    # bflag only needed for LP successes (physical/unphysical)
                    bflag_A = b_matrix_flag(A; tol=tol)  # -1 physical, +1 unphysical
                    push!(df_persist, (
                        sigma=Float64(σ), seed=seed, delta=res.δ, p=res.p,
                        A=A, xstar=xstar, necessary_screenAx=true,
                        n_boundary_points=length(out.lhs),
                        sufficient_fail_trials=sufficient_fails_since_last_success,
                        stable=st, bflag=bflag_A
                    ))
                    found += 1
                    sufficient_fails_since_last_success = 0
                else
                    sufficient_fails_since_last_success += 1
                    sufficient_fails_total += 1
                end
            end
        end

        push!(df_counts, (sigma=Float64(σ), found=found, attempts=attempts,
            infeasible_trials=infeasible_trials, hs_fail_trials=hs_fail_trials,
            necAx_passes=necAx_passes, necAx_fails=necAx_fails,
            sufficient_fails_total=sufficient_fails_total))
    end

    return df_persist, df_counts, df_necpass
end

# ========= ASCII fate-tree printer with stability × bflag split =========

function print_fate_trees(df_counts::DataFrame, df_persist::DataFrame)
    pct(x, tot) = tot == 0 ? "—" : @sprintf("%.1f%%", 100*x/tot)
    # group successes by σ for the final split
    byσ = groupby(df_persist, :sigma)
    for r in eachrow(df_counts)
        σ          = r.sigma
        attempts   = r.attempts
        infeas     = r.infeasible_trials
        feasible   = attempts - infeas
        hs_fail    = r.hs_fail_trials
        hs_pass    = feasible - hs_fail
        nec_pass   = r.necAx_passes
        nec_fail   = r.necAx_fails
        found      = r.found
        lp_fail_np = max(nec_pass - found, 0)

        # LP-success categories (stable×bflag)
        dfσ = @view df_persist[df_persist.sigma .== σ, :]
        sp = sum((dfσ.stable .== true)  .& (dfσ.bflag .== -1))  # stable & physical
        up = sum((dfσ.stable .== false) .& (dfσ.bflag .== -1))  # unstable & physical
        su = sum((dfσ.stable .== true)  .& (dfσ.bflag .==  1))  # stable & unphysical
        uu = sum((dfσ.stable .== false) .& (dfσ.bflag .==  1))  # unstable & unphysical


        println("")
        println("σ=$(σ)  (attempts=$(attempts))")
        println("├─ infeasible: $(infeas)  ($(pct(infeas, attempts)))")
        println("└─ feasible:   $(feasible) ($(pct(feasible, attempts)))")
        println("   ├─ HS fail:   $(hs_fail)  ($(pct(hs_fail, feasible)))")
        println("   └─ HS pass:   $(hs_pass)  ($(pct(hs_pass, feasible)))")
        println("      ├─ necAx fail: $(nec_fail) ($(pct(nec_fail, hs_pass))) → LP fail: $(nec_fail)")
        println("      └─ necAx pass: $(nec_pass) ($(pct(nec_pass, hs_pass)))")
        println("         ├─ LP success: $(found) ($(pct(found, nec_pass)))")
        println("         │   ├─ stable & physical:     $(sp) ($(pct(sp,   found)))   (bflag = -1)")
        println("         │   ├─ unstable & physical:   $(up) ($(pct(up,   found)))   (bflag = -1)")
        println("         │   ├─ stable & unphysical:   $(su) ($(pct(su,   found)))   (bflag =  1)")
        println("         │   └─ unstable & unphysical: $(uu) ($(pct(uu,   found)))   (bflag =  1)")
        println("         └─ LP fail:    $(lp_fail_np) ($(pct(lp_fail_np, nec_pass)))")
        println()
    end
end

# ========= run all three distributions =========

const DISTS = [
    ("Normal (1/n, matched μ,σ)", sampler_normal_rank1),
    ("Gamma  (1/n, matched μ,σ)", sampler_gamma),
    ("Lognormal (1/n, matched μ,σ)", sampler_lognormal),
]

function run_all(; n::Int=8, mu::Float64=-1.0, sigma_vals=2.6:.2:2.8, K::Int=100,
                  rho::Float64=0.0, ϵ::Float64=1e-9, tol::Float64=1e-7,
                  max_cycles_per_cell::Int=10_000_000)
    for (label, sampler) in DISTS
        println("\n==================== $label ====================")
        df_persist, df_counts, _ = sample_K_persistent_per_sigma(
            ; K=K, n=n, mu=mu, sigma_vals=sigma_vals, rho=rho,
              ϵ=ϵ, tol=tol, max_cycles_per_cell=max_cycles_per_cell,
              sampler=sampler
        )
        print_fate_trees(df_counts, df_persist)
    end
end

# Execute
run_all(sigma_vals = [2.8], n = 8, mu = -1.0, K = 100)
