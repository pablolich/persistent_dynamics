#code to sample systems with persistent dynamics
using Distributions, Random
using Base.Threads: Atomic, @spawn, nthreads
using LinearAlgebra
using Combinatorics
using DifferentialEquations, Plots
using DataFrames
using CairoMakie
BLAS.set_num_threads(1)  # avoid oversubscription with Julia threads
import Printf: @sprintf

# seed layout scales (avoid collisions across grid dims)
const MU_SCALE    = 1_000_000_000
const N_SCALE     = 1_000_000
const REP_SCALE   = 10_000
const SIGMA_SCALE = 1_000

#I. functions to sample a system: 
# a. sample matrix of interactions
# b. sample growth rates (for now, all 1's)

function sample_interaction(n::Int, seed::Int=42; 
                            mu::Float64=0.0, sigma::Float64=1.0, rho::Float64=0.0)
    Random.seed!(seed)
    mean = fill(mu / n, 2)
    cov = (sigma^2 / n) * [1.0 rho; rho 1.0]
    A = zeros(n, n)
    for i in 1:n-1, j in i+1:n
        pair = rand(MvNormal(mean, cov))
        A[i, j] = pair[1]
        A[j, i] = pair[2]
    end
    for i in 1:n
        A[i, i] = -1.0
    end
    return A
end

function sample_growth_rates(n::Int, seed::Int=42, model::Symbol=:ones; 
                             mu::Float64=1.0, sigma::Float64=0.0)
    if model == :ones
        return ones(n)
    end
end

#II. function to filter out unfeasible systems

function is_feasible(A::Matrix{Float64}, r::Vector{Float64})
    n = size(A, 1)
    x_eq = -A \ r
    return all(x_eq .> 0.0), x_eq
end

#III. function to filter out not-persistent systems
# a. using conditions in determinant from H & S
function build_jacobian_at_equilibrium(A::Matrix{Float64}, xstar::Vector{Float64})
    return Diagonal(xstar) * A
end

function is_stable(J::Matrix{Float64})
    eigvals = eigen(J).values
    return all(real.(eigvals) .< 0.0)
end

function necessary_conditions_persistence(A::Matrix{Float64}, J::Matrix{Float64})
    n = size(A, 1)
    #13.19
    condition_a = (-1)^n * det(A) > 0
    #13.20
    condition_b = tr(J) < 0
    #13.21
    condition_c = (-1)^n * det(J) > 0
    return condition_a && condition_b && condition_c
end

# b. finding a nash equilibrium on the boundary
#    i. use genetic algorithm if n < 10
#    ii. interrupt exhaustive search for persistentn if I find a saturated equilibirum

#IV. main function to sample persistent systems
# a. add option to check for B-matrix while checking for persistence.
# b. function builds all equilibria in the boundary
# c. solves the set of inequalities for a positive p in the simplex

# --- helper: flip to positive if all entries share the same (nonzero) sign ---
@inline function make_positive_if_same_sign!(x::AbstractVector{T}, tol::T) where {T<:Real}
    pos = all(x .>  tol)
    neg = all(x .< -tol)
    if !(pos || neg)
        return false
    end
    if neg
        @inbounds @simd for i in eachindex(x)
            x[i] = -x[i]
        end
    end
    return true
end

"""
    boundary_equilibria_bflag(M2; tol=1e-7, M_for_growth=M2)

Enumerate boundary supports (sizes 1..n-1). For each support S, solve Mk*x = 1
(Mk = M2[S,S]); enforce same-sign via helper above; require x > 0; embed in ℝⁿ,
normalize to the simplex, and compute invasion growths: g = M_for_growth * p,
rhs = sum(p .* g). It also determines the B-matrix flag of A by reusing solves 
on pure-A faces S ⊆ {1..n}.

Threaded over support size k.

Returns NamedTuple:
- E   :: Vector{Vector{Float64}}    # equilibria in the simplex
- lhs :: Vector{Vector{Float64}}    # M_for_growth * p
- rhs :: Vector{Float64}            # sum(p .* (M_for_growth * p))
"""
# Joint pass: boundary equilibria on M and B-flag on A (reusing solves on M2[S,S])
function boundary_equilibria_b_flag(A::AbstractMatrix{T}, r::AbstractVector{T};
                                    tol::T = T(1e-7)) where {T<:Real}
    n = size(A,1)
    @assert size(A,2) == n
    @assert length(r) == n

    # Extended matrix M = [A  r; 0  0]
    M = zeros(T, n+1, n+1)
    @views M[1:n, 1:n] .= A
    @views M[1:n, n+1] .= r
    # replicator-invariant rank-1 shift J = 11ᵀ (only used for solves on faces)
    M2 = M .+ (ones(T, n+1) * ones(T, n+1)')

    nM = n + 1
    nT = Threads.nthreads()

    E_chunks   = [Vector{Vector{T}}() for _ in 1:nT]
    lhs_chunks = [Vector{Vector{T}}() for _ in 1:nT]
    rhs_chunks = [Vector{T}()         for _ in 1:nT]

    found_nonB = Atomic{Bool}(false)  # B-flag accumulator (true ⇒ non-B ⇒ +1)

    Threads.@threads for k in 1:nM-1
        tid   = Threads.threadid()
        onesk = ones(T, k)

        for S in Combinatorics.combinations(1:nM, k)
            Mk = @view M2[S, S]

            # cheap screens (skip impossible faces quickly)
            colsum = vec(sum(Mk; dims=1))
            if maximum(colsum) <= T(0)
                continue
            end
            badrow = false
            @inbounds for i in 1:k
                if maximum(@view Mk[i, :]) <= T(0)
                    badrow = true; break
                end
            end
            if badrow; continue; end

            # LU solve on M2[S,S]
            F = lu(Mk)
            if any(abs.(diag(F.U)) .< tol)
                continue
            end

            xk = F \ onesk
            if !make_positive_if_same_sign!(xk, tol)
                continue
            end

            # embed and normalize (length n+1)
            p = zeros(T, nM); p[S] = xk
            s = sum(p); if s <= tol; continue; end
            p ./= s

            g = M * p
            @inbounds begin
                push!(E_chunks[tid], p)
                push!(lhs_chunks[tid], g)
                push!(rhs_chunks[tid], sum(p .* g))
            end

            # ---- B-matrix check reusing xk when S ⊆ {1..n} (pure-A face) ----
            if !found_nonB[] && all(i -> i <= n, S)
                AS = @view A[S, S]
                v  = AS * xk                 # should be α * 1 if xk ∝ A^{-1}1
                α  = mean(v)
                # consistency: entries ~ equal to α within tolerance
                if abs(α) > tol && maximum(abs.(v .- α)) <= 10*tol
                    xA = xk ./ α             # recover A^{-1}1 from (A+J)^{-1}1
                    if all(xA .> tol)
                        found_nonB[] = true   # non-B found
                    end
                end
            end
        end
    end

    # concatenate thread-local chunks
    E   = Vector{Vector{T}}()
    lhs = Vector{Vector{T}}()
    rhs = Vector{T}()
    for t in 1:nT
        append!(E,   E_chunks[t])
        append!(lhs, lhs_chunks[t])
        append!(rhs, rhs_chunks[t])
    end

    return (E=E, lhs=lhs, rhs=rhs, bflag = (found_nonB[] ? 1 : -1))
end

#VI. wrapper: sample feasible & persistent A; report stability and B-flag ---
function sample_persistent_glv(n::Int;
                               seed::Int=42, mu::Float64=0.0, sigma::Float64=1.0, rho::Float64=0.0,
                               max_tries::Int=10_000, ϵ::Float64=1e-9)
    for t in 1:max_tries
        A = sample_interaction(n, seed + t - 1; mu=mu, sigma=sigma, rho=rho)
        r = sample_growth_rates(n, seed + t - 1, :ones)

        feas, xeq = is_feasible(A, r)
        feas || continue

        ok, pstar, δ, bflag = is_persistent_glv(A, r, xeq; ϵ=ϵ)
        ok || continue

        J = build_jacobian_at_equilibrium(A, xeq)
        stable = is_stable(J)

        return (A=A, r=r, xeq=xeq, stable=stable, Bflag=bflag, p=pstar, δ=δ, tries=t)
    end
    return nothing
end

using LinearAlgebra, JuMP, HiGHS

# Build C where each row is cᵗ = (g - rhs*1)ᵗ from boundary_equilibria output
# inputs: lhs::Vector{Vector{T}}  (each g = M2*x)
#         rhs::Vector{T}          (each xᵗ M2 x)
function build_C(lhs::Vector{<:AbstractVector}, rhs::AbstractVector)
    m = length(lhs)
    n = length(lhs[1])
    C = Matrix{eltype(rhs)}(undef, m, n)
    one = ones(eltype(rhs), n)
    @inbounds for j in 1:m
        C[j, :] = lhs[j] .- rhs[j] .* one
    end
    return C
end

# Recommended: maximize the safety margin δ ≥ 0 subject to simplex and C*p ≥ δ
function find_p_maxmargin(C::AbstractMatrix; ϵ=1e-9)
    n = size(C, 2)
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, p[1:n] >= ϵ)
    @variable(model, δ >= 0.0)
    @constraint(model, sum(p) == 1)
    @constraint(model, C * p .>= δ)
    @objective(model, Max, δ)
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return (p = value.(p), δ = value(δ))
    else
        return nothing
    end
end

# --- persistence via boundary equilibria on M, reusing b-flag from same pass ---
# Returns (ok::Bool, p::Vector, δ::Float64, bflag::Int)  where bflag∈{1,-1}
function is_persistent_glv(A::AbstractMatrix{T}, r::AbstractVector{T}, xstar::AbstractVector{T};
                           ϵ::T = T(1e-9)) where {T<:Real}
    n = size(A,1); @assert size(A,2) == n; @assert length(r) == n

    #firs look at necessary conditions
    J = build_jacobian_at_equilibrium(A, xstar)
    if !necessary_conditions_persistence(A, J)
        return (false, zeros(T, n+1), zero(T), 0)
    end
    # one pass over faces: get (E,lhs,rhs) for M and bflag for A without recomputation
    out = boundary_equilibria_b_flag(A, r)  # uses M and recovers bflag for A
    #if no feasible boundary equilibria, abort
    if isempty(out.lhs)
        return (false, zeros(T, n+1), zero(T), out.bflag)
    end

    #now look at sufficient conditions
    C = build_C(out.lhs, out.rhs)           # rows: g - rhs*1 from the same pass on M
    res = find_p_maxmargin(C; ϵ=ϵ)
    if res === nothing
        return (false, zeros(T, n+1), zero(T), out.bflag)
    end

    return (res.δ > 0, res.p, res.δ, out.bflag)
end

# b. finding a nash equilibrium on the boundary
#    i. use genetic algorithm if n < 10
#    ii. interrupt exhaustive search for persistentn if I find a saturated equilibirum

#VII. a sanity check: simulate and plot dynamics of persistent, unstable systems 

# GLV dynamics:  ẋ = x .* (r + A*x)
function glv!(dx, x, p, t)
    A, r = p
    mul!(dx, A, x)           # dx = A*x
    @. dx = x * (r + dx)     # dx = x .* (r + A*x)
end

# simulate one system from a simplex initial condition
function simulate_glv(A::AbstractMatrix, r::AbstractVector;
                      tspan=(0.0, 50.0), saveat=0.1, seed=0)
    n = length(r)
    if seed != 0; Random.seed!(seed); end
    x0 = abs.(rand(n)); x0 ./= sum(x0)
    prob = ODEProblem(glv!, x0, tspan, (A, r))
    solve(prob, Tsit5(); reltol=1e-8, abstol=1e-10, saveat=saveat)
end

# collect 10 persistent but UNSTABLE systems of size 6 and plot dynamics
function check_and_plot(n::Int=6; target::Int=10, base_seed::Int=1)
    systems = Vector{NamedTuple}()  # store (A,r,Bflag,sol)
    t = 0
    while length(systems) < target
        print(length(systems) + 1, " ")
        t += 1
        A = sample_interaction(n, base_seed + t; mu = -2.0, sigma = 2.5)
        r = sample_growth_rates(n, base_seed + t, :ones)

        feas, xeq = is_feasible(A, r); feas || continue

        ok, pstar, δ, _ = is_persistent_glv(A, r, xeq)
        ok || continue

        J = build_jacobian_at_equilibrium(A, xeq)
        stable = is_stable(J)
        stable && continue  # we want UNSTABLE

        Bflag = b_matrix_flag(A)  # -1=B-matrix, +1=non-B

        sol = simulate_glv(A, r; seed=base_seed + t)
        push!(systems, (A=A, r=r, Bflag=Bflag, sol=sol))
    end

    plt = plot(layout=(2,5), size=(1500,650), legend=false)
    for (i, sys) in enumerate(systems)
        sol = sys.sol
        Y = hcat(sol.u...)'            # size: length(t) × n
        plot!(plt[i], sol.t, Y, lw=1.5,
              title = "sys $(i): Bflag=$(sys.Bflag)")
        xlabel!(plt[i], "t"); ylabel!(plt[i], "x_i")
        savefig(plt, "persistent_unstable_systems_n$(n).png")
    end
    systems
end

#VIII. functions to compute and plot p1, p2 stacked bars by (n, sigma)

# p1 := P(stable), p2 := P(nonB) per (n, sigma)
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
    CairoMakie.save("p1p2_stacks_by_sigma_mu_m1.png", fig)
    return fig
end

using DataFrames
using Statistics
using CairoMakie

"""
    unstable_heatmap(df; n::Union{Int,Nothing}=nothing, outfile="unstable_heatmap.png")

Compute P(unstable) on the (μ, σ) grid and plot a heatmap.
- If `n` is provided, restrict to that n; otherwise aggregate across all n.
- Returns (xs, ys, Z, summary::DataFrame).
"""
function unstable_heatmap(df::DataFrame; n::Union{Int,Nothing}=nothing,
                          outfile::AbstractString="unstable_heatmap.png")

    d = isnothing(n) ? df : filter(:n => ==(n), df)
    @assert nrow(d) > 0 "No rows after filtering."

    # summarize per (mu, sigma)
    summ = combine(groupby(d, [:mu, :sigma]),
                   :stable => (s -> mean(.!s)) => :p_unstable,
                   nrow => :count)

    xs = sort(unique(summ.mu))      # μ on x-axis
    ys = sort(unique(summ.sigma))   # σ on y-axis

    Z = fill(Float64(NaN), length(ys), length(xs))  # Z[j,i] = P(unstable) at (xs[i], ys[j])
    idx_x = Dict(x => i for (i,x) in enumerate(xs))
    idx_y = Dict(y => j for (j,y) in enumerate(ys))
    for row in eachrow(summ)
        i = idx_x[row.mu]
        j = idx_y[row.sigma]
        Z[j,i] = row.p_unstable
    end

    # plot
    fig = Figure(resolution=(760, 560))
    ax  = Axis(fig[1,1], xlabel="μ", ylabel="σ", title="P(unstable | persistent)")
    hm  = heatmap!(ax, xs, ys, Z)
    cb  = Colorbar(fig[1,2], hm, label="probability")
    fig[1,1] = ax
    CairoMakie.save(outfile, fig)

    return xs, ys, Z, summ
end

#IX. main function to sample persistent systems on a fixed mu - sigma sample_interaction
"""
    classify_persistent_grid(; nsim=10, mu_vals=-2.0:1.0:2.0, sigma_vals=1.0:0.1:3.0,
                              n_vals=3:11, base_seed=1, max_tries=500_000, ϵ=1e-9)

For each (n, σ, μ) on the grid and each rep ∈ 1:nsim, sample persistent GLV systems,
then classify stability and B-flag. Returns a DataFrame with columns:
seed, n, mu, sigma, stable, bflag.
"""
function classify_persistent_grid(; nsim::Int=10,
                                   mu_vals = -2.0:1.0:2.0,
                                   sigma_vals = 1.0:0.1:3.0,
                                   n_vals = 3:11,
                                   base_seed::Int=1,
                                   max_tries::Int=2_000_000,
                                   ϵ::Float64=1e-9)

    df = DataFrame(seed=Int[], n=Int[], mu=Float64[], sigma=Float64[],
                   stable=Bool[], bflag=Int[])

    for rep in 1:nsim
        for mu in mu_vals
            for sigma in sigma_vals
                for n in n_vals
                    println("rep=$rep, n=$n, μ=$(float(mu)), σ=$(float(sigma))")
                    found = false
                    for t in 1:max_tries
                        seed = base_seed +
                               MU_SCALE   * round(Int, float(mu)    * 100) +
                               N_SCALE    * n +
                               REP_SCALE  * rep +
                               SIGMA_SCALE* round(Int, float(sigma) * 100) +
                               t

                        A = sample_interaction(n, seed; mu=float(mu), sigma=float(sigma), rho=0.0)
                        r = sample_growth_rates(n, seed, :ones)

                        feas, xeq = is_feasible(A, r); feas || continue
                        ok, _, δ, bf = is_persistent_glv(A, r, xeq; ϵ=ϵ); ok || continue

                        J  = build_jacobian_at_equilibrium(A, xeq)
                        st = is_stable(J)

                        push!(df, (seed, n, float(mu), float(sigma), st, bf))
                        found = true
                        break
                    end
                    println(found ? "  → found persistent system" : "  → gave up after $max_tries tries")
                end
            end
        end

        # optional: update your summary plot (currently aggregates over μ)
        df_probs = p1p2_by_nsigma(df)
        try
            plot_p1p2_stacks_by_sigma(df_probs; size=(700,1500))
        catch err
            @warn "Plot update failed" err
        end
    end
    return df
end

# # Example:
sigma_vals = 1.0:0.1:3.
mu_vals = [-1.]
n_vals = 4:2:12
df = classify_persistent_grid(;nsim=1000, mu_vals = mu_vals, sigma_vals=sigma_vals, n_vals=n_vals)
df_probs = p1p2_by_nsigma(df)
plot_p1p2_stacks_by_sigma(df_probs; size=(700,1500))

# mu_range = (-2, 2)
# sigma_range = (1, 3)
# n_vals = [10]
# #Example on a grid
# mu_vals    = collect(LinRange(mu_range[1],    mu_range[2],    10))
# sigma_vals = collect(LinRange(sigma_range[1], sigma_range[2], 10))
# df = classify_persistent_grid(nsim=100, sigma_vals=sigma_vals, n_vals=n_vals)
# unstable_heatmap(df; n=10)

using Random, LinearAlgebra, Printf
BLAS.set_num_threads(1)  # avoid BLAS oversubscription with Julia threads

"""
    time_single_case(; n=20, sigma=3.0, mu=-2.0, base_seed=1, max_tries=300_000,
                      ϵ=1e-9, rho=0.0, status_period=1.0)

Search for a persistent GLV at (n, μ, σ) and print elapsed time in real time.
Returns (A, r, xeq, p, δ, stable, bflag, tries, elapsed) or `nothing`.
"""
function time_single_case(; n::Int=20, sigma::Float64=3.0, mu::Float64=-2.0,
                          base_seed::Int=1, max_tries,
                          ϵ::Float64=1e-9, rho::Float64=0.0, status_period::Float64=60.0)
    t0 = time(); last = t0
    t = 0
    while t < max_tries
        seed = base_seed + 1_000_000*n + 10_000*1 + 1_000*round(Int, sigma*100) + t
        A = sample_interaction(n, seed; mu=mu, sigma=sigma, rho=rho)
        r = sample_growth_rates(n, seed, :ones)

        feas, xeq = is_feasible(A, r)
        if feas
            ok, p, δ, bflag = is_persistent_glv(A, r, xeq; ϵ=ϵ)
            if ok
                st = is_stable(build_jacobian_at_equilibrium(A, xeq))
                elapsed = time() - t0
                @printf("FOUND after %d tries | elapsed = %.2fs | δ = %.3g | stable=%s | bflag=%d\n",
                        t, elapsed, δ, st, bflag)
                flush(stdout)
                return (A=A, r=r, xeq=xeq, p=p, δ=δ, stable=st, bflag=bflag, tries=t, elapsed=elapsed)
            end
        end

        now = time()
        if now - last ≥ status_period
            @printf("... tries=%d | elapsed=%.2fs\n", t, now - t0)
            flush(stdout)
            last = now
        end
        t = t + 1
    end
    elapsed = time() - t0
    @printf("GAVE UP after %d tries | elapsed = %.2fs\n", max_tries, elapsed)
    flush(stdout)
    return nothing
end

# Example run:
#res = time_single_case(n=20, sigma=3.0, mu=-2.0, base_seed=1, max_tries=Inf, ϵ=1e-9)


# # --- exact reconstruction matching your sampling in classify_persistent_slice ---

# function solve_subsystems_until_positive(A::Matrix{Float64})
#     n = size(A, 1)
#     found_flag = Atomic{Bool}(false)

#     @sync for tid in 1:nthreads()
#         @spawn begin
#             for mask in UInt(tid):UInt(nthreads()):UInt(2)^n - 1
#                 found_flag[] && break

#                 S = findall(i -> (mask >> (i - 1)) & 1 == 1, 1:n)
#                 isempty(S) && continue
#                 A_S = A[S, S]
#                 b_S = ones(length(S))

#                 try
#                     x = A_S \ b_S
#                     if all(x .> 0)
#                         found_flag[] = true
#                         break
#                     end
#                 catch; continue; end
#             end
#         end
#     end

#     return found_flag[] ? "unbounded" : "bounded"
# end

# function reconstruct_system(row::DataFrameRow)
#     n    = Int(row.n)
#     μ    = Float64(row.mu)
#     σ    = Float64(row.sigma)
#     seed = Int(row.seed)

#     A = sample_interaction(n, seed; mu=μ, sigma=σ, rho=0.0)
#     r = sample_growth_rates(n, seed, :ones)
#     return A, r, seed
# end


# function run_test_I(df::DataFrame; outdir::AbstractString="plots_testI",
#                     min_per_flag::Int=30, tspan::Tuple{<:Real,<:Real}=(0.0, 50.0))

#     # --- minimalist inlined plotter (no external helpers) ---
#     _plot_dynamics = function (A::AbstractMatrix{<:Real}, r::AbstractVector{<:Real};
#                                outpath::AbstractString, tspan::Tuple{<:Real,<:Real},
#                                title::AbstractString="")
#         n = size(A, 1)
#         @assert size(A,2) == n
#         @assert length(r) == n

#         # GLV dynamics:  x' = x .* (r + A*x)
#         function glv!(dx, x, p, t)
#             @inbounds mul!(dx, p.A, x)         # dx = A*x
#             @inbounds @. dx = x * (p.r + dx)   # dx .= x .* (r + A*x)
#             return nothing
#         end

#         x0 = ones(eltype(A), n)               # simple deterministic IC
#         prob = ODEProblem(glv!, x0, tspan, (; A=A, r=r))
#         sol  = solve(prob, Tsit5(); saveat=range(tspan[1], tspan[2], length=400),
#                      abstol=1e-9, reltol=1e-7)

#         t  = sol.t
#         X  = hcat(sol.u...)                   # size: n × T
#         X .= max.(X, zero(eltype(X)))         # clip negatives for readability

#         fig = Figure(resolution=(900, 320))
#         ax  = Axis(fig[1,1], xlabel="time", ylabel="abundance", title=title)
#         for i in 1:n
#             lines!(ax, t, X[i, :], linewidth=1.6)
#         end
#         fig[1,1] = ax
#         save(outpath, fig)
#         return nothing
#     end

#     # two flat folders only
#     stable_dir   = joinpath(outdir, "stable");   mkpath(stable_dir)
#     unstable_dir = joinpath(outdir, "unstable"); mkpath(unstable_dir)

#     g = groupby(df, [:mu, :sigma, :n])
#     for sub in g
#         μ = first(sub.mu); σ = first(sub.sigma); n = first(sub.n)
#         for flag in (true, false)  # true → stable, false → unstable
#             rows = sub[sub.stable .== flag, :]
#             ncat = nrow(rows); ncat == 0 && continue
#             m = max(ncat, min_per_flag)

#             rng = MersenneTwister(xor(hash((μ, σ, n, flag)), 0xC0FFEE))
#             ids = [rand(rng, 1:ncat) for _ in 1:m]

#             target_dir = flag ? stable_dir : unstable_dir
#             for (j, idx) in enumerate(ids)
#                 row = DataFrameRow(rows, idx)
#                 A, r, seed = reconstruct_system(row)  # exact reproduction of your sampling
#                 fn = @sprintf("mu=%.6g_sigma=%.6g_n=%d_seed=%d_idx=%04d.png", μ, σ, n, seed, j)
#                 outpng = joinpath(target_dir, fn)
#                 _plot_dynamics(A, r; outpath=outpng, tspan=tspan,
#                                 title=@sprintf("n=%d, μ=%.6g, σ=%.6g", n, μ, σ))
#             end
#         end
#     end
#     return nothing
# end

# # TEST II (recompute bflag from A using the SAME seed)
# _bis_to_bflag(s::AbstractString) = s == "unbounded" ? 1 : -1
# _norm_bflag(x::Int) = x > 0 ? 1 : -1

# function run_test_II(df::DataFrame)
#     df2 = deepcopy(df)
#     b_bis   = Vector{Int}(undef, nrow(df2))
#     b_match = Vector{Bool}(undef, nrow(df2))

#     for i in 1:nrow(df2)
#         row = DataFrameRow(df2, i)
#         n, μ, σ, seed = Int(row.n), Float64(row.mu), Float64(row.sigma), Int(row.seed)

#         A = sample_interaction(n, seed; mu=μ, sigma=σ, rho=0.0)  # exact reproduction
#         bis  = solve_subsystems_until_positive(A)                 # "unbounded"/"bounded"
#         bbis = _bis_to_bflag(bis)
#         borig = _norm_bflag(Int(row.bflag))

#         b_bis[i]   = bbis
#         b_match[i] = (bbis == borig)
#     end

#     df2.bflag_bis   = b_bis
#     df2.bflag_match = b_match

#     summary = combine(groupby(df2, [:mu, :sigma, :n]),
#                       :bflag_match => (x -> (all_match = all(x),
#                                              nrows     = length(x),
#                                              n_ok      = count(identity, x),
#                                              n_bad     = count(!, x))) => AsTable)
#     return df2, summary
# end

# run_test_I(df)
# run_test_II(df)

# #do a run over a parameter grid mu-sigma (mu in [-2, 2], sigma in [1, 3]), grid is 15 by 15, 100 reps in each cell. 
