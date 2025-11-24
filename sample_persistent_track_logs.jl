#this script identifies persistent systems and characterizes the p_opt, the nash equilibria, etc.
using LinearAlgebra, JuMP, HiGHS
using Random, DataFrames, Statistics
using Distributions, Combinatorics
using CairoMakie, Colors
CairoMakie.activate!()



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

function sampler_normal_rank1(n::Int, seed::Int; mu::Float64, sigma::Float64, rho::Float64=0.0)
    @assert sigma > 0
    @assert -1.0 < rho < 1.0
    Random.seed!(seed)

    A = Matrix{Float64}(undef, n, n)
    m = fill(mu/n, 2)                       # mean before sign flip
    Σ = (sigma^2/n) * [1.0 rho; rho 1.0]
    d = MvNormal(m, Matrix(Σ))

    @inbounds for i in 1:n-1, j in i+1:n
        pair = rand(d)
        A[i, j] = pair[1]
        A[j, i] = pair[2]
    end
    @inbounds for i in 1:n
        A[i, i] = -1.0
    end
    return A
end

# All-at-once sampler, in-place; single allocation (A) total, reuses RNG
function sampler_normal_allatonce(n::Int, seed::Int;
                                   mu::Float64, sigma::Float64, rho::Float64=0.0)
    Random.seed!(seed)
    @assert sigma > 0
    A = Matrix{Float64}(undef, n, n)
    randn!(A)                          # fill in-place
    @inbounds @. A = muladd(sigma / sqrt(n), A, mu/n)   # A .= sigma*A .+ mu
    @inbounds A[diagind(A)] .= -1.0
    return A
end

function is_feasible(A::Matrix{Float64}, r::Vector{Float64})
    n = size(A, 1)
    x_eq = -A \ r
    return all(x_eq .> 0.0), x_eq
end

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

# Enumerate boundary equilibria; INTERRUPT if necessary condition fails:
# require max(Ax_off) > φ (to tolerance). On first failure, return empties.
function boundary_equilibria(A::AbstractMatrix{T}, r::AbstractVector{T};
                             tol::T = T(1e-7)) where {T<:Real}
    n = size(A, 1)
    @assert size(A, 2) == n
    @assert length(r) == n

    # Extended matrix M = [A  r; 0  0]
    M = zeros(T, n + 1, n + 1)
    @views M[1:n, 1:n]   .= A
    @views M[1:n, n + 1] .= r

    # Rank-1 shift J = 11ᵀ (improves conditioning on faces)
    M2 = M .+ (ones(T, n + 1) * ones(T, n + 1)')

    E   = Vector{Vector{T}}()
    lhs = Vector{Vector{T}}()
    rhs = Vector{T}()

    nec_fail = false
    nM = n + 1

    for k in 1:nM-1                          # enumerate all nonempty proper faces
        onesk = ones(T, k)
        for S in Combinatorics.combinations(1:nM, k)
            Mk = @view M2[S, S]

            # ---- cheap screens ----
            cs = vec(sum(Mk; dims = 1))
            if maximum(cs) <= T(0); continue; end
            badrow = false
            @inbounds for i in 1:k
                if maximum(@view Mk[i, :]) <= T(0)
                    badrow = true; break
                end
            end
            badrow && continue

            # ---- face solve ----
            F = lu(Mk)
            any(abs.(diag(F.U)) .< tol) && continue
            xk = F \ onesk
            make_positive_if_same_sign!(xk, tol) || continue

            # ---- embed + normalize ----
            p = zeros(T, nM); p[S] = xk
            s = sum(p); s <= tol && continue
            p ./= s

            # ---- growths & game value ----
            @views g = M * p
            φ = sum(p .* g)
            @views y = g[1:n]

            # ---- NECESSARY CONDITION (early exit) ----
            inS = falses(n)
            @inbounds for i in S
                i <= n && (inS[i] = true)
            end
            if !all(inS)                     # there is at least one off-support species
                off = findall(!, inS)        # indices (not a mask)
                offmax = maximum(@view y[off])
                if !(offmax > φ + tol)
                    nec_fail = true
                    break
                end
            end

            # ---- accumulate ----
            push!(E, p)
            push!(lhs, g)
            push!(rhs, φ)
        end
        nec_fail && break
    end

    if nec_fail
        return (E = Vector{Vector{T}}(),
                lhs = Vector{Vector{T}}(),
                rhs = Vector{T}(),
                necessary_maxAx = false)
    else
        return (E = E, lhs = lhs, rhs = rhs, necessary_maxAx = true)
    end
end

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

"""
Maximize a uniform invasion margin δ ≥ 0 over the simplex.

Given C ∈ ℝ^{K×n} with rows c_k = g^{(k)} - π^{(k)}·1,
solve   max_{p∈Δ, p_i≥ϵ}  min_k  c_kᵀ p.
Returns (p, δ) if optimal; otherwise `nothing`.
"""
function find_p_maxmargin(C::AbstractMatrix; ϵ=1e-9)
    n = size(C, 2)

    # LP: maximize δ subject to simplex p and C*p ≥ δ·1
    model = Model(HiGHS.Optimizer); set_silent(model)
    @variable(model, p[1:n] >= ϵ)          # interior simplex (p_i ≥ ϵ)
    @variable(model, δ >= 0.0)             # common margin (uniform across rows)
    @constraint(model, sum(p) == 1)        # p ∈ Δ
    @constraint(model, C * p .>= δ)        # every row c_kᵀ p ≥ δ
    @objective(model, Max, δ)              # maximize the worst-case margin
    optimize!(model)

    # If optimal, return the certificate (p, δ); else no certification.
    if termination_status(model) == MOI.OPTIMAL
        return (p = value.(p), δ = value(δ))
    else
        return nothing
    end
end

#plotting functions
function p1p2_by_nsigma(df::DataFrame)
    out = DataFrame(n=Int[], sigma=Float64[], p1=Float64[], p2=Float64[])
    for g in groupby(df, [:n, :sigma])
        sub = dropmissing(g, [:stable, :bflag])
        isempty(sub) && continue
        p1 = mean(Bool.(sub.stable))     # P(stable)
        p2 = mean(sub.bflag .== 1)       # P(nonB) (bflag == 1)
        push!(out, (n=first(sub.n), sigma=Float64(first(sub.sigma)), p1=p1, p2=p2))
    end
    sort!(out, [:n, :sigma])
    return out
end

function plot_p1p2_single!(fig, cell, df_probs::DataFrame;
                           title::AbstractString,
                           colors=(colorant"steelblue", colorant"lightskyblue",
                                   colorant"tomato", colorant"peachpuff"))
    sub = sort(select(df_probs, :sigma, :p1, :p2), :sigma)
    T = nrow(sub); T == 0 && error("empty df")
    t  = collect(sub.sigma)
    p1 = collect(sub.p1)
    p2 = collect(sub.p2)

    a = p1 .* p2
    b = p1 .* (1 .- p2)
    c = (1 .- p1) .* p2
    d = (1 .- p1) .* (1 .- p2)

    heights = vcat(hcat(a, b, c, d)'...)           # [a1,b1,c1,d1,a2,b2,c2,d2,...]
    cat     = repeat(1:T, inner=4)                 # 1,1,1,1, 2,2,2,2, ...
    grp     = repeat(1:4, outer=T)                 # 1,2,3,4, 1,2,3,4, ...
    colv    = getindex.(Ref(colors), grp)

    ax = Axis(fig[cell...], xlabel="σ", ylabel="probability",
              xticks=(collect(1:T) .+ 2, string.(t)), title=title)

    barplot!(ax, cat .+ 2, heights; stack=grp, color=colv, width=1.0, gap=0.0)

    elems  = [PolyElement(color=colors[i]) for i in 1:4]
    labels = ["Stable & non-B", "Stable & B", "Unstable & non-B", "Unstable & B"]
    return ax, elems, labels
end



########################################################################################

"""
    run_grid_persistent_bflag_reps(n_vals, sigma_vals;
                                   K::Int=1000,
                                   mu::Float64=-1.0,
                                   rho::Float64=0.0,
                                   ϵ::Float64=1e-9,
                                   tol::Float64=1e-7,
                                   max_attempts_per_cell::Int=10_000_000)

Do K *replicates*; in each replicate sweep all (n, σ) cells before moving to the next.
For every (n, σ, rep), find one persistent system using `sampler_normal_allatonce`,
certify with LP built from `boundary_equilibria_b_flag_efficient`, then record stability and bflag.

Returns a DataFrame with columns: :rep, :n, :sigma, :stable, :bflag.
Prints a one-line message every time a persistent system is found.
"""
function run_grid_persistent_bflag(n_vals, sigma_vals;
                                        K::Int=1000,
                                        mu::Float64=-1.0,
                                        rho::Float64=0.0,
                                        ϵ::Float64=1e-9,
                                        tol::Float64=1e-7,
                                        max_attempts_per_cell::Int=1_000_000_000)

    df = DataFrame(rep=Int[], n=Int[], sigma=Float64[], stable=Bool[], bflag=Int[])

    # helper: deterministic seed base per (rep, n, σ)
    seed_base(rep::Int, n::Int, σ::Float64) =
        10_000_000_000*rep + 1_000_000*n + 10_000*round(Int, 100*σ)

    for rep in 1:K
        for n in n_vals
            onevec = ones(Float64, n)
            for σ in sigma_vals
                attempts = 0
                base = seed_base(rep, n, Float64(σ))

                found_this_cell = false
                while attempts < max_attempts_per_cell
                    attempts += 1
                    seed = base + attempts

                    # 1) sample A
                    A = sampler_normal_allatonce(n, seed; mu=mu, sigma=σ, rho=rho)

                    # 2) feasibility at r = 1
                    feas, xstar = is_feasible(A, onevec)
                    feas || continue

                    # 3) Jacobian + necessary screen
                    J = build_jacobian_at_equilibrium(A, xstar)
                    nec_ok = necessary_conditions_persistence(A, J)
                    nec_ok || continue

                    # 4) boundary enumeration (with bflag)
                    out = boundary_equilibria_b_flag_efficient(A, onevec; tol=tol)
                    isempty(out.lhs) && continue

                    # 5) LP certificate
                    C   = build_C(out.lhs, out.rhs)
                    res = find_p_maxmargin(C; ϵ=ϵ)
                    (res === nothing || res.δ <= 0) && continue

                    # 6) stability
                    st = is_stable(J)

                    # record + PRINT
                    push!(df, (rep=rep, n=n, sigma=Float64(σ), stable=st, bflag=out.bflag))
                    println("FOUND rep=$rep n=$n σ=$(Float64(σ)) attempts=$attempts stable=$st bflag=$(out.bflag) δ=$(round(res.δ, digits=6))")
                    flush(stdout)

                    found_this_cell = true
                    break
                end

                if !found_this_cell
                    @warn "No persistent system found" rep=rep n=n σ=Float64(σ) attempts=max_attempts_per_cell
                end
            end
        end
        let d=p1p2_by_nsigma(filter(:rep => <=(rep), df)); f=Figure(resolution=(900,400)); ax,es,ls=plot_p1p2_single!(f,(1,1),d; title="Progress up to rep=$(rep)"); Legend(f[2,1], es, ls; orientation = :horizontal, title = ""); save("progress_n_$(maximum(n_vals))_p1p2.png", f) end
    end

    return df
end

# Example:
# df = run_grid_persistent_bflag([14], 1.0:0.1:2.8; K=1, mu=-1.0, rho=0.0)
