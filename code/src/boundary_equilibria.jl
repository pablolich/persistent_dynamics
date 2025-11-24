using RecursiveFactorization, LinearSolve, Combinatorics, LinearAlgebra, Statistics

include("necessary_conditions.jl")

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
    #check if all elements in r are 1
    all_r_one = all(r .== one(T))

    # Extended matrix M = [A  r; 0  0]
    M = zeros(T, n + 1, n + 1)
    @views M[1:n, 1:n]   .= A
    @views M[1:n, n + 1] .= r

    # Rank-1 shift J = 11ᵀ (ensures invertibility on boundaries)
    M2 = M .+ (ones(T, n + 1) * ones(T, n + 1)')

    #initialize accumulators
    E   = Vector{Vector{T}}()
    lhs = Vector{Vector{T}}()
    rhs = Vector{T}()

    nec_fail = false
    found_nonB = false
    nM = n + 1

    for k in 1:nM-1 # enumerate all nonempty sub-community sizes
        onesk = ones(T, k)
        for S in Combinatorics.combinations(1:nM, k) # all index sets of size k
            Mk = @view M2[S, S]

            # ---- cheap screens ---- (only for r = 1)
            if all_r_one
                necessary_conditions_feas_r_one(Mk, k) || continue
            end

            # ---- face solve ----
            prob = LinearProblem(-Mk, onesk)
            xk = try 
                solve(prob, RFLUFactorization()).u
            catch
                continue
            end
            make_positive_if_same_sign!(xk, tol) || continue # we have one free variable

            # ---- embed + normalize ----
            p = zeros(T, nM); p[S] = xk
            s = sum(p); s <= tol && continue
            p ./= s

            # ---- growths & game value ----
            @views g = M * p
            φ = sum(p .* g)
            @views y = g[1:n]

            # ---- NECESSARY CONDITION FOR STRONG PERMANENCE (there can't be saturated equilibria on the boudnary) ----
            if !necessary_conditions_strong_perm(y, S, φ, tol, n)
                nec_fail = true
                break
            end

            # B-flag reuse on pure-A faces S ⊆ {1..n}, only if all growth rates are 1
            if all_r_one && !found_nonB && all(i -> i <= n, S)
                AS = @view A[S,S]
                v  = AS * xk
                α  = mean(v)
                # If v = A[S,S]*xk is nearly constant (≈α·1), rescale xk to solve A[S,S]*x ≈ 1 and, if x > 0, mark A as non-B.
                if abs(α) > tol && maximum(abs.(v .- α)) <= 10*tol
                    xA = xk ./ α
                    all(xA .> tol) && (found_nonB = true)
                end
            end

            # ---- accumulate ----
            push!(E, p)
            push!(lhs, g)
            push!(rhs, φ)
        end
        nec_fail && break
    end
    # B-flag: 1 (non-B), -1 (B), or nothing if growth rates not all 1
    bflag = all_r_one ? (found_nonB ? 1 : -1) : nothing

    if nec_fail
        return (E = Vector{Vector{T}}(),
                lhs = Vector{Vector{T}}(),
                rhs = Vector{T}(),
                necessary_maxAx = false,
                bflag = bflag)
    else
        return (E = E,
                lhs = lhs,
                rhs = rhs,
                necessary_maxAx = true,
                bflag = bflag)
    end
end

# include("feasibility.jl")

# # Sample data
# A = sampler_normal_f32(15, 1)
# r = ones(Float32, 15)

# # Call boundary_equilibria
# result = boundary_equilibria(A, r)

# # Print equilibria
# println("Equilibria:")
# for p in result.E
#     println(p)
# end