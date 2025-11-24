using JuMP, HiGHS

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

function determine_persistence_type(lhs::Vector{<:AbstractVector}, rhs::AbstractVector; ϵ=1e-9)
    C = build_C(lhs, rhs)
    res = find_p_maxmargin(C; ϵ=ϵ)
    if res === nothing
        return "unknown"
    elseif res.δ > 0
        return "strong"
    else res.δ <= 0
        return "weak"
    end
end