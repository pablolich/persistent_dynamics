#necessary conditions functions

function necessary_conditions_feas_r_one(Mk::AbstractMatrix{T}, k::Int) where {T <: Real}
    # At least one positive column sum
    cs = vec(sum(Mk; dims = 1)) # column sums
    if maximum(cs) <= T(0)
        return false
    end
    # At least one positive entry per row
    @inbounds for i in 1:k
        if maximum(@view Mk[i, :]) <= T(0) # row max test
            return false
        end
    end
    return true
end

function necessary_conditions_persistence(A::Matrix{T}, J::Matrix{T}) where {T <: Real}
    n = size(A, 1)
    #13.19
    condition_a = (-1)^n * det(A) > 0
    #13.20
    condition_b = tr(J) < 0
    #13.21
    condition_c = (-1)^n * det(J) > 0
    return condition_a && condition_b && condition_c
end

function necessary_conditions_strong_perm(y::AbstractVector{T}, S::AbstractVector{Int}, φ::T, tol::T, n::Int) where {T <: Real}
    inS = falses(n)
    @inbounds for i in S
        i <= n && (inS[i] = true)
    end
    if !all(inS)
        off = findall(!, inS)
        offmax = maximum(@view y[off])
        if !(offmax > φ + tol)
            return false
        end
    end
    return true
end