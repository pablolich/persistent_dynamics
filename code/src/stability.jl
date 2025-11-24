#functions for stability analysis

function build_jacobian_at_equilibrium(A::Matrix{T}, xstar::Vector{T}) where {T <: Real}
    return Diagonal(xstar) * A
end

function is_stable(J::Matrix{T}) where {T <: Real}
    eigvals = eigen(J).values
    return all(real.(eigvals) .< 0.0)
end