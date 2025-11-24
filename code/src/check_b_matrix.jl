using RecursiveFactorization, LinearSolve, Combinatorics, LinearAlgebra

function solve_subsystems_until_positive(A::Matrix{T}) where {T <: Real}
    n = size(A, 1)

    # loop over all non-empty masks
    for mask in UInt(1):UInt(2)^UInt(n) - UInt(1)
        # build index set S from mask
        S = findall(i -> ((mask >> (i - 1)) & 0x1) == 0x1, 1:n)
        isempty(S) && continue

        A_S = @view A[S, S]
        b_S = ones(T, length(S))

        try
            prob = LinearProblem(A_S, b_S)
            x = solve(prob, RFLUFactorization()).u
            if all(x .> zero(T))
                return 1   # first success
            end
        catch
            # singular / badly conditioned, just skip
        end
    end

    return -1   # no subsystem with x > 0 found
end


# using Test


# # Sample matrix A (positive definite, so solution exists)
# A1 = [2.0 1.0; 1.0 2.0]
# # Matrix with no positive solution
# A2 = [-1.0 0.0; 0.0 -2.0]
# # Matrix with mixed signs
# A3 = [1.0 -1.0; -1.0 2.0]

# @testset "solve_subsystems_until_positive tests" begin
#     res1 = solve_subsystems_until_positive(A1)
#     @test res1 !== nothing
#     @test all(res1[2] .> 0)

#     res2 = solve_subsystems_until_positive(A2)
#     @test res2 === nothing

#     res3 = solve_subsystems_until_positive(A3)
#     if res3 !== nothing
#         @test all(res3[2] .> 0)
#     end
# end