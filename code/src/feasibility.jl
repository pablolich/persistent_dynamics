using Random, LinearSolve, RecursiveFactorization


"""
Generate an n x n matrix with Float32 entries from a normal distribution
with mean `mu` and standard deviation `sigma`. The diagonal entries are set to -1
"""
function sampler_normal_f32(n::Int, rng::Any;
                            mu::Float32=0.0f0, 
                            sigma::Float32=1.0f0, 
                            rho::Float32=0.0f0) #rho is left for future use
    @assert sigma > 0.
    A = Matrix{Float32}(undef, n, n)
    randn!(rng, A)
    s = sigma / sqrt(n)
    m = mu / n
    @inbounds @. A = muladd(s, A, m)
    @inbounds for i in 1:n
        A[i, i] = -1.0f0
    end    
    return A
end 

"""
A structure to hold model information: name, dimension, and parameters.
`pars` is a NamedTuple to allow flexible parameter sets for different models.
"""
struct EcologicalModel
    name::String
    n::Int
    pars::NamedTuple
end

"""
Construct a model structure given the model name, dimension, and parameters.
"""
function build_model(name::String, n::Int; pars...)
    return EcologicalModel(name, n, NamedTuple{keys(pars)}(values(pars)))
end

"""
Wrapper that samples A, r from a given the model, and the model parameters
"""
function sample_pars(m::EcologicalModel, rng::Any=MersenneTwister(1))
    n = m.n
    if m.name == "normal_f32"
        A = sampler_normal_f32(n, rng; mu=m.pars.mu, sigma=m.pars.sigma, rho=m.pars.rho)
        r = ones(Float32, n)
    else
        error("Unknown model name: $(model.name)")
    end
    return (A, r)
end

"""
Check feasibility of the system Ax = r with x > 0 using Float32 arithmetic.
Uses a two-step approach: first solves without pivoting (fast, unstable), 
then with pivoting (slow, stable) if feasible.
Returns a tuple (is_feasible::Bool, x_eq::Vector{Float32}).
"""
function is_feasible_fast(A::Matrix{Float32}, r::Vector{Float32}, tol::Float32=1f-8)
    # Solve once with try-catch for singular matrices
    x_eq = try 
        solve(LinearProblem(-A, r), RFLUFactorization()).u
    catch
        return false, zeros(Float32, size(A, 1))
    end

    if all(x_eq .> 0.f0)
        return true, x_eq
    else
        return false, x_eq
    end
end

# # Test the functions
# function main()

#     model = build_model("normal_f32", 3; mu=0.0f0, sigma=1.0f0, rho=0.0f0)
#     A, r = sample_pars(model; seed=1)
#     println("A = ", A)
#     println("r = ", r)

#     feasible, x_eq = is_feasible_fast(A, r)
#     println("Feasible: ", feasible)
#     println("Solution: ", x_eq)
# end

# main()
