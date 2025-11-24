using Random
using Printf
using JLD2
using Arrow

include("src/feasibility.jl")
include("src/necessary_conditions.jl")
include("src/boundary_equilibria.jl")
include("src/solve_lp.jl")
include("src/stability.jl")
include("src/check_b_matrix.jl")

function save_model_batch(model_name::AbstractString, model_dim::Int,
                          pars::NamedTuple, base_seed::Int,
                          successes::Vector; rootdir::AbstractString = "results",
                          format::Symbol = :jld2)
    isempty(successes) && return nothing

    param_str = join(("$(k)=$(v)" for (k, v) in pairs(pars)), "_")
    dirpath = joinpath(rootdir, model_name, string(model_dim))
    isdir(dirpath) || mkpath(dirpath)

    if format == :jld2
        filename = @sprintf("%s_dim%d_%s_seed%d_nsuccess%d.jld2",
                            model_name, model_dim, param_str, base_seed, length(successes))
        fullpath = joinpath(dirpath, filename)

        As   = [s.A for s in successes]
        rs   = [s.r for s in successes]
        meta_attempts = [s.attempt for s in successes]
        meta_persistence = [s.persistence_type for s in successes]
        meta_stable = [s.stable for s in successes]
        meta_bflag  = [s.bflag for s in successes]

        jldsave(fullpath;
                model_name = model_name,
                model_dim = model_dim,
                params = pars,
                base_seed = base_seed,
                success_count = length(successes),
                attempts = meta_attempts,
                persistence_types = meta_persistence,
                stable_flags = meta_stable,
                bflags = meta_bflag,
                As = As,
                rs = rs)
        return fullpath
    elseif format == :arrow
        filename = @sprintf("%s_dim%d_%s_seed%d_nsuccess%d.arrow",
                            model_name, model_dim, param_str, base_seed, length(successes))
        fullpath = joinpath(dirpath, filename)
        # Flatten arrays into columns; store A,r as serialized blobs for simplicity
        tbl = (; model_name = fill(model_name, length(successes)),
               model_dim = fill(model_dim, length(successes)),
               params = fill(string(pars), length(successes)),
               base_seed = fill(base_seed, length(successes)),
               attempt = [s.attempt for s in successes],
               persistence_type = [s.persistence_type for s in successes],
               stable = [s.stable for s in successes],
               bflag = [s.bflag for s in successes],
               A = [Vector{Float32}(vec(s.A)) for s in successes],
               r = [Vector{Float32}(s.r) for s in successes],
               n = fill(model_dim, length(successes)))
        Arrow.write(fullpath, tbl)
        return fullpath
    else
        error("Unknown format: $format. Use :jld2 or :arrow")
    end
end

function try_k_model_pars(m::EcologicalModel, k::Int;
                          base_seed::Int = 1,
                          tol::Float32 = 1f-7,
                          save_format::Symbol = :jld2)

    rng = MersenneTwister(base_seed)
    n_saved = 0
    successes = NamedTuple[]

    for attempts in 1:k
        # 1) sample parameters (A, r) in Float32
        A, r = sample_pars(m, rng)

        # 2) feasibility
        feas, xstar = is_feasible_fast(A, r)
        feas || continue

        # 3) Jacobian at equilibrium
        J = build_jacobian_at_equilibrium(A, xstar)

        # 4) necessary conditions (cheap screens)
        nec_ok = necessary_conditions_persistence(A, J)
        nec_ok || continue

        # 5) boundary enumeration (+ bflag, early-exit on necessary fail inside)
        onevec = ones(eltype(A), size(A, 1))
        out = boundary_equilibria(A, onevec; tol = tol)
        isempty(out.lhs) && continue

        # 6) persistence type from boundary equilibria
        persistence_type = determine_persistence_type(out.lhs, out.rhs; ϵ=1e-5)
        persistence_type == "unknown" && continue

        # 7) stability from Jacobian
        st = is_stable(J)

        # 8) b-flag: always ±1 (no `nothing` by the time we use it)
        bflag = out.bflag === nothing ? check_b_matrix(A) : out.bflag
        @assert bflag == 1 || bflag == -1

        # 9) build record and save
        res = (
            model_name       = m.name,   # String
            model_dim        = m.n,      # Int
            pars             = m.pars,   # NamedTuple
            base_seed        = base_seed,
            attempt          = attempts,
            persistence_type = persistence_type,
            stable           = st,
            bflag            = bflag,
            A                = A,        # Matrix{Float32}
            r                = r,        # Vector{Float32}
        )

        push!(successes, res)
        n_saved += 1
    end

    save_model_batch(m.name, m.n, m.pars, base_seed, successes; format = save_format)
    return n_saved
end

function main()
    # parameters
    n         = 4
    mu        = 0.0f0
    sigma     = 3.5f0
    rho       = 0.0f0
    base_seed = 1
    k         = 30_000

    # build model
    model = build_model("normal_f32", n; mu = mu, sigma = sigma, rho = rho)

    # run search + saving
    n_saved = try_k_model_pars(model, k; base_seed = base_seed, tol = 1f-7, save_format = :arrow)
    @info "Completed $k attempts for model=$(model.name), saved $n_saved systems."
end

main()
