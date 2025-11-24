include("src/feasibility.jl")
include("src/necessary_conditions.jl")
include("src/boundary_equilibria.jl")
include("src/solve_lp.jl")
include("src/stability.jl")
include("src/check_b_matrix.jl")

using DataFrames
# --- tiny timing helper ---
const _NS2S = 1e-9
timed(f) = (local t=time_ns(); local v=f(); (v, (time_ns()-t)*_NS2S))

mutable struct StepTimers
    t_sample::Float64
    t_feasible::Float64
    t_jacobian::Float64
    t_necessary::Float64
    t_boundary::Float64
    t_buildC::Float64
    t_LP::Float64
    t_stable::Float64
end
StepTimers() = StepTimers(0.,0.,0.,0.,0.,0.,0.,0.)


"""
    run_grid_times(n_vals, sigma_vals; K, mu=-1.0, rho=0.0, ϵ=1e-9, tol=1e-7,
                   trials_per_success=2_000_000, progress=true)

Run the same search as `run_grid_simple`, but **do not** plot.  
Returns a pair `(df_min, df_times)`:

- `df_times`: one row per `(n, σ)` cell with seconds spent in each step, plus totals:
  `:n, :sigma, :attempts, :found, :t_sample, :t_feasible, :t_jacobian, :t_necessary,
   :t_boundary, :t_buildC, :t_LP, :t_stable, :t_total`.
"""
function run_grid_times(n_vals, sigma_vals;
                        K::Int,
                        mu::Float32 = -1.0f0,
                        rho::Float32 = 0.0f0,
                        ϵ::Float32 = 1f-9,
                        tol::Float32 = 1f-7,
                        trials_per_success::Any = Inf,
                        progress::Bool = true)

    df_times = DataFrame(n=Int[], sigma=Float32[], attempts=Int[], found=Int[],
                         t_sample=Float64[], t_feasible=Float64[], t_jacobian=Float64[],
                         t_necessary=Float64[], t_boundary=Float64[], t_buildC=Float64[],
                         t_LP=Float64[], t_stable=Float64[], t_total=Float64[])

    for n in n_vals
        onevec = ones(Float32, n)
        for σ in sigma_vals
            found    = 0
            attempts = 0
            attempts_limit = K * trials_per_success
            T = StepTimers()

            while found < K && attempts < attempts_limit
                attempts += 1
                seed = 1_000_000*n + 10_000*round(Int, 100*Float64(σ)) + attempts

                # 1) sample
                (A, dt) = timed(() -> sampler_normal_f32(n, seed; mu=mu, sigma=σ, rho=rho))
                T.t_sample += dt

                # 2) feasible
                ((feas, xstar), dt) = timed(() -> is_feasible_fast(A, onevec))
                T.t_feasible += dt
                feas || continue

                # 3) jacobian
                (J, dt) = timed(() -> build_jacobian_at_equilibrium(A, xstar))
                T.t_jacobian += dt

                # 4) necessary conditions
                (nec_ok, dt) = timed(() -> necessary_conditions_persistence(A, J))
                T.t_necessary += dt
                nec_ok || continue

                # 5) boundary enumeration (+ bflag, early-exit on necessary fail inside)
                (out, dt) = timed(() -> boundary_equilibria(A, onevec; tol=tol))
                T.t_boundary += dt
                isempty(out.lhs) && continue

                # 6) build C
                (C, dt) = timed(() -> build_C(out.lhs, out.rhs))
                T.t_buildC += dt

                # 7) LP
                (res, dt) = timed(() -> find_p_maxmargin(C; ϵ=ϵ))
                T.t_LP += dt
                (res === nothing || res.δ <= 0) && continue

                # 8) stability
                (st, dt) = timed(() -> is_stable(J))
                T.t_stable += dt

                found += 1
            end

            t_total = T.t_sample + T.t_feasible + T.t_jacobian + T.t_necessary +
                      T.t_boundary + T.t_buildC + T.t_LP + T.t_stable

            push!(df_times, (n=n, sigma=Float64(σ), attempts=attempts, found=found,
                             t_sample=T.t_sample, t_feasible=T.t_feasible, t_jacobian=T.t_jacobian,
                             t_necessary=T.t_necessary, t_boundary=T.t_boundary, t_buildC=T.t_buildC,
                             t_LP=T.t_LP, t_stable=T.t_stable, t_total=t_total))

            # progress && println((found == K ?
            #     "✓" : "×"), " n=$n, σ=$(Float64(σ))  found=$found / K=$K  attempts=$attempts / $attempts_limit  total_time=$(round(t_total,digits=3))s")
        end
    end

    return df_times
end

#df_times = run_grid_times(15, 0.01f0:0.2f0:2.0f0; K=1)
df_times = run_grid_times(17, 2.0f0; K=1)
