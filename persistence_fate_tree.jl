using DataFrames, Random, Printf
using DifferentialEquations, CairoMakie

include("sample_persistent_track_logs.jl")

# + assumes `is_stable(J)::Bool` is defined

function sample_K_persistent_per_sigma(; K::Int=10, n::Int=10, mu::Float64=-1.0,
    sigma_vals = 1.0:1.0:3.0, rho::Float64=0.0, ϵ::Float64=1e-9, tol::Float64=1e-7,
    max_cycles_per_cell::Int=1_000_000)

    onevec = ones(Float64, n)

    df_persist = DataFrame(sigma=Float64[], seed=Int[], delta=Float64[],
        p=Vector{Float64}[], A=Matrix{Float64}[], xstar=Vector{Float64}[],
        necessary_screenAx=Union{Bool,Nothing}[], n_boundary_points=Int[],
        sufficient_fail_trials=Int[], stable=Bool[])  # <--- added

    df_necpass = DataFrame(
        sigma=Float64[], seed=Int[], A=Matrix{Float64}[], xstar=Vector{Float64}[],
        n_boundary_points=Int[], necessary_screenAx=Bool[], lp_success=Bool[],
        delta=Union{Float64,Missing}[], p=Union{Vector{Float64},Missing}[],
        sufficient_fails_since_last_success=Int[],
        stable=Union{Bool,Missing}[]  # <--- added (only set when lp_success)
    )

    df_counts  = DataFrame(sigma=Float64[], found=Int[], attempts=Int[],
        infeasible_trials=Int[], hs_fail_trials=Int[],
        necAx_passes=Int[], necAx_fails=Int[], sufficient_fails_total=Int[])

    for σ in sigma_vals
        found = 0; attempts = 0
        infeasible_trials = 0; hs_fail_trials = 0
        necAx_passes = 0; necAx_fails = 0
        sufficient_fails_total = 0
        sufficient_fails_since_last_success = 0

        while found < K && attempts < max_cycles_per_cell
            attempts += 1
            seed = 10_000 * round(Int, 100*σ) + attempts

            # A, r = sample_A_with_r_rank1(n, seed; mu=abs(mu), sigma=σ, rho=rho)
            A = sampler_normal_rank1(n, seed; mu=abs(mu), sigma=σ, rho=rho)
            r = onevec
            feas, xstar = is_feasible(A, r)
            if !feas
                infeasible_trials += 1
                continue
            end

            J = build_jacobian_at_equilibrium(A, xstar)
            if !necessary_conditions_persistence(A, J)
                hs_fail_trials += 1
                continue
            end
            st = is_stable(J)  # <--- stability at xstar

            out = boundary_equilibria(A, r; tol=tol)

            necAx = out.necessary_maxAx
            necAx === true  && (necAx_passes += 1)
            necAx === false && (necAx_fails  += 1)

            isempty(out.lhs) && continue    # only skip LP step, not counting


            if necAx === true
                C   = build_C(out.lhs, out.rhs)
                res = find_p_maxmargin(C; ϵ=ϵ)
                lp_ok = (res !== nothing) && (res.δ > 0)

                push!(df_necpass, (
                    sigma=Float64(σ), seed=seed, A=A, xstar=xstar,
                    n_boundary_points=length(out.lhs), necessary_screenAx=true,
                    lp_success=lp_ok,
                    delta = lp_ok ? res.δ : missing,
                    p     = lp_ok ? res.p  : missing,
                    sufficient_fails_since_last_success = sufficient_fails_since_last_success,
                    stable = lp_ok ? st : missing                 # <--- set when success
                ))

                if lp_ok
                    push!(df_persist, (
                        sigma=Float64(σ), seed=seed, delta=res.δ, p=res.p,
                        A=A, xstar=xstar, necessary_screenAx=true,
                        n_boundary_points=length(out.lhs),
                        sufficient_fail_trials=sufficient_fails_since_last_success,
                        stable=st                                   # <--- added
                    ))
                    found += 1
                    sufficient_fails_since_last_success = 0
                else
                    sufficient_fails_since_last_success += 1
                    sufficient_fails_total += 1
                end
                continue
            end
        end

        push!(df_counts, (sigma=Float64(σ), found=found, attempts=attempts,
            infeasible_trials=infeasible_trials, hs_fail_trials=hs_fail_trials,
            necAx_passes=necAx_passes, necAx_fails=necAx_fails,
            sufficient_fails_total=sufficient_fails_total))
    end

    return df_persist, df_counts, df_necpass
end


using DataFrames, CSV

# params
K = 100
n = 8
mu = -1.0
sigma_vals = 1.0:1.0:3.0
rho = 0.0
ϵ = 1e-9
tol = 1e-7
max_cycles_per_cell = 10_000_000

# assumes: sampler_normal_rank1, is_feasible, build_jacobian_at_equilibrium,
# necessary_conditions_persistence, boundary_equilibria, build_C, find_p_maxmargin,
# and sample_K_persistent_per_sigma are already defined in scope

df_persist, df_counts, df_necpass = sample_K_persistent_per_sigma(; K, n, mu, sigma_vals, rho, ϵ, tol, max_cycles_per_cell)

println("Persistent systems:")
show(df_persist, allrows=true, allcols=true)
println("\nCounts:")
show(df_counts, allrows=true, allcols=true)

# CSV.write("persistent_systems.csv", df_persist)
# CSV.write("persistent_counts.csv", df_counts)

using Printf, DataFrames

"""
Print an ASCII decision tree of fates per σ row from `df_counts`.

Expected columns per row:
: sigma, :attempts, :infeasible_trials, :hs_fail_trials,
:necAx_passes, :necAx_fails, :found, :sufficient_fails_total
"""
function print_fate_trees(df::DataFrame)
    pct(x, tot) = tot == 0 ? "—" : @sprintf("%.1f%%", 100*x/tot)

    for r in eachrow(df)
        σ          = r.sigma
        attempts   = r.attempts
        infeas     = r.infeasible_trials
        feasible   = attempts - infeas

        hs_fail    = r.hs_fail_trials
        hs_pass    = feasible - hs_fail

        nec_pass   = r.necAx_passes
        nec_fail   = r.necAx_fails

        found      = r.found                       # LP successes
        lp_fail    = r.sufficient_fails_total      # total LP failures

        # split LP failures by branch (consistent with your wrapper)
        lp_fail_necpass = max(nec_pass - found, 0)
        lp_fail_necfail = max(nec_fail, 0)

        println("")
        println("σ=$(σ)  (attempts=$(attempts))")
        println("├─ infeasible: $(infeas)  ($(pct(infeas, attempts)))")
        println("└─ feasible:   $(feasible) ($(pct(feasible, attempts)))")
        println("   ├─ HS fail:  $(hs_fail)  ($(pct(hs_fail, feasible)))")
        println("   └─ HS pass:  $(hs_pass)  ($(pct(hs_pass, feasible)))")

        # boundary screen branch
        println("      ├─ necAx fail: $(nec_fail) ($(pct(nec_fail, hs_pass))) → LP fail: $(lp_fail_necfail)")
        println("      └─ necAx pass: $(nec_pass) ($(pct(nec_pass, hs_pass)))")
        println("         ├─ LP success: $(found) ($(pct(found, nec_pass)))")
        println("         └─ LP fail:    $(lp_fail_necpass) ($(pct(lp_fail_necpass, nec_pass)))")

        # sanity (optional): LP accounting equals HS-pass count
        # @assert lp_fail == lp_fail_necpass + lp_fail_necfail == hs_pass - found
        println()
    end
end

# Example:
print_fate_trees(df_counts)
using DataFrames, Random, Printf
using DifferentialEquations, CairoMakie

# Plot all necessary-pass cases, split by LP outcome.
# Save into per-σ subfolders under each LP category.
function plot_necpass_by_LP(df_necpass::DataFrame;
        use_stored_A::Bool = true,
        mu::Float64 = -1.0, rho::Float64 = 0.0,
        tspan::Tuple{Float64,Float64} = (0.0, 50.0), saveat::Float64 = 0.1,
        outdir_fail::AbstractString = "dynamics_unique_nash_no_LP",
        outdir_success::AbstractString = "dynamics_unique_nash_yes_LP",
        epsy::Float64 = 1e-12)

    mkpath(outdir_fail); mkpath(outdir_success)

    # necessary pass + has boundary points
    sel = df_necpass[(df_necpass.necessary_screenAx .== true) .&
                     (df_necpass.n_boundary_points .> 0), :]
    unique_pairs = unique(sel[:, [:sigma, :seed]])

    logged = DataFrame(sigma=Float64[], seed=Int[], lp_success=Bool[],
                       delta=Union{Float64,Missing}[], file=String[])

    for row in eachrow(unique_pairs)
        σ    = Float64(row.sigma)
        seed = Int(row.seed)

        i        = findfirst((sel.sigma .== σ) .& (sel.seed .== seed))
        A_stored = sel.A[i]
        n        = size(A_stored, 1)
        lp_ok    = sel.lp_success[i]
        δ        = sel.delta[i]

        A = use_stored_A ? A_stored :
            sampler_normal_rank1(n, seed; mu=abs(mu), sigma=σ, rho=rho)

        r   = ones(n)
        sol = simulate_glv(A, r; tspan=tspan, saveat=saveat, seed=seed)

        outd  = lp_ok ? outdir_success : outdir_fail
        sdir  = @sprintf("sigma_%.2f", σ)
        sigdir = joinpath(outd, sdir)
        mkpath(sigdir)

        δtxt = (ismissing(δ) || !lp_ok) ? "" : @sprintf("  δ=%.3g", δ)
        fig = Figure(resolution=(1000, 560))
        ax  = Axis(fig[1,1], xlabel="Time", ylabel="Abundance", yscale=log10,
                   title=@sprintf("σ=%.2f  seed=%d  (necAx pass, %s%s)",
                                  σ, seed, lp_ok ? "LP success" : "LP fail", δtxt))

        for j in 1:n
            y = sol[j, :]
            yclip = map(v -> v > epsy ? v : NaN, y)
            lines!(ax, sol.t, yclip)
        end
        hidespines!(ax, :t, :r)

        fname = @sprintf("seed_%d_%s.png", seed, lp_ok ? "yesLP" : "noLP")
        file  = joinpath(sigdir, fname)
        save(file, fig)

        push!(logged, (σ, seed, lp_ok, δ, file))
    end

    return logged
end

# # Example:
# logged = plot_necpass_by_LP(df_necpass; use_stored_A=true,
#     mu=-1.0, rho=0.0, tspan=(0.0,150.0), saveat=0.1,
#     outdir_fail="dynamics_unique_nash_no_LP",
#     outdir_success="dynamics_unique_nash_yes_LP",
#     epsy=1e-12)

function scatter_delta_vs_minabundance_by_sigma(df_necpass::DataFrame;
    use_stored_A::Bool = true, mu::Float64 = -1.0, rho::Float64 = 0.0,
    tspan::Tuple{Float64,Float64} = (0.0, 50.0), saveat::Float64 = 0.1,
    ncols::Int = 3, epsy::Float64 = 1e-12,
    outpath::AbstractString = "scatter_delta_vs_min_abundance_by_sigma.pdf")

    # LP successes w/ necessary pass (stability available)
    mask = (df_necpass.necessary_screenAx .== true) .&
           (df_necpass.lp_success .== true) .&
           (df_necpass.n_boundary_points .> 0) .&
           (.!ismissing.(df_necpass.delta)) .& (df_necpass.delta .> 0)
    sel = df_necpass[mask, :]

    sigmas = sort(unique(sel.sigma))
    perσ = Dict{Float64, DataFrame}()

    for σ in sigmas
        keys = unique(sel[sel.sigma .== σ, [:sigma, :seed]])
        out  = DataFrame(delta=Float64[], min_abundance=Float64[], seed=Int[], stable=Bool[])
        for row in eachrow(keys)
            seed = Int(row.seed)
            i    = findfirst((sel.sigma .== σ) .& (sel.seed .== seed))
            A_stored = sel.A[i]
            n        = size(A_stored, 1)
            δ        = Float64(sel.delta[i])
            st       = Bool(sel.stable[i])  # stored above

            A = use_stored_A ? A_stored :
                sampler_normal_rank1(n, seed; mu=abs(mu), sigma=σ, rho=rho)

            r   = ones(n)
            sol = simulate_glv(A, r; tspan=tspan, saveat=saveat, seed=seed)

            # log-safe min abundance
            m = Inf
            for j in 1:n
                y  = sol[j, :]
                m  = min(m, minimum(@. max(y, epsy)))
            end
            push!(out, (δ, m, seed, st))
        end
        perσ[Float64(σ)] = out
    end

    nσ         = length(sigmas)
    nrows_grid = ceil(Int, nσ / ncols)
    fig        = Figure(resolution=(360*ncols, 280*nrows_grid))

    for (k, σ) in enumerate(sigmas)
        rr = div(k-1, ncols) + 1
        cc = mod(k-1, ncols) + 1
        data = perσ[Float64(σ)]

        ax = Axis(fig[rr, cc],
                  title = @sprintf("σ = %.2f", σ),
                  xlabel = "δ (LP margin)",
                  ylabel = "Min abundance during dynamics",
                  yscale = log10)

        if nrow(data) > 0
            # split by stability for legendable colors
            stab_mask = data.stable .== true
            if any(stab_mask)
                scatter!(ax, data.delta[stab_mask], data.min_abundance[stab_mask];
                         color=:teal, markersize=7, label="stable")
            end
            if any(.!stab_mask)
                scatter!(ax, data.delta[.!stab_mask], data.min_abundance[.!stab_mask];
                         color=:tomato, markersize=7, label="unstable")
            end
            axislegend(ax, position=:rb, framevisible=false)
        end

        hidespines!(ax, :t, :r)
        if nrows_grid > 1 && rr < nrows_grid; ax.xlabel = ""; end
        if ncols > 1 && cc > 1;               ax.ylabel = ""; end
    end

    save(outpath, fig)

    combined = DataFrame(sigma=Float64[], seed=Int[], delta=Float64[],
                         min_abundance=Float64[], stable=Bool[])
    for σ in sigmas
        d = perσ[Float64(σ)]
        for i in 1:nrow(d)
            push!(combined, (Float64(σ), d.seed[i], d.delta[i], d.min_abundance[i], d.stable[i]))
        end
    end
    return combined
end


# # Example:
# df_byσ = scatter_delta_vs_minabundance_by_sigma(df_necpass;
#     use_stored_A=true, mu=-1.0, rho=0.0,
#     tspan=(0.0,50.0), saveat=0.1,
#     ncols=3,
#     outpath="scatter_delta_vs_min_abundance_by_sigma.pdf")
