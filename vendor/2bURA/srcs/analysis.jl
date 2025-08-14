# Tastefully written by Harry Kim
# Start Date: June 1st, 2025
# 2B URA Project: Porous Materials Transitive Equation solver for Deff 
# For Professor Jeff Gostick 

## File contains Analytical solvers / Calculations for equations

function analytical_concentration(t, D_eff, x, L, C_left; terms=100)
    s = 0.0
    for n in 1:terms
        s += (1 / n) * sin(n * pi * x / L) * exp(-n^2 * π^2 * D_eff * t / L^2)
    end
    return (C_left - (C_left * (x / L)) - (2 / pi) * C_left * s)
end

function fit_Deff(sim_times, sim_concs, x, L, C_left, C_right; p0=[1e-5], clip_low=0.05, clip_high=0.9, terms=100)
    sim_concs = (sim_concs .- C_right) ./ (C_left - C_right)
    maxC = maximum(sim_concs)
    idx_start = findfirst(c -> c > clip_low * maxC, sim_concs)
    idx_stop = findfirst(c -> c > clip_high * maxC, sim_concs)
    idx_start = isnothing(idx_start) ? 1 : idx_start
    idx_stop = isnothing(idx_stop) ? length(sim_concs) : idx_stop
    model = (t, p) -> [analytical_concentration(ti, p[1], x, L, C_left; terms=terms) for ti in t]
    fit = curve_fit(model, sim_times[idx_start:idx_stop], sim_concs[idx_start:idx_stop], p0)
    return fit.param[1]
end

function fit_multiple_virtual_pores(sol, N, dx, L, sim_times, C_left, C_right)
    println("Fitting virtual pores...")
    x_positions = [0.25 * L, 0.5 * L, 0.75 * L]
    col_indices = [Int(round(x / dx)) for x in x_positions]
    colors = [:cyan, :green, :blue]
    markers = [:star, :utriangle, :cross]
    labels = ["0.25L", "0.5L", "0.75L"]
    D_eff_results = Vector{Union{Float64,Missing}}(undef, length(col_indices))
    p = plot(title="Transient Diffusion Fit at Virtual Pores", xlabel="Time [s]", ylabel="Concentration")
    sol_u_cpu = [Array(u) for u in sol.u]

    Threads.@threads for thread_i in 1:length(col_indices)
        col = col_indices[thread_i]
        try
            col_idxs = [(j - 1) * N + col for j in 1:N]
            sim_concs = [mean(u[col_idxs]) for u in sol_u_cpu]
            D_fit = fit_Deff(sim_times, sim_concs, col * dx, L, C_left, C_right)
            D_eff_results[thread_i] = D_fit
        catch e
            @warn "Fitting failed at col = $col" exception = (e, catch_backtrace())
        end
    end

    for (i, col) in enumerate(col_indices)
        if ismissing(D_eff_results[i])
            continue
        end
        col_idxs = [(j - 1) * N + col for j in 1:N]
        sim_concs = [mean(u[col_idxs]) for u in sol_u_cpu]
        sim_concs_norm = (sim_concs .- C_right) ./ (C_left - C_right)
        D_fit = D_eff_results[i]
        println("x = $(labels[i]), Recovered D_eff ≈ $D_fit")
        model = (t, p) -> [analytical_concentration(ti, p[1], col * dx, L, C_left) for ti in t]
        plot!(p, sim_times, sim_concs_norm, label="x=$(labels[i]) sim", lw=2, marker=markers[i], color=colors[i])
        plot!(p, sim_times, model(sim_times, [D_fit]), label="x=$(labels[i]) fit", lw=2, linestyle=:dash, color=colors[i])
    end
    display(p)
end

function solve_Deff(sol, N, dx, L, sim_times, mask_cpu, C_left, C_right)
    println("Fitting all virtual pores...")
    d_eff_map = Matrix{Union{Float64,Missing}}(missing, N, N)
    sol_u_cpu = [Array(u) for u in sol.u]
    U = reshape(hcat(sol_u_cpu...), N, N, :)

    Threads.@threads for j in 1:N
        for i in 2:N-1
            if mask_cpu[i, j] == 0.0 || all(==(0.0), U[i, j, :])
                continue
            end
            sim_concs = vec(U[i, j, :])
            try
                d_eff_map[i, j] = fit_Deff(sim_times, sim_concs, i * dx, L, C_left, C_right)
            catch
                d_eff_map[i, j] = missing
            end
        end
    end

    d_eff_profile = [mean(skipmissing(d_eff_map[i, :])) for i in 2:N-1 if any(!ismissing, d_eff_map[i, :])]
    x_arr = [i * dx for i in 2:N-1 if any(!ismissing, d_eff_map[i, :])]
    p = plot(x_arr, d_eff_profile, seriestype=:scatter, label="Mean D_eff per column", xlabel="x [m]", ylabel="D_eff", title="D_eff Profile vs X")
    display(p)
    println("Deff profile is ", d_eff_profile)
    return d_eff_profile
end

function calculate_porosity(mask_cpu)
    porosity = sum(mask_cpu) / length(mask_cpu)
    println("Calculated Porosity: ", porosity)
    return porosity
end

function calculate_tortuosity(porosity, D, D_Eff)
    tort = porosity * D / D_Eff
    println("Calculated Tortuosity (from D_eff fit): ", tort)
    return tort
end

function compute_tortuosity(mask_cpu, sol)
    println("Calculating tortuosity directly from steady-state concentration field...")
    steady_state = reshape(Array(sol.u[end]), N, N)
    steady_state[mask_cpu.==0.0] .= NaN
    τ = tortuosity(steady_state; axis=:x)
    println("Tortuosity from tortuosity.jl is ≈ ", τ)
    return τ
end