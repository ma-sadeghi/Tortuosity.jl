# Tastefully written by Harry Kim
# Date: June 1st, 2025
# 2B URA Project: Porous Materials Transitive Equation solver for Deff 
# For Professor Jeff Gostick 


#!!!! IMPORTANT !!!!
# This program uses Threads, so please allow your program to use more threads.
# (Thank you ece 252)

# I developed on Windows. To do so, open up cmd, "set JULIA_NUM_THREADS=4", and restart Julia. 
# Alternatively, in VScode, open up the Command Pallette (command/ctrl + shift + p),
# Open up "Preferences: Open Settings (UI), search julia numThreads, change the value as desired. 

# Currently, I'm using 12 threads on my Desktop, and 4 on my Laptop. 
# There must be better ways to make this faster... 


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################



# TO DO LIST # 
# Find Tortuosity <-- Seperate the code so it looks more like Tortuosity.jl 
# Fit the analytical solution to the colormap at the second last step in Github 

# Turn Global Variables into input using a simple GUI?


# - Start using GPU 


#To do for Thurs:
# Fix Masked concentration 0 being moTre attractive logic
# Fix D_Eff profile error (Check D_Eff Profile Masking Issue on GPT for notes)
# Concentration map (img 1) for soem time steps <-- 

using OrdinaryDiffEq
using DifferentialEquations
using SparseArrays
using LinearAlgebra
using Plots
using ColorSchemes
using Statistics
using Random
using KrylovKit
using Base.Threads
using LsqFit
using Tortuosity

# --- GPU Packages ---
# Make sure you have the correct package for your hardware.
# Add with Pkg.add("CUDA") or Pkg.add("AMDGPU")
try
    using CUDA
    using CUDA.CUSPARSE
    println("NVIDIA CUDA.jl package loaded successfully")
catch
    println("Could not load CUDA.jl. NVIDIA GPU backend will not be available")
end


#NOTE: Linux required for this. 
try
    using AMDGPU
    using AMDGPU.ROCSPARSE
    println("AMD AMDGPU.jl package loaded successfully.")
catch
    println("Could not load AMDGPU.jl. AMD GPU backend will not be available.")
end

println("hello world")

##### CONFIGURATION & GLOBAL VARIABLES #####

# CHOOSE YOUR BACKEND
# Options: :nvidia, :amd, :cpu
const BACKEND = :cpu

# Grid Settings
const N = 40; # Number of grid points in x + y direction.
const L = 0.01 # domain length in meters (1cm)
const dx = L / N # grid spacing in meters
const D = 2.09488e-5 # Bulk diffusivity of oxygen in air (m^2/s)

# Time settings
const tspan = (0.0, 5.0) # simulates 0 to 5 seconds

# Boundary conditions
const C_left = 1.0
const C_right = 0.0

# Porous media geometry
const sphere_radius = 3
const num_spheres = 5

##### HELPER FUNCTIONS (UNCHANGED) #####
function make_analytical_model(x; terms=100)
    return (t, p) -> begin
        D_eff = p[1]
        @inbounds [analytical_concentration(ti, D_eff, x; terms=terms) for ti in t]
    end
end

function fit_Deff(sim_times::AbstractVector, sim_concs::AbstractVector, x::Float64;
    p0=[1e-5], clip_low=0.05, clip_high=0.9, terms=100)
    sim_concs = (sim_concs .- C_right) ./ (C_left - C_right)
    maxC = maximum(sim_concs)
    idx_start = findfirst(c -> c > clip_low * maxC, sim_concs)
    idx_stop = findfirst(c -> c > clip_high * maxC, sim_concs)
    idx_start = isnothing(idx_start) ? 1 : idx_start
    idx_stop = isnothing(idx_stop) ? length(sim_concs) : idx_stop
    model = (t, p) -> [analytical_concentration(ti, p[1], x; terms=terms) for ti in t]
    fit = curve_fit(model, sim_times[idx_start:idx_stop], sim_concs[idx_start:idx_stop], p0)
    return fit.param[1]
end

function analytical_concentration(t, D_eff, x; terms=100)
    s = 0.0
    for n in 1:terms
        s += (1 / n) * sin(n * pi * x / L) * exp(-n^2 * π^2 * D_eff * t / L^2)
    end
    return (C_left - (C_left * (x / L)) - (2 / pi) * C_left * s)
end

function generate_mask(N)
    mask = ones(Float64, N, N)
    rng = MersenneTwister(1234) # for reproducibility
    for _ in 1:num_spheres
        x_c = rand(rng, sphere_radius+1:N-sphere_radius)
        y_c = rand(rng, sphere_radius+1:N-sphere_radius)
        for i in 1:N, j in 1:N
            if (i - x_c)^2 + (j - y_c)^2 ≤ sphere_radius^2
                mask[i, j] = 0.0
            end
        end
    end
    return mask
end


################################################################################
########################### MATRIX BUILDERS ####################################
################################################################################

# ------------------------- NVIDIA GPU VARIANT ------------------------- #
function build_diffusion_matrix_nvidia(N, dx, D, mask_gpu)
    I = Int32[]
    J = Int32[]
    V = Float32[]
    cpu_mask = Array(mask_gpu) # Copy to CPU for fast iteration

    for i in 1:N, j in 1:N
        if cpu_mask[i, j] == 0.0
            continue
        end
        idx = (j - 1) * N + i
        diagonal_val = 0.0f0
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ni, nj = i + di, j + dj
            if 1 ≤ ni ≤ N && 1 ≤ nj ≤ N && cpu_mask[ni, nj] == 1.0
                nidx = (nj - 1) * N + ni
                push!(I, idx)
                push!(J, nidx)
                push!(V, 1.0f0)
                diagonal_val -= 1.0f0
            end
        end
        if diagonal_val != 0.0f0
            push!(I, idx)
            push!(J, idx)
            push!(V, diagonal_val)
        end
    end

    A_gpu = CuSparseMatrixCSR(sparse(I, J, V, N * N, N * N))
    A_gpu .*= Float32(D / dx^2)
    u0_gpu = CUDA.zeros(Float32, N * N)
    return A_gpu, u0_gpu
end

# ------------------------- AMD GPU VARIANT ------------------------- #
function build_diffusion_matrix_amd(N, dx, D, mask_gpu)
    I = Int32[]
    J = Int32[]
    V = Float32[]
    cpu_mask = Array(mask_gpu) # Copy to CPU for fast iteration

    for i in 1:N, j in 1:N
        if cpu_mask[i, j] == 0.0
            continue
        end
        idx = (j - 1) * N + i
        diagonal_val = 0.0f0
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ni, nj = i + di, j + dj
            if 1 ≤ ni ≤ N && 1 ≤ nj ≤ N && cpu_mask[ni, nj] == 1.0
                nidx = (nj - 1) * N + ni
                push!(I, idx)
                push!(J, nidx)
                push!(V, 1.0f0)
                diagonal_val -= 1.0f0
            end
        end
        if diagonal_val != 0.0f0
            push!(I, idx)
            push!(J, idx)
            push!(V, diagonal_val)
        end
    end

    A_cpu = sparse(I, J, V, N * N, N * N)
    A_gpu = AMDGPU.ROCSPARSE.ROCSPARSEMatrixCSR(A_cpu) # Convert to AMD sparse matrix
    A_gpu .*= Float32(D / dx^2)
    u0_gpu = AMDGPU.zeros(Float32, N * N)
    return A_gpu, u0_gpu
end

# ------------------------- CPU VARIANT ------------------------- #
function build_diffusion_matrix_cpu(N, dx, D, mask)
    N2 = N * N
    A = spzeros(Float64, N2, N2)
    for i in 1:N, j in 1:N
        if mask[i, j] == 0.0
            continue
        end
        idx = (j - 1) * N + i
        A[idx, idx] = 0.0
        for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ni, nj = i + di, j + dj
            if 1 ≤ ni ≤ N && 1 ≤ nj ≤ N && mask[ni, nj] == 1.0
                nidx = (nj - 1) * N + ni
                A[idx, nidx] = 1
                A[idx, idx] -= 1
            end
        end
    end
    A .*= (D / dx^2)
    u0 = zeros(N2)
    return A, u0
end


################################################################################
########################### EQUATION SOLVERS ###################################
################################################################################

# ------------------------- NVIDIA GPU VARIANT ------------------------- #
function transient_equation_nvidia(N, dx, D; mask_gpu)
    A, u0 = build_diffusion_matrix_nvidia(N, dx, D, mask_gpu)
    C_left_32, C_right_32 = Float32(C_left), Float32(C_right)
    masked_indices = findall(iszero, vec(mask_gpu)) # Pre-calculate indices

    function f_gpu!(du, u, p, t)
        u[1:N:end] .= C_left_32
        u[N:N:end] .= C_right_32
        mul!(du, A, u)
        du[1:N:end] .= 0.0f0
        du[N:N:end] .= 0.0f0
        du[masked_indices] .= 0.0f0 # Efficiently apply mask
    end

    prob = ODEProblem(f_gpu!, u0, tspan)
    println("Solving on NVIDIA GPU...")
    sol = solve(prob, KenCarp4(linsolve=KrylovJL_GMRES()); saveat=0.05)
    println("GPU Simulation complete.")
    return sol, sol.t
end

# ------------------------- AMD GPU VARIANT ------------------------- #
function transient_equation_amd(N, dx, D; mask_gpu)
    A, u0 = build_diffusion_matrix_amd(N, dx, D, mask_gpu)
    C_left_32, C_right_32 = Float32(C_left), Float32(C_right)
    masked_indices = findall(iszero, vec(mask_gpu)) # Pre-calculate indices

    function f_gpu!(du, u, p, t)
        u[1:N:end] .= C_left_32
        u[N:N:end] .= C_right_32
        mul!(du, A, u)
        du[1:N:end] .= 0.0f0
        du[N:N:end] .= 0.0f0
        du[masked_indices] .= 0.0f0 # Efficiently apply mask
    end

    prob = ODEProblem(f_gpu!, u0, tspan)
    println("Solving on AMD GPU...")
    println("Note: ROCm must be correctly installed on your Linux system.")
    sol = solve(prob, KenCarp4(linsolve=KrylovJL_GMRES()); saveat=0.05)
    println("GPU Simulation complete.")
    return sol, sol.t
end


# ------------------------- CPU VARIANT ------------------------- #
function transient_equation_cpu(N, dx, D; mask)
    A, u0 = build_diffusion_matrix_cpu(N, dx, D, mask)

    function f_cpu!(du, u, p, t)
        u[1:N:end] .= C_left
        u[N:N:end] .= C_right
        mul!(du, A, u)
        du[1:N:end] .= 0.0
        du[N:N:end] .= 0.0
        for i in 1:N, j in 1:N
            if mask[i, j] == 0.0
                idx = (j - 1) * N + i
                du[idx] = 0.0
            end
        end
    end

    prob = ODEProblem(f_cpu!, u0, tspan)
    println("Solving on CPU using $(nthreads()) threads...")
    sol = solve(prob, KenCarp4(linsolve=KrylovJL_GMRES()); saveat=0.05)
    println("CPU Simulation complete.")
    return sol, sol.t
end


################################################################################
########################### POST-PROCESSING ####################################
################################################################################
# These functions now handle both CPU and GPU solution objects.

function fit_multiple_virtual_pores(sol, N, dx, L, sim_times, C_left, C_right)
    println("Fitting virtual pores...")
    x_positions = [0.25 * L, 0.5 * L, 0.75 * L]
    col_indices = [Int(round(x / dx)) for x in x_positions]
    colors = [:cyan, :green, :blue]
    markers = [:star, :utriangle, :cross]
    labels = ["0.25L", "0.5L", "0.75L"]
    D_eff_results = Vector{Union{Float64,Missing}}(undef, length(col_indices))
    p = plot(title="Transient Diffusion Fit at Virtual Pores", xlabel="Time [s]", ylabel="Concentration")

    # This line is key: copy solution from GPU to CPU if necessary
    sol_u_cpu = [Array(u) for u in sol.u]

    Threads.@threads for thread_i in 1:length(col_indices)
        col = col_indices[thread_i]
        try
            col_idxs = [(j - 1) * N + col for j in 1:N]
            sim_concs = [mean(u[col_idxs]) for u in sol_u_cpu] # Use CPU data
            D_fit = fit_Deff(sim_times, sim_concs, col * dx)
            D_eff_results[thread_i] = D_fit
        catch e
            @warn "Fitting failed at col = $col" exception = (e, catch_backtrace())
            D_eff_results[thread_i] = missing
        end
    end

    for (i, col) in enumerate(col_indices)
        if ismissing(D_eff_results[i])
            continue
        end
        col_idxs = [(j - 1) * N + col for j in 1:N]
        sim_concs = [mean(u[col_idxs]) for u in sol_u_cpu] # Use CPU data
        sim_concs_norm = (sim_concs .- C_right) ./ (C_left - C_right)
        D_fit = D_eff_results[i]
        println("x = ", labels[i], ", Recovered D_eff ≈ ", D_fit)
        model = make_analytical_model(col * dx)
        plot!(p, sim_times, sim_concs_norm, label="x=$(labels[i]) sim", lw=2, marker=markers[i], color=colors[i])
        plot!(p, sim_times, model(sim_times, [D_fit]), label="x=$(labels[i]) fit", lw=2, linestyle=:dash, color=colors[i])
    end
    display(p)
end

function solve_Deff(sol, N, dx, L, sim_times, mask_cpu)
    println("Fitting all virtual pores...")
    d_eff_map = Matrix{Union{Float64,Missing}}(missing, N, N)

    # This line is key: copy solution from GPU to CPU if necessary
    sol_u_cpu = [Array(u) for u in sol.u]
    U = reshape(hcat(sol_u_cpu...), N, N, :) # U is now a CPU array

    Threads.@threads for j in 1:N
        for i in 2:N-1
            if mask_cpu[i, j] == 0.0 || all(==(0.0), U[i, j, :])
                continue
            end
            sim_concs = vec(U[i, j, :])
            try
                d_eff_map[i, j] = fit_Deff(sim_times, sim_concs, i * dx)
            catch
                d_eff_map[i, j] = missing
            end
        end
    end

    d_eff_profile = [mean(skipmissing(d_eff_map[i, :])) for i in 2:N-1 if any(!ismissing, d_eff_map[i, :])]
    x_arr = [i * dx for i in 2:N-1 if any(!ismissing, d_eff_map[i, :])]

    p = plot(x_arr, d_eff_profile, seriestype=:scatter, label="Mean D_eff per column",
        xlabel="x [m]", ylabel="D_eff", title="D_eff Profile vs X")
    display(p)
    println("Deff profile is ", d_eff_profile)
    return d_eff_profile
end

function calculate_porosity(mask_cpu)
    porosity = sum(mask_cpu) / length(mask_cpu)
    println("Calculated Porosity: ", porosity)
    return porosity
end

function calculate_tortuosity(porosity, D_Eff)
    tort = porosity * D / D_Eff
    println("Calculated Tortuosity (from D_eff fit): ", tort)
    return tort
end

function compute_tortuosity(mask_cpu, sol)
    println("Calculating tortuosity directly from steady-state concentration field...")
    # This line is key: copy solution from GPU to CPU if necessary
    steady_state = reshape(Array(sol.u[end]), N, N)
    steady_state[mask_cpu.==0.0] .= NaN
    τ = tortuosity(steady_state; axis=:x)
    println("Tortuosity from tortuosity.jl is ≈ ", τ)
    return τ
end

function visualize_final_concentration(sol)
    final_C_cpu = reshape(Array(sol[end]), N, N) # Copy to CPU
    final_C_norm = (final_C_cpu .- C_right) ./ (C_left - C_right)
    p = heatmap(final_C_norm',
        title="Final Concentration ($(BACKEND) backend)",
        yflip=true, colorbar=true, c=reverse(ColorSchemes.rainbow.colors))
    display(p)
end

################################################################################
########################### MAIN EXECUTION #####################################
################################################################################


function main()

    # 1. Generate the geometry on the CPU
    mask_cpu = generate_mask(N)

    # 2. Declare variables for the solution
    local sol, sim_times

    # 3. Run the simulation using the selected backend
    if BACKEND === :nvidia
        mask_gpu = CuArray(mask_cpu)
        sol, sim_times = transient_equation_nvidia(N, dx, D; mask_gpu=mask_gpu)
    elseif BACKEND === :amd
        mask_gpu = ROCArray(mask_cpu)
        sol, sim_times = transient_equation_amd(N, dx, D; mask_gpu=mask_gpu)
    else # Default to CPU
        sol, sim_times = transient_equation_cpu(N, dx, D; mask=mask_cpu)
    end

    # 4. Post-processing and analysis
    # These functions are now safe to call with either a GPU or CPU solution object.
    visualize_final_concentration(sol)
    fit_multiple_virtual_pores(sol, N, dx, L, sim_times, C_left, C_right)
    D_Eff_array = solve_Deff(sol, N, dx, L, sim_times, mask_cpu)

    if !isempty(D_Eff_array)
        D_Eff = D_Eff_array[end]
        c_porosity = calculate_porosity(mask_cpu)
        c_tortuosity = calculate_tortuosity(c_porosity, D_Eff)
        tort_direct = compute_tortuosity(mask_cpu, sol)
    else
        println("Could not determine D_eff profile. Skipping final calculations.")
    end
end


main()