# main.jl


# Tastefully written by Harry Kim
# Date: June 1st, 2025
# 2B URA Project: Porous Materials Transitive Equation solver for Deff 
# For Professor Jeff Gostick 
# Shoutout to Matthews Ma for graciously allowing me to borrow his GPU for this program 


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



# Focus on GPU wrapping 
# SEE if we can change the tolerance for the solver so it's more accurate --> Smaller time steps? 
# If time, find way to choose concentration 



#Tortuosity from any Time. Any (x,y) and time 
# Pick random spots in the domain, and for that specific spot, at each time, what was the concentration
# So like a two col vector 
# (Could be a useful API feature) <--just tell the code which pixel to store to efficient. (Probe spots, maybe like half a dozen)
# (New column for each pixel choice) <-- Concentration ateach of those locations 
# Can do fick's law analysis on each of those points, ideally get the same Deff 

# Main output column is t, c1, c2, c3, c4... 
# Then it can be passed into another function that Fick's laws to each of the columns 

# Or maybe take every pixel of a single column (in heatmap) and average it out
# Or maybe a patch (simulates a probe) 

# Easiest thing: Return Concentration at specified pixels. Higher level wrapper can use it later. 

#Send Github to Prof Gostick 


# --- Package Loading ---
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
using Tortuosity: tortuosity, vec_to_grid
using CUDSS
using LinearSolve
using DiffEqGPU

try
    using CUDA
    using CUDA.CUSPARSE
    println("NVIDIA CUDA.jl package loaded successfully.")
catch
    println("Could not load CUDA.jl. NVIDIA GPU backend will not be available.")
end

try
    using AMDGPU
    using AMDGPU.ROCSPARSE
    println("AMD AMDGPU.jl package loaded successfully.")
catch
    println("Could not load AMDGPU.jl. AMD GPU backend will not be available.")
end

# --- Include Other Project Files ---
include("utils.jl")
include("simulations.jl")
include("analysis.jl")

println("hello world")

# --- Global Configuration ---
const selected_backend = :cpu  # Options: :nvidia, :amd, :cpu
const N = 40
const L = 0.01f0
const differential_x = L / N
const D = 2.09488f-5
const tspan = (0.0f0, 5.0f0)
const C_left = 1.0f0
const C_right = 0.0f0
const sphere_radius = 3
const num_spheres = 5


function main()
    # 1. Generate the geometry
    mask_cpu = generate_mask(N, sphere_radius, num_spheres)

    # 2. Declare variables for the solution
    local sol, sim_times

    # 3. Run the simulation using the selected backend
    if selected_backend === :nvidia
        if !isdefined(Main, :CUDA) || !CUDA.functional()
            @error "NVIDIA backend selected, but CUDA is not functional. Falling back to CPU."
            sol, sim_times = transient_equation_cpu(N, differential_x, D; mask=mask_cpu)
            println("hsidfjslakjgd")

        else
            mask_gpu = CuArray(mask_cpu)
            sol, sim_times = transient_equation_nvidia(N, differential_x, D; mask_gpu=mask_gpu)
            println("hsidfjslakjgd")
        end
    elseif selected_backend === :amd
        if !isdefined(Main, :AMDGPU) || !AMDGPU.functional()
            @error "AMD backend selected, but AMDGPU is not functional. Falling back to CPU."
            sol, sim_times = transient_equation_cpu(N, differential_x, D; mask=mask_cpu)
        else
            mask_gpu = ROCArray(mask_cpu)
            sol, sim_times = transient_equation_amd(N, differential_x, D; mask_gpu=mask_gpu)
            println("hsidfjslakjgd")

        end
    else # Default to CPU
        sol, sim_times = transient_equation_cpu(N, differential_x, D; mask=mask_cpu)
        println("hsidfjslakjgd")

    end

    # 4. Post-processing and analysis
    visualize_final_concentration(sol, C_left, C_right)
    fit_multiple_virtual_pores(sol, N, differential_x, L, sim_times, C_left, C_right)
    D_Eff_array = solve_Deff(sol, N, differential_x, L, sim_times, mask_cpu, C_left, C_right)

    if !isempty(D_Eff_array)
        D_Eff = D_Eff_array[end]
        c_porosity = calculate_porosity(mask_cpu)
        calculate_tortuosity(c_porosity, D, D_Eff)
        compute_tortuosity(mask_cpu, sol)
    else
        println("Could not determine D_eff profile. Skipping final calculations.")
    end
end

# --- Run the script ---
main()


