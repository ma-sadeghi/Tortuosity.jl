# simulation.jl

# Tastefully written by Harry Kim
# Date: June 1st, 2025
# 2B URA Project: Porous Materials Transitive Equation solver for Deff 
# For Professor Jeff Gostick 


## File info: contains core logic for building diffusion matrix, and solving ODEs 
## Based on different backends, can choose between  CPU, GPU (NVidia or AMD)
## NOTE: AMD ROCm (Which supports DifferentialEquations) requires Linux D: 
## Installation guide on Linux: https://www.reddit.com/r/steamdeck_linux/comments/102hzav/guide_how_to_install_rocm_for_gpu_julia/

## AMD part hasn't been tested yet... 

# ------------------------- NVIDIA GPU VARIANT ------------------------- #
function build_diffusion_matrix_nvidia(N, dx, D, mask_gpu)
    I = Int32[]
    J = Int32[]
    V = Float32[]
    cpu_mask = Array(mask_gpu)
    for i = 1:N, j = 1:N
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
    # 1. Scale the values on the CPU *before* creating the GPU matrix
    V .*= Float32(D / dx^2)

    # 2. Now, create the final GPU matrix with the already-scaled values
    A_gpu = CuSparseMatrixCSR(sparse(I, J, V, N * N, N * N))

    u0_gpu = CUDA.zeros(Float32, N * N)
    return A_gpu, u0_gpu
end
#     A_gpu = CuSparseMatrixCSR(sparse(I, J, V, N * N, N * N))
#     A_gpu .*= Float32(D / dx^2)

#     u0_gpu = CUDA.zeros(Float32, N * N)
#     return A_gpu, u0_gpu
# end


#GPU Wants any operations that's passed to solver Vectorized 
# FIt the integration over time instead 
# Use Flux vs Time instead of using time 

function transient_equation_nvidia(N, dx, D; mask_gpu)
    A, u0 = build_diffusion_matrix_nvidia(N, dx, D, mask_gpu)
    C_left_32, C_right_32 = Float32(C_left), Float32(C_right)
    masked_indices = findall(iszero, vec(mask_gpu))
    function f_gpu!(du, u, p, t)
        u[1:N:end] .= C_left_32
        u[N:N:end] .= C_right_32
        mul!(du, A, u)
        du[1:N:end] .= 0.0f0
        du[N:N:end] .= 0.0f0
        du[masked_indices] .= 0.0f0
    end
    func = ODEFunction(f_gpu!; jac_prototype=A, jac=(J, u, p, t) -> nothing)
    prob = ODEProblem(func, u0, tspan)

    println("Solving on NVIDIA GPU...")
    # This solver could be faster... but causing errors with GPU 
    # sol = solve(prob, KenCarp4(linsolve=KrylovJL_GMRES()); saveat=0.05f0)
    # sol = solve(prob, ROCK4(); saveat=0.01f0, abstol=1e-9, reltol=1e-6)
    sol = solve(prob, ROCK4(); saveat=0.05f0)
    println("GPU Simulation complete.")
    return sol, sol.t
end

# ------------------------- AMD GPU VARIANT ------------------------- #
function build_diffusion_matrix_amd(N, dx, D, mask_gpu)
    I = Int32[]
    J = Int32[]
    V = Float32[]
    cpu_mask = Array(mask_gpu)
    for i = 1:N, j = 1:N
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
    A_gpu = AMDGPU.ROCSPARSE.ROCSPARSEMatrixCSR(A_cpu)
    A_gpu .*= Float32(D / dx^2)
    u0_gpu = AMDGPU.zeros(Float32, N * N)
    return A_gpu, u0_gpu
end

function transient_equation_amd(N, dx, D; mask_gpu)
    A, u0 = build_diffusion_matrix_amd(N, dx, D, mask_gpu)
    C_left_32, C_right_32 = Float32(C_left), Float32(C_right)
    masked_indices = findall(iszero, vec(mask_gpu))
    function f_gpu!(du, u, p, t)
        u[1:N:end] .= C_left_32
        u[N:N:end] .= C_right_32
        mul!(du, A, u)
        du[1:N:end] .= 0.0f0
        du[N:N:end] .= 0.0f0
        du[masked_indices] .= 0.0f0
    end
    prob = ODEProblem(f_gpu!, u0, tspan)
    println("Solving on AMD GPU...")
    sol = solve(prob, KenCarp47(linsolve=KrylovJL_GMRES()); saveat=0.005, abstol=1e-9, reltol=1e-6)
    println("GPU Simulation complete.")
    return sol, sol.t
end

# ------------------------- CPU VARIANT ------------------------- #
function build_diffusion_matrix_cpu(N, dx, D, mask)
    N2 = N * N
    A = spzeros(Float64, N2, N2)
    for i = 1:N, j = 1:N
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

function transient_equation_cpu(N, dx, D; mask)
    A, u0 = build_diffusion_matrix_cpu(N, dx, D, mask)
    function f_cpu!(du, u, p, t)
        u[1:N:end] .= C_left
        u[N:N:end] .= C_right
        mul!(du, A, u)
        du[1:N:end] .= 0.0
        du[N:N:end] .= 0.0
        for i = 1:N, j = 1:N
            if mask[i, j] == 0.0
                idx = (j - 1) * N + i
                du[idx] = 0.0
            end
        end
    end
    prob = ODEProblem(f_cpu!, u0, tspan)
    println("Solving on CPU using $(nthreads()) threads...")
    sol = solve(prob, KenCarp4(linsolve=KrylovJL_GMRES()); saveat=0.005)
    # sol = solve(prob, ROCK4(); saveat=0.05f0)


    println("CPU Simulation complete.")
    return sol, sol.t
end