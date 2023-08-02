# %% Imports
using SparseArrays
using LinearSolve
using Statistics
using CUDA
using CUDA.CUSPARSE
include("utils.jl")

# %% Initialize and set paths
set_plotsjl_defaults()
# path_matrix = "/home/amin/Code/Gore/matrixlib"
path_matrix = "/mnt/optane/Amin/matrixlib"
fname = "linsys_100x100x100_i32f64_csc.h5"
sparse_fmt = occursin("csc", fname) ? "csc" : "coo"
@info "Filename: $(fname)"

# %% Read the matrix ingredients and the RHS to build Ax = b
@info "Reading the matrix and the right-hand side"
fpath = joinpath(path_matrix, fname)
(spmat_args..., shape), rhs, template = read_linear_sys(fpath, sparse_fmt=sparse_fmt);

gpu = length(rhs) >= 100_000 ? true : false
@info "Size of the matrix: $(shape)"
@info "GPU is enabled: $gpu"

# %% Build the sparse matrix (COO/CSC)
if gpu
    device!(1)
    if sparse_fmt == "csc"
        A_gpu = CuSparseMatrixCSC{Float32, Int32}(map(cu, spmat_args)..., shape)
        # A_gpu = CuSparseMatrixCOO{Float32}(SparseMatrixCSC(shape..., spmat_args...))
    elseif sparse_fmt == "coo"
        A_gpu = CuSparseMatrixCOO{Float32, Int32}(map(cu, spmat_args)..., shape)
        A_gpu = CuSparseMatrixCSC(A_gpu)
    end
    b_gpu = CuArray{Float32}(rhs)
else
    # COO is not supported on CPU
    if sparse_fmt != "csc"; error("CPU only supports CSC format"); end
    A_cpu = SparseMatrixCSC(shape..., spmat_args...)
    b_cpu = rhs
end
# Create device-agnostic aliases for A and b
A = gpu ? A_gpu : A_cpu
b = gpu ? b_gpu : b_cpu

# %% Solve Ax = b using an iterative solver
@info "Solving the system using KrylovJL_CG"
rtol = eltype(b)(1e-5)
prob = LinearProblem(A, b)
@time sol = solve(prob, KrylovJL_CG(), verbose=true, reltol=rtol, maxiters=100)
@info "Average concentration: $(mean(sol.u))"

# %% Visualize the solution (2D slice)
c = zeros(size(template)) * NaN
c[template] = Array(sol.u)

display(heatmap(c[1, :, :], aspect_ratio=:equal, color=:viridis, clim=(0, 1)))


nothing
