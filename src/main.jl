# %% Imports
using SparseArrays
using LinearSolve
using Statistics
using NaNStatistics
using CUDA
using CUDA.CUSPARSE
includet("utils.jl")

# %% Initialize and set paths
set_plotsjl_defaults()
path_matrix = "/mnt/optane/Amin/matrixlib/sample_B/chunked"
path_export = "./results/sample_B/chunked"
fname = "sample_B6_v0_eps=0.791_large_s0.h5"
fpath = joinpath(path_matrix, fname)
sparse_fmt = "csc"
gpu_id = 1

# Parse CLI parameter(s)
parsed = parse_args(join(ARGS, " "))
if length(parsed) > 0
    fpath = parsed["fpath"]
    fname = basename(fpath)
    gpu_id = parse(Int, parsed["gpu_id"])
    path_export = parsed["path_export"]
end
@info "Filepath: $(fpath)"
@info "GPU ID: $(gpu_id)"
@info "Export path: $(path_export)"

# %% Read the matrix ingredients and the RHS to build Ax = b
@info "Reading the matrix and the right-hand side"
(spmat_args..., shape), rhs, template = read_linear_sys(fpath, sparse_fmt=sparse_fmt);

gpu = length(rhs) >= 100_000 ? true : false
@info "Size of the matrix: $(shape)"
@info "GPU is enabled: $gpu"

# %% Build the sparse matrix (COO/CSC)
if gpu
    device!(gpu_id)
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
@time sol = solve(prob, KrylovJL_CG(), verbose=true, reltol=rtol)
@info "Average concentration: $(mean(sol.u))"

# %% Visualize the solution (2D slice)
c = zeros(size(template)) * NaN
c[template] = Array(sol.u)

if isinteractive()
    display(heatmap(c[:, :, 100], aspect_ratio=:equal, color=:viridis, clim=(0, 1)))
end

tau = calc_tortuosity(c, template)
ff = calc_formation_factor(c, template)
eps = sum(template) / length(template)
@info "Tortuosity factor: $tau"
@info "Formation factor: $ff"
@info "Porosity: $eps"

# Export results
fname = replace(fname, "sample" => "results")
h5open(joinpath(path_export, fname), "w") do fid
    # fid["c"] = c
    # fid["template"] = UInt8.(template)
    fid["tau"] = tau
    fid["ff"] = ff
    fid["eps"] = eps
    # Only export the 100th slice to save space
    fid["c(x=100)"] = c[:, :, 100]
end


nothing
