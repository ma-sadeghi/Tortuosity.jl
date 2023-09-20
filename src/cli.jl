# %% Imports
using CUDA
using CUDA.CUSPARSE
using HDF5
using LinearSolve
using NPZ
using Printf

includet("dnstools.jl")
includet("simulations.jl")
includet("utils.jl")

# %% Initialize and set paths
if isinteractive()
    gpu_id = 0
    fpath = "./images/testimage.npz"
    path_export = "./results"
    args = ["--fpath=$fpath", "--gpu_id=$gpu_id", "--path_export=$path_export"]
else
    args = ARGS
end

# %% Parse command line arguments
args_dict = args_to_dict(args)
fpath, path_export, gpu_id = format_args_dict(args_dict)

@info "Filepath: $(fpath)"
@info "Export path: $(path_export)"
@info "GPU ID: $(gpu_id)"

# %% Read the matrix ingredients and the RHS to build Ax = b
@info "Reading the voxel image"
img = npzread(fpath)["arr_0"][1:50, 1:50, :]
@info "Size of the image: $(size(img))"

# %% Build Ax = b on CPU/GPU
@info "Setting up τ simulation (assembling Ax = b)"
prob = tortuosity_fdm(img, axis=:z)
gpu = sum(img) >= 100_000 ? true : false
gpu && device!(gpu_id)
@info "Offloading the system to GPU: $gpu"
prob = gpu ? LinearProblem(cu(prob.A), cu(prob.b)) : prob
reltol = eltype(prob.b)(1e-5)

# %% Solve Ax = b using an iterative solver
@info "Solving the system using KrylovJL_CG"
sol = solve(prob, KrylovJL_CG(), verbose=false, reltol=reltol)
@info "Average concentration: $(mean(sol.u))"

# %% Compute the tortuosity factor and visualize the solution
c = vec_to_field(sol.u, img)
tau = compute_tortuosity_factor(c, :z)
ff = compute_formation_factor(c, :z)
eps = sum(img) / length(img)
@info "τ: $(@sprintf("%.3f", tau)), ℱℱ: $(@sprintf("%.3f", ff)), ε = $(@sprintf("%.3f", eps))"

# %% Export results
fname = replace(basename(fpath), "sample" => "results", "npz" => "h5")
fpath_out = joinpath(path_export, fname)
export_to_hdf5(fpath_out, tau=tau, ff=ff, eps=eps, cxz=c[:,25,:])
@info "Results exported to $fpath_out"
