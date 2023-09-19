using LinearSolve
using NPZ
using Test

includet("dnstools.jl")
includet("simulations.jl")

gpu_id = 0
Random.seed!(2)
reltol = 1e-10

# Load test image and ground truth tau values
fpath = "tau_test_data.npz"
test_data = npzread(fpath)
img = test_data["im"]
tau_gt = test_data["tau"]

# Get the tortuosity factor for each axis
tau = Dict()
for axis in [:x, :y, :z]
    prob = tortuosity_fdm(img, axis=axis)
    sol = solve(prob, KrylovJL_CG(), verbose=false, reltol=reltol)
    c = vec_to_field(sol.u, img)
    tau[axis] = compute_tortuosity_factor(c, axis)
end

# Test against ground truth (PoreSpy's tortuosity_fd)
@test tau[:x] ≈ tau_gt[1]
@test tau[:y] ≈ tau_gt[2]
@test tau[:z] ≈ tau_gt[3]
