using JLD2
using LinearSolve
using Statistics
using Test
using Tortuosity
using Tortuosity: TortuositySimulation, tortuosity, vec_to_grid

# ---------------------------------------- #
# Generate test data
# ---------------------------------------- #

# using PythonCall
# op = pyimport("openpnm")
# ps = pyimport("porespy")

# img = ps.generators.blobs(; shape=(32, 42, 52), porosity=0.7, blobiness=[0.5, 1, 2])
# img = Imaginator.trim_nonpercolating_paths(img, :x)
# img = Imaginator.trim_nonpercolating_paths(img, :y)
# img = Imaginator.trim_nonpercolating_paths(img, :z)

# pardiso = op.solvers.PardisoSpsolve()
# res = Dict(
#     axis => ps.simulations.tortuosity_fd(Py(img).to_numpy(); axis=i - 1, solver=pardiso) for
#     (i, axis) in enumerate([:x, :y, :z])
# )

# tau = Dict(axis => pyconvert(Float64, res[axis].tortuosity) for axis in [:x, :y, :z])
# c̄ = Dict(
#     axis => mean(pyconvert(Array{Float64}, res[axis].concentration)[img]) for
#     axis in [:x, :y, :z]
# )
# jldsave("blobs.jld2"; img=img, tau=tau, c̄=c̄)

# ---------------------------------------- #

# Set up fixtures
fpath = joinpath(@__DIR__, "blobs.jld2")
test_data = load(fpath)
img = test_data["img"]
tau_gt = test_data["tau"]
c̄_gt = test_data["c̄"]
ε = phase_fraction(img, 1)
# Build diffusivity matrix
D = fill(NaN, size(img))
D[img] .= 1.0               # Fluid phase
D[.!img] .= 1e-4            # Solid phase

@testset "Blobs 3D, $(ax)-axis" for ax in (:x, :y, :z)
    sim = TortuositySimulation(img; axis=ax, gpu=false)
    sol = solve(sim.prob, KrylovJL_CG(); reltol=1e-6)
    c̄ = mean(sol.u)
    @test c̄ ≈ c̄_gt[ax] atol = 1e-4
    c_grid = vec_to_grid(sol.u, img)
    tau = tortuosity(c_grid, ax)
    @test tau ≈ tau_gt[ax] atol = 1e-4
end

@testset "Blobs 3D, variable diffusivity, $(ax)-axis" for ax in (:x, :y, :z)
    domain = ones(Bool, size(img))
    sim = TortuositySimulation(domain; axis=ax, gpu=false, D=D)
    sol = solve(sim.prob, KrylovJL_CG(); reltol=1e-6)
    c̄ = mean(sol.u[img[:]])
    @test c̄ ≈ c̄_gt[ax] atol = 1e-2
    c_grid = vec_to_grid(sol.u, domain)
    tau = tortuosity(c_grid, ax; eps=ε, D=D)
    @test tau ≈ tau_gt[ax] atol = 1e-2
end
