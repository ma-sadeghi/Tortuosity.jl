using LinearSolve
using NPZ
using Test
using Tortuosity

# Set up fixtures
img = Imaginator.blobs(shape=(32, 32, 32), porosity=0.65, blobiness=0.5, seed=2)
ε = phase_fraction(img, 1)
# Build diffusivity matrix
D = fill(NaN, size(img))
D[img] .= 1.0               # Fluid phase
D[.!img] .= 1e-4            # Solid phase
# Ground truth values for blobs
tau_gt = Dict(:x => 1.637568, :y => 1.501668, :z => 1.564219)
c̄_gt = Dict(:x => 0.447173, :y => 0.509658, :z => 0.461578)

@testset "Blobs 3D, $(ax)-axis" for ax in (:x, :y, :z)
    sim = TortuositySimulation(img, axis=ax, gpu=false)
    sol = solve(sim.prob, KrylovJL_CG(), reltol=1e-6)
    c̄ = mean(sol.u)
    @test c̄ ≈ c̄_gt[ax] atol=1e-4
    c_grid = vec_to_field(sol.u, img)
    tau = tortuosity(c_grid, ax)
    @test tau ≈ tau_gt[ax] atol=1e-4
end

@testset "Blobs 3D, variable diffusivity, $(ax)-axis" for ax in (:x, :y, :z)
    domain = ones(Bool, size(img))
    sim = TortuositySimulation(domain, axis=ax, gpu=false, D=D)
    sol = solve(sim.prob, KrylovJL_CG(), reltol=1e-6)
    c̄ = mean(sol.u[img[:]])
    @test c̄ ≈ c̄_gt[ax] atol=1e-2
    c_grid = vec_to_field(sol.u, domain)
    tau = tortuosity(c_grid, ax, eps=ε, D=D)
    @test tau ≈ tau_gt[ax] atol=1e-2
end
