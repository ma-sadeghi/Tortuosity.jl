using LinearSolve
using Statistics
using Test
using Tortuosity
using Tortuosity: TortuositySimulation, formation_factor, tortuosity, vec_to_grid

# Set up fixtures
open_space = ones(Bool, (32, 32, 32))
straight_channel = ones(Bool, (32, 32, 32))
straight_channel[:, :, 1:16] .= 0

@testset "Open space 3D, $(ax)-axis" for ax in (:x, :y, :z)
    sim = TortuositySimulation(open_space; axis=ax, gpu=false)
    sol = solve(sim.prob, KrylovJL_CG(); reltol=1e-6)
    c̄ = mean(sol.u)
    # Open space has a perfectly linear concentration profile, i.e., c̄ = 0.5
    @test c̄ ≈ 0.5 atol = 1e-4
    c_grid = vec_to_grid(sol.u, open_space)
    tau = tortuosity(c_grid; axis=ax)
    # Open space has no tortuosity, i.e., τ = 1
    @test tau ≈ 1.0 atol = 1e-4
end

@testset "Straight channel (half-open cube)" begin
    sim = TortuositySimulation(straight_channel; axis=:x, gpu=false)
    sol = solve(sim.prob, KrylovJL_CG(); reltol=1e-6)
    c̄ = mean(sol.u)
    # Open space has a perfectly linear concentration profile, i.e., c̄ = 0.5
    @test c̄ ≈ 0.5 atol = 1e-4
    c_grid = vec_to_grid(sol.u, straight_channel)
    tau = tortuosity(c_grid; axis=:x)
    # Open space has no tortuosity, i.e., τ = 1
    @test tau ≈ 1.0 atol = 1e-4
    # Formation factor though should be exactly 2 since it's half open
    ff = formation_factor(c_grid, axis=:x)
    @test ff ≈ 2.0 atol = 1e-4
end
