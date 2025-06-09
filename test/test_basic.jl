using Statistics
using Test
using Tortuosity
using Tortuosity: formation_factor, tortuosity, vec_to_grid

# 3D fixtures
open_space = ones(Bool, (32, 32, 32))
straight_channel = ones(Bool, (32, 32, 32))
straight_channel[:, :, 1:16] .= 0

# 2D fixtures
open_space_2d = ones(Bool, (32, 32))
straight_channel_2d = ones(Bool, (32, 32))
straight_channel_2d[:, 1:16] .= 0

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
    ff = formation_factor(c_grid; axis=:x)
    @test ff ≈ 2.0 atol = 1e-4
end

@testset "Open space 2D, $(ax)-axis" for ax in (:x, :y)
    sim = TortuositySimulation(open_space_2d; axis=ax, gpu=false)
    sol = solve(sim.prob, KrylovJL_CG(); reltol=1e-6)
    c̄ = mean(sol.u)
    # Open space has a perfectly linear concentration profile, i.e., c̄ = 0.5
    @test c̄ ≈ 0.5 atol = 1e-4
    c_grid = vec_to_grid(sol.u, open_space_2d)
    tau = tortuosity(c_grid; axis=ax)
    # Open space has no tortuosity, i.e., τ = 1
    @test tau ≈ 1.0 atol = 1e-4
end

@testset "Straight channel 2D (half-open square)" begin
    sim = TortuositySimulation(straight_channel_2d; axis=:x, gpu=false)
    sol = solve(sim.prob, KrylovJL_CG(); reltol=1e-6)
    c̄ = mean(sol.u)
    # Open space has a perfectly linear concentration profile, i.e., c̄ = 0.5
    @test c̄ ≈ 0.5 atol = 1e-4
    c_grid = vec_to_grid(sol.u, straight_channel_2d)
    tau = tortuosity(c_grid; axis=:x)
    # Open space has no tortuosity, i.e., τ = 1
    @test tau ≈ 1.0 atol = 1e-4
    # Formation factor though should be exactly 2 since it's half open
    ff = formation_factor(c_grid; axis=:x)
    @test ff ≈ 2.0 atol = 1e-4
end
