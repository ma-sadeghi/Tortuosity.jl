using LinearSolve
using Statistics
using Tortuosity
using Test

@testset verbose=true "Tortuosity.jl" begin

    # Set up fixtures
    img_open = ones(Bool, (32, 32, 32))
    img_blobs = Imaginator.blobs(shape=(32, 32, 32), porosity=0.65, blobiness=0.5, seed=2)
    eps_blobs = sum(img_blobs) / length(img_blobs)
    straight_channel = ones(Bool, (32, 32, 32))
    straight_channel[:, :, 1:16] .= 0
    # Build diffusivity matrix
    D_blobs = fill(NaN, size(img_blobs))
    D_blobs[img_blobs] .= 1.0               # Fluid phase
    D_blobs[.!img_blobs] .= 1e-4            # Solid phase
    # Ground truth values for blobs
    tau_blobs = Dict(:x => 1.637568, :y => 1.501668, :z => 1.564219)
    c̄_blobs = Dict(:x => 0.447173, :y => 0.509658, :z => 0.461578)

    @testset "Blobs 3D, $(ax)-axis" for ax in (:x, :y, :z)
        sim = TortuositySimulation(img_blobs, axis=ax, gpu=false)
        sol = solve(sim.prob, KrylovJL_CG(), reltol=1e-6)
        c̄ = mean(sol.u)
        @test c̄ ≈ c̄_blobs[ax] atol=1e-4
        c_grid = vec_to_field(sol.u, img_blobs)
        tau = tortuosity(c_grid, ax)
        @test tau ≈ tau_blobs[ax] atol=1e-4
    end

    @testset "Blobs 3D, variable diffusivity, $(ax)-axis" for ax in (:x, :y, :z)
        domain = ones(Bool, size(img_blobs))
        sim = TortuositySimulation(domain, axis=ax, gpu=false, D=D_blobs)
        sol = solve(sim.prob, KrylovJL_CG(), reltol=1e-6)
        c̄ = mean(sol.u[img_blobs[:]])
        @test c̄ ≈ c̄_blobs[ax] atol=1e-2
        c_grid = vec_to_field(sol.u, domain)
        tau = tortuosity(c_grid, ax, eps=eps_blobs, D=D_blobs)
        @test tau ≈ tau_blobs[ax] atol=1e-2
    end

    @testset "Open space 3D, $(ax)-axis" for ax in (:x, :y, :z)
        sim = TortuositySimulation(img_open, axis=ax, gpu=false)
        sol = solve(sim.prob, KrylovJL_CG(), reltol=1e-6)
        c̄ = mean(sol.u)
        # Open space has a perfectly linear concentration profile, i.e., c̄ = 0.5
        @test c̄ ≈ 0.5 atol=1e-4
        c_grid = vec_to_field(sol.u, img_open)
        tau = tortuosity(c_grid, ax)
        # Open space has no tortuosity, i.e., τ = 1
        @test tau ≈ 1.0 atol=1e-4
    end

    @testset "Straight channel (half-open cube)" begin
        sim = TortuositySimulation(straight_channel, axis=:x, gpu=false)
        sol = solve(sim.prob, KrylovJL_CG(), reltol=1e-6)
        c̄ = mean(sol.u)
        # Open space has a perfectly linear concentration profile, i.e., c̄ = 0.5
        @test c̄ ≈ 0.5 atol=1e-4
        c_grid = vec_to_field(sol.u, straight_channel)
        tau = tortuosity(c_grid, :x)
        # Open space has no tortuosity, i.e., τ = 1
        @test tau ≈ 1.0 atol=1e-4
        # Formation factor though should be exactly 2 since it's half open
        ff = formation_factor(c_grid, :x)
        @test ff ≈ 2.0 atol=1e-4
    end

end
