using LinearSolve
using Statistics
using Tortuosity
using Test

@testset verbose=true "Tortuosity.jl" begin

    @testset verbose=true "Utility functions" begin
        include("test_utils.jl")
    end

    @testset verbose=true "Basic geometries" begin
        include("test_basic.jl")
    end

    @testset verbose=true "Blobs (Gaussian noise)" begin
        include("test_blobs.jl")
    end

    @testset verbose=true "Against PoreSpy's tortuosity_fd" begin
        include("test_benchmark.jl")
    end

end
