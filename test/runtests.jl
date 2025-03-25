using Test
using Tortuosity

@testset verbose = true "Tortuosity.jl" begin
    @testset verbose = true "Utility functions" begin
        include("test_utils.jl")
    end

    @testset verbose = true "Basic geometries" begin
        include("test_basic.jl")
    end

    @testset verbose = true "Blobs (Gaussian noise)" begin
        include("test_blobs.jl")
    end
end
