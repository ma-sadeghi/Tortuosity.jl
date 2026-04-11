using Test
using Tortuosity

# Try to enable optional GPU parity tests (requires CUDA + functional GPU)
const _has_cuda = try
    @eval using CUDA
    Base.invokelatest(CUDA.functional)
catch e
    @info "GPU parity tests disabled: $(sprint(showerror, e))"
    false
end

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

    @testset verbose = true "Transient" begin
        include("test_transient.jl")
    end

    if _has_cuda
        @testset verbose = true "GPU parity vs old CUDA baseline" begin
            include("test_gpu_parity.jl")
        end
        @testset verbose = true "GPU end-to-end pipeline" begin
            include("test_gpu_e2e.jl")
        end
    else
        @info "Skipping GPU parity + end-to-end tests (CUDA not functional)"
    end
end
