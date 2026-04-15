using Test
using Pkg

# Metal.jl has Apple-only binary artifacts and therefore cannot live in
# [extras] unconditionally (Pkg resolution would fail on Linux/Windows).
# On Apple Silicon, install it into the active test environment at runtime.
# This matches the pattern used by NonlinearSolve.jl and LinearSolve.jl.
#
# CUDA.jl stays in [extras] because it installs cleanly on every supported
# platform (including macOS-arm64 — it just reports `functional() == false`
# there). Running `Pkg.add` on CUDA would silently promote it from [weakdeps]
# to [deps], breaking the extension-loading model.
if Sys.isapple() && Sys.ARCH === :aarch64
    try
        Pkg.add("Metal"; preserve=Pkg.PRESERVE_ALL)
    catch e
        @info "Failed to install Metal.jl: $(sprint(showerror, e))"
    end
end

using Tortuosity

# Try to enable a GPU backend. CUDA is the oracle for test_gpu_parity.jl;
# any functional backend is enough for test_gpu_e2e.jl.
const _has_cuda = try
    @eval using CUDA
    Base.invokelatest(CUDA.functional)
catch e
    @info "CUDA not available: $(sprint(showerror, e))"
    false
end

const _has_metal = try
    @eval using Metal
    # Metal.functional() exists on recent Metal.jl; fall back to load-success.
    isdefined(Metal, :functional) ? Base.invokelatest(Metal.functional) : true
catch e
    @info "Metal not available: $(sprint(showerror, e))"
    false
end

const _has_gpu = _has_cuda || _has_metal

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

    @testset verbose = true "PortableSparseCSC operations" begin
        include("test_sparse_ops.jl")
    end

    @testset verbose = true "Cross-implementation parity" begin
        include("test_impl_parity.jl")
    end

    @testset verbose = true "Example scripts" begin
        include("test_examples.jl")
    end

    if _has_cuda
        @testset verbose = true "GPU parity vs old CUDA baseline" begin
            include("test_gpu_parity.jl")
        end
    else
        @info "Skipping CUDA parity tests (CUDA not functional)"
    end

    if _has_gpu
        @testset verbose = true "GPU end-to-end pipeline" begin
            include("test_gpu_e2e.jl")
        end
    else
        @info "Skipping GPU end-to-end tests (no GPU backend functional)"
    end
end
