# Input validation and failure modes.
#
# Every assertion here guards a case where the code would otherwise produce
# silently wrong numbers rather than crash — an all-solid image, a diffusivity
# field that doesn't line up with the mask, a flux index off the end of the
# domain, a pore-only vector handed to an observable without the lookup table
# it needs to be interpreted. Optimisation passes tend to strip validation on
# the grounds that it costs time in a hot path; these tests make that a visible
# decision rather than an accident.

using Test
using Tortuosity
using Tortuosity:
    PortableSparseCSC,
    _build_connectivity_list_ka,
    fit_voxel_diffusivity,
    flux,
    reconstruct_field,
    slice_concentration

# `gpu=true` only errors when no backend package is loaded. The suite runs with
# CUDA loaded on machines that have it, so those checks are conditional.
const NO_GPU_BACKEND = isnothing(Tortuosity._preferred_gpu_backend[])

# --- SteadyDiffusionProblem ---

@testset "SteadyDiffusionProblem input validation" begin
    @testset "an all-solid image is rejected" begin
        @test_throws AssertionError SteadyDiffusionProblem(falses(4, 4, 4); axis=:x, gpu=false)
    end

    @testset "the mask must be boolean" begin
        @test_throws AssertionError SteadyDiffusionProblem(ones(Int, 4, 4, 4); axis=:x, gpu=false)
        @test_throws AssertionError SteadyDiffusionProblem(rand(4, 4, 4); axis=:x, gpu=false)
    end

    @testset "a diffusivity field must match the image shape" begin
        img = ones(Bool, 4, 4, 4)
        @test_throws AssertionError SteadyDiffusionProblem(img; axis=:x, gpu=false, D=ones(5, 4, 4))
    end

    @testset "a diffusivity field must be positive exactly on the pore space" begin
        # Otherwise the harmonic-mean edge weights silently include solid voxels.
        img = ones(Bool, 4, 4, 4)
        img[1, 1, 1] = false
        @test_throws AssertionError SteadyDiffusionProblem(img; axis=:x, gpu=false, D=ones(4, 4, 4))
    end

    @testset "the axis symbol must be one of :x, :y, :z" begin
        @test_throws ErrorException SteadyDiffusionProblem(ones(Bool, 4, 4, 4); axis=:w, gpu=false)
    end

    @testset "the transport axis needs at least two voxels" begin
        # Inlet and outlet would otherwise be the same voxels, and the solve
        # returns a silently meaningless field rather than failing. Easy to hit
        # by accident: a 2D image is promoted to (m, n, 1), so `axis=:z` on 2D
        # data lands here. Matches TransientDiffusionProblem's guard.
        @test_throws AssertionError SteadyDiffusionProblem(ones(Bool, 4, 4, 1); axis=:z, gpu=false)
        @test_throws AssertionError SteadyDiffusionProblem(ones(Bool, 1, 4, 4); axis=:x, gpu=false)
        @test_throws AssertionError SteadyDiffusionProblem(ones(Bool, 1, 1, 1); axis=:x, gpu=false)
        # A 2D image promoted to (m, n, 1) is still fine for in-plane transport.
        @test SteadyDiffusionProblem(ones(Bool, 4, 4); axis=:x, gpu=false) isa SteadyDiffusionProblem
    end

    if NO_GPU_BACKEND
        @testset "gpu=true without a loaded backend errors instead of falling back" begin
            @test_throws ErrorException SteadyDiffusionProblem(ones(Bool, 4, 4, 4); axis=:x, gpu=true)
        end
    end
end

@testset "non-percolating pore space is warned about, not altered" begin
    # A duct spanning the domain, plus a detached blob that does not.
    img = falses(12, 8, 8)
    img[:, 3:5, 3:5] .= true
    img[5:7, 7, 7] .= true

    @testset "warns when part of the pore space is stranded" begin
        sim = @test_logs (:warn,) match_mode = :any SteadyDiffusionProblem(
            img; axis=:x, gpu=false,
        )
        @test sim isa SteadyDiffusionProblem
    end

    @testset "silent when every pore voxel percolates" begin
        clean = falses(12, 8, 8)
        clean[:, 3:5, 3:5] .= true
        @test_logs SteadyDiffusionProblem(clean; axis=:x, gpu=false)
        @test_logs SteadyDiffusionProblem(ones(Bool, 6, 6, 6); axis=:y, gpu=false)
    end

    @testset "can be silenced, and forced" begin
        @test_logs SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
        @test_logs (:warn,) match_mode = :any SteadyDiffusionProblem(
            img; axis=:x, gpu=false, warn_nonpercolating=true,
        )
    end

    @testset "the warning changes nothing about the assembled system" begin
        quiet = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
        loud = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=true)
        @test quiet.prob.A == loud.prob.A
        @test quiet.prob.b == loud.prob.b
    end
end

# --- TransientDiffusionProblem ---

@testset "TransientDiffusionProblem input validation" begin
    @testset "an all-solid image is rejected" begin
        @test_throws AssertionError TransientDiffusionProblem(falses(4, 4, 4); axis=:z, gpu=false)
    end

    @testset "a diffusivity field must match the image shape" begin
        @test_throws AssertionError TransientDiffusionProblem(
            ones(Bool, 4, 4, 4); axis=:z, D=ones(4, 4, 5), gpu=false,
        )
    end

    @testset "the transport axis needs at least two voxels" begin
        # With one voxel the inlet and outlet faces coincide and voxel_size
        # would divide by zero.
        @test_throws AssertionError TransientDiffusionProblem(trues(4, 4, 1); axis=:z, gpu=false)
    end

    @testset "a time-dependent boundary must return a number" begin
        @test_throws AssertionError TransientDiffusionProblem(
            trues(4, 4, 4); axis=:z, bc_inlet=t -> "one", gpu=false,
        )
    end

    @testset "the axis symbol must be one of :x, :y, :z" begin
        @test_throws ErrorException TransientDiffusionProblem(trues(4, 4, 4); axis=:w, gpu=false)
    end

    if NO_GPU_BACKEND
        @testset "gpu=true without a loaded backend errors instead of falling back" begin
            @test_throws ErrorException TransientDiffusionProblem(trues(4, 4, 4); axis=:z, gpu=true)
        end
    end
end

# --- Observables ---

@testset "observable input validation" begin
    img = ones(Bool, 6, 4, 4)
    img[3, 2, 2] = false
    prob = TransientDiffusionProblem(img; axis=:x, gpu=false)
    u = rand(count(prob.img))

    @testset "flux rejects an index outside the domain" begin
        # `ind` names the *upstream* slice of a face, so it must stay below N.
        @test_throws AssertionError flux(u, 1.0, 1.0, prob.img, :x; ind=0, pore_index=prob.pore_index)
        @test_throws AssertionError flux(u, 1.0, 1.0, prob.img, :x; ind=6, pore_index=prob.pore_index)
        @test_throws AssertionError flux(u, 1.0, 1.0, prob.img, :x; ind=99, pore_index=prob.pore_index)
    end

    @testset "a pore-only vector needs a pore_index to be interpreted" begin
        @test_throws AssertionError flux(u, 1.0, 1.0, prob.img, :x; ind=1)
        @test_throws AssertionError slice_concentration(u, prob.img, :x, 1)
    end

    @testset "reconstruct_field requires the vector length to match the mask" begin
        @test_throws AssertionError reconstruct_field(rand(count(prob.img) + 1), prob.img)
        @test_throws AssertionError reconstruct_field(rand(count(prob.img) - 1), prob.img)
    end
end

# --- Fitting ---

@testset "fitting input validation" begin
    img = trues(1, 1, 17)

    @testset "time-dependent boundaries are unsupported" begin
        prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=t -> 1.0, bc_outlet=0.0, gpu=false)
        sol = solve(prob, ROCK4(); saveat=0.05, tspan=(0.0, 0.5))
        @test_throws AssertionError fit_effective_diffusivity(sol, prob, :conc)
        @test_throws AssertionError fit_voxel_diffusivity(sol, prob; n_samples=1)
    end

    @testset "an insulated inlet cannot be fitted" begin
        prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=nothing, bc_outlet=0.0, gpu=false)
        sol = solve(prob, ROCK4(); saveat=0.05, tspan=(0.0, 0.5), u0=fill(0.5, size(img)))
        @test_throws AssertionError fit_effective_diffusivity(sol, prob, :conc)
    end

    @testset "unknown observables are rejected by name" begin
        prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
        sol = solve(prob, ROCK4(); saveat=0.05, tspan=(0.0, 0.5))
        @test_throws ErrorException fit_effective_diffusivity(sol, prob, :velocity)
    end

    @testset "fit_voxel_diffusivity guards its sampling and time window" begin
        prob = TransientDiffusionProblem(trues(3, 3, 9); axis=:z, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
        sol = solve(prob, ROCK4(); saveat=0.05, tspan=(0.0, 0.5))
        # The slice at depth=0.5 holds 9 pore voxels; asking for more must fail
        # rather than silently sampling with replacement.
        @test_throws AssertionError fit_voxel_diffusivity(sol, prob; depth=0.5, n_samples=100)
        # A degenerate window collapses to a single time point.
        @test_throws AssertionError fit_voxel_diffusivity(
            sol, prob; depth=0.5, n_samples=2, t_fit=(0.25, 0.25),
        )
    end
end

# --- Stop conditions ---

@testset "stop-condition parameter validation" begin
    prob = TransientDiffusionProblem(trues(1, 1, 8); axis=:z, bc_inlet=1.0, bc_outlet=nothing, gpu=false)
    @test_throws AssertionError StopAtPeriodicState(1.0, prob; compare_window=0.0)
    @test_throws AssertionError StopAtPeriodicState(1.0, prob; compare_window=1.5)
    @test_throws AssertionError StopAtPeriodicState(1.0, prob; depth=0.0)
    @test_throws AssertionError StopAtPeriodicState(1.0, prob; depth=1.5)
end

# --- Sparse type ---

@testset "PortableSparseCSC guards" begin
    A = PortableSparseCSC(2, 2, [1, 2, 3], [1, 2], Float64[1, 1])

    @testset "scalar indexing is refused rather than silently slow" begin
        # Falling back to AbstractMatrix getindex would make `show` and any
        # accidental element access walk the whole matrix — on GPU it would
        # trigger scalar iteration.
        @test_throws ErrorException A[1, 1]
    end

    @testset "set_diag! checks the length of the supplied diagonal" begin
        @test_throws DimensionMismatch Tortuosity.set_diag!(A, Float64[1, 2, 3])
    end
end

@testset "the KA connectivity builder validates a supplied index array" begin
    # A dense Bool array, not a BitArray: KernelAbstractions has no backend for
    # the latter, so it would fail earlier for an unrelated reason.
    img = ones(Bool, 4, 4, 4)
    @test_throws ErrorException _build_connectivity_list_ka(img; inds=zeros(Int32, 3, 3, 3))
end
