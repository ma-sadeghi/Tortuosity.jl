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
    _warn_nonpercolating,
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

    @testset "above ~50M voxels the check is skipped rather than run" begin
        # The automatic choice is by size, because the check labels connected
        # components over the full grid and that allocates an `Int` array the
        # shape of the image — 512 MB at 400³, several GB at 800³, at the moment
        # the solve is about to need all of it. Only the "check" side of that
        # branch is reachable through `SteadyDiffusionProblem`; assembling a
        # 64M-voxel system to reach the other side is not, so this drives
        # `_warn_nonpercolating` directly.
        #
        # Not skipped, and it does not need to be: a `BitArray` of 64M voxels is
        # 7.6 MiB, and the guard returns before touching it. Measured on this
        # machine: 0 bytes allocated, 3.6 µs.
        big = falses(400, 400, 400)
        big[:, 100:102, 100:102] .= true      # a duct spanning :x
        big[200:205, 300, 300] .= true        # …and stranded volume beside it
        @test length(big) > 50_000_000

        # Silence alone would also be what a *clean* image produces, so the
        # load-bearing assertion is that no work happened at all: a real check
        # over 64M voxels cannot come in under a few bytes. Bounded rather than
        # compared to zero because `@allocated` differences `gc_bytes()`, which
        # sums every thread's counter: a background task allocating during the
        # measurement would fail an exact zero without the guard having run.
        @test_logs _warn_nonpercolating(big, :x, nothing)
        _warn_nonpercolating(big, :x, nothing)                  # warm
        @test (@allocated _warn_nonpercolating(big, :x, nothing)) < 4096

        # …and the image really does have something to warn about, so the
        # silence above is the guard rather than the geometry. This is the
        # expensive half — 625 MiB and 0.36 s, an order of magnitude under a
        # GitHub-hosted runner's 7 GB — and it is what stops a fixture edit from
        # quietly making the testset vacuous.
        @test_logs (:warn,) match_mode = :any _warn_nonpercolating(big, :x, true)

        # `false` skips below the threshold too, which is the keyword's whole
        # purpose on an image small enough to be checked automatically.
        @test_logs _warn_nonpercolating(img, :x, false)
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

    @testset "a diffusivity field handed to τ must line up with the mask" begin
        # `_reference_diffusivity` reduces over `eachindex(img, D)` rather than
        # materialising `D[img]`, and the shape agreement it checks is the one
        # logical indexing used to give for free. Without it a mismatched field
        # reads the wrong voxels and returns a plausible D0 — the reference the
        # whole of τ is measured against.
        img = ones(Bool, 6, 5, 4)
        c = reconstruct_field(
            solve(SteadyDiffusionProblem(img; axis=:x, gpu=false).prob,
                  KrylovJL_CG(); reltol=1e-10).u, img,
        )
        @test_throws DimensionMismatch tortuosity(c, img; axis=:x, D=ones(6, 5, 5))
        # Same element count, different shape: the one case a length check alone
        # would let through.
        @test_throws DimensionMismatch tortuosity(c, img; axis=:x, D=ones(5, 6, 4))
        @test_throws DimensionMismatch formation_factor(c, img; axis=:x, D=ones(5, 6, 4))
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

# --- Preconditioner ---

@testset "the two-level preconditioner rejects a mask that is not the matrix's" begin
    # `agg` is sized from `size(A, 1)` but filled at the pore ordinals `img`
    # produces, and it is later read back as an unchecked index into the
    # stencil. A mask with a different pore count leaves entries unwritten, so
    # the consequence is an out-of-bounds device write off uninitialised memory
    # rather than a wrong answer — which is why it is refused up front.
    img = ones(Bool, 12, 12, 12)
    img[5:8, 5:8, 5:8] .= false
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false)

    trimmed = copy(img)
    trimmed[6, 6, 2] = false          # one pore voxel fewer than `A` has rows
    @test count(trimmed) == count(img) - 1
    @test_throws AssertionError two_level_preconditioner(sim.prob.A, trimmed; block=4)

    # The mask the matrix was actually built from is accepted, so the guard is
    # rejecting the disagreement rather than the call shape.
    @test two_level_preconditioner(sim.prob.A, img; block=4) isa
          Tortuosity.TwoLevelPreconditioner
end
