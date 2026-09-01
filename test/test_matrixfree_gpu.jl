# Device half of the matrix-free parity suite. `test_matrixfree.jl` is the CPU
# half and the two share a specification: `build_steady_system` says what the
# operator must do, and `MaskedLaplacian` must do it wherever its index array
# lives. Everything here runs in Float32 against an assembled
# `PortableSparseCSC`, which on CUDA is served by CUSPARSE.
#
# Caller (runtests.jl) is responsible for ensuring CUDA is loaded and functional
# before including this file.

using Test
using Random
using LinearAlgebra
using CUDA
using Tortuosity
using Tortuosity: Imaginator,
    MaskedLaplacian,
    DEFAULT_GPU_MAX_COARSE,
    build_steady_operator,
    build_steady_system,
    _gpu_max_coarse,
    _resolve_max_coarse,
    _free!

# The sizes `test_gpu_e2e.jl` runs: several workgroups deep in every dimension,
# still small enough to stay inside the suite's time budget.
function matrixfree_gpu_fixtures()
    imgs = Tuple{String,Array{Bool,3}}[]
    push!(imgs, ("open 16^3", ones(Bool, 16, 16, 16)))
    let img = ones(Bool, 24, 24, 24)
        img[:, :, 1:12] .= false
        push!(imgs, ("half 24^3", img))
    end
    for seed in (1, 42)
        img = Array{Bool}(
            Imaginator.blobs(;
                shape=(32, 32, 32), porosity=0.55f0, blobiness=1, seed=seed
            ),
        )
        push!(imgs, ("blob 32^3 seed=$(seed)", img))
    end
    return imgs
end

# Smooth, strictly positive and different in every voxel, so no two harmonic
# means coincide by accident.
device_diffusivity(img) = CuArray(
    Float32[0.5f0 + 0.1f0i + 0.02f0j + 0.003f0k
            for i in 1:size(img, 1), j in 1:size(img, 2), k in 1:size(img, 3)]
)

# Each fixture's device buffers go back to the pool once the assertions on them
# are done, so a long loop of images never holds them all at once. `nothing`
# takes the no-op fallback, which is what the uniform-diffusivity case passes.
release!(xs...) = foreach(_free!, xs)

matrixfree_gpu_images = matrixfree_gpu_fixtures()

# --- Apply parity ---
#
# CUSPARSE reads the assembled matrix as CSR and reduces a row per thread; the
# stencil kernel recomputes the same row from the mask. Both accumulate in
# Float32, so the agreement is bounded by rounding rather than by the physics —
# the gap measured across these fixtures is ~1e-7.

@testset "apply parity vs the assembled matrix on device" begin
    for (label, img) in matrixfree_gpu_images
        nnodes = count(img)
        d_img = CuArray(img)
        D_dev = device_diffusivity(img)
        for (dlabel, D) in (("uniform D", nothing), ("variable D", D_dev)),
            axis in (:x, :y, :z)

            @testset "$(label) — $(dlabel) — axis=$(axis)" begin
                A, _ = build_steady_system(d_img; nnodes=nnodes, axis=axis, D=D, T=Float32)
                op, _ = build_steady_operator(d_img; nnodes=nnodes, axis=axis, D=D, T=Float32)
                rng = MersenneTwister(4711)
                for _ in 1:2
                    x = CuArray(randn(rng, Float32, nnodes))
                    y_asm = mul!(CUDA.zeros(Float32, nnodes), A, x)
                    y_mf = mul!(CUDA.zeros(Float32, nnodes), op, x)
                    @test isapprox(Array(y_mf), Array(y_asm); rtol=1e-5)
                    release!(x, y_asm, y_mf)
                end
                release!(A, op)
            end
        end
        release!(d_img, D_dev)
    end
end

@testset "5-argument mul! applies alpha and beta on device" begin
    img = ones(Bool, 24, 24, 24)
    nnodes = count(img)
    d_img = CuArray(img)
    D_dev = device_diffusivity(img)
    for (dlabel, D) in (("uniform D", nothing), ("variable D", D_dev))
        @testset "$(dlabel)" begin
            A, _ = build_steady_system(d_img; nnodes=nnodes, axis=:x, D=D, T=Float32)
            op, _ = build_steady_operator(d_img; nnodes=nnodes, axis=:x, D=D, T=Float32)
            rng = MersenneTwister(1301)
            x = CuArray(randn(rng, Float32, nnodes))
            y0 = CuArray(randn(rng, Float32, nnodes))

            for (alpha, beta) in ((1f0, 0f0), (2.5f0, -0.75f0), (0f0, 3f0), (1f0, 1f0))
                y_asm = mul!(copy(y0), A, x, alpha, beta)
                y_mf = mul!(copy(y0), op, x, alpha, beta)
                @test isapprox(Array(y_mf), Array(y_asm); rtol=1e-5)
                release!(y_asm, y_mf)
            end

            # beta = 0 must not read `y`, so a dirty buffer cannot leak into the
            # answer — and a NaN one advertises the leak loudly. The reference
            # is taken from a clean buffer: nothing here claims CUSPARSE ignores
            # a NaN `y` too.
            y_mf = mul!(CUDA.fill(NaN32, nnodes), op, x, 2f0, 0f0)
            @test !any(isnan, Array(y_mf))
            y_asm = mul!(CUDA.zeros(Float32, nnodes), A, x, 2f0, 0f0)
            @test isapprox(Array(y_mf), Array(y_asm); rtol=1e-5)

            # Integer scalars must not widen the arithmetic away from eltype(y).
            y_int = mul!(CUDA.zeros(Float32, nnodes), op, x, 1, 0)
            @test Array(y_int) == Array(mul!(CUDA.zeros(Float32, nnodes), op, x))

            release!(x, y0, y_mf, y_asm, y_int, A, op)
        end
    end
    release!(d_img, D_dev)
end

# --- Right-hand side ---

@testset "RHS parity: b matches build_steady_system on device" begin
    # Each entry is a sum of edge weights taken in `_NEIGHBOURS` order by a
    # single thread in both paths, so the device reduction has no order to vary
    # and `==` holds. Both claims are asserted on the same pair of builds: the
    # tolerance one is what the physics needs, the equality one is what the
    # shared per-thread summation buys, so a future kernel reordering shows up
    # as a rounding-level change rather than as a bare failure.
    for (label, img) in matrixfree_gpu_images
        nnodes = count(img)
        d_img = CuArray(img)
        D_dev = device_diffusivity(img)
        for (dlabel, D) in (("uniform D", nothing), ("variable D", D_dev)),
            axis in (:x, :y, :z)

            @testset "$(label) — $(dlabel) — axis=$(axis)" begin
                _, b_asm = build_steady_system(d_img; nnodes=nnodes, axis=axis, D=D, T=Float32)
                _, b_mf = build_steady_operator(d_img; nnodes=nnodes, axis=axis, D=D, T=Float32)
                @test eltype(b_mf) === eltype(b_asm) === Float32
                @test length(b_mf) == nnodes
                @test isapprox(Array(b_mf), Array(b_asm); rtol=1e-6)
                @test Array(b_mf) == Array(b_asm)
                release!(b_asm, b_mf)
            end
        end
        release!(d_img, D_dev)
    end
end

# --- Host/device cross-parity ---

@testset "the host and device operators agree on the same image" begin
    img = Array{Bool}(
        Imaginator.blobs(; shape=(32, 32, 32), porosity=0.55f0, blobiness=1, seed=1)
    )
    nnodes = count(img)
    d_img = CuArray(img)
    D_host = Float64[0.5 + 0.1i + 0.02j + 0.003k
                     for i in 1:size(img, 1), j in 1:size(img, 2), k in 1:size(img, 3)]
    D_dev = CuArray(Float32.(D_host))

    for (dlabel, D_cpu, D_gpu) in
        (("uniform D", nothing, nothing), ("variable D", D_host, D_dev))

        @testset "$(dlabel)" begin
            op_cpu, b_cpu = build_steady_operator(img; nnodes=nnodes, axis=:x, D=D_cpu)
            op_gpu, b_gpu = build_steady_operator(
                d_img; nnodes=nnodes, axis=:x, D=D_gpu, T=Float32
            )

            # The pore numbering is what makes `sol.u` mean the same thing on
            # either device, so it is compared entry for entry rather than
            # inferred from the answers agreeing.
            @test Array(op_gpu.idx) == op_cpu.idx
            @test size(op_gpu) == size(op_cpu)

            x = randn(MersenneTwister(9), nnodes)
            y_cpu = mul!(zeros(nnodes), op_cpu, x)
            x_dev = CuArray(Float32.(x))
            y_gpu = mul!(CUDA.zeros(Float32, nnodes), op_gpu, x_dev)
            @test isapprox(Array(y_gpu), y_cpu; rtol=1e-5)
            @test isapprox(Array(b_gpu), b_cpu; rtol=1e-5)

            release!(x_dev, y_gpu, b_gpu, op_gpu, op_cpu)
        end
    end
    release!(d_img, D_dev)
end

# --- LinearSolve integration ---

@testset "solving on device leaves the solution in the cache's own u" begin
    # Proves the `init_cacheval` specialization is the one that fires: the
    # generic path allocates the workspace's `x` and lets LinearSolve replace it
    # afterwards, so `cacheval.x === cache.u` holds only when ours built it.
    img = ones(Bool, 16, 16, 16)
    nnodes = count(img)
    d_img = CuArray(img)
    op, b = build_steady_operator(d_img; nnodes=nnodes, axis=:x, T=Float32)
    prob = Tortuosity.LinearProblem(op, b)
    cache = Tortuosity.LinearSolve.init(prob, KrylovJL_CG(); reltol=1f-6)
    sol = solve!(cache)
    @test cache.cacheval.x === cache.u
    @test cache.u isa CuArray{Float32}
    @test Symbol(sol.retcode) === :Success
    release!(b, op, d_img)
end

# --- End to end ---

@testset "end-to-end device solve agrees with the assembled path" begin
    img = Array{Bool}(
        Imaginator.blobs(; shape=(32, 32, 32), porosity=0.55f0, blobiness=1, seed=1)
    )
    @test any(img[1, :, :]) && any(img[end, :, :])
    nnodes = count(img)
    d_img = CuArray(img)

    A, b_asm = build_steady_system(d_img; nnodes=nnodes, axis=:x, T=Float32)
    op, b_mf = build_steady_operator(d_img; nnodes=nnodes, axis=:x, T=Float32)

    sol_asm = solve(Tortuosity.LinearProblem(A, copy(b_asm)), KrylovJL_CG(); reltol=1e-8)
    sol_mf = solve(Tortuosity.LinearProblem(op, copy(b_mf)), KrylovJL_CG(); reltol=1e-8)
    @test Symbol(sol_asm.retcode) === :Success
    @test Symbol(sol_mf.retcode) === :Success

    tau_asm = tortuosity(reconstruct_field(sol_asm.u, img), img; axis=:x)
    tau_mf = tortuosity(reconstruct_field(sol_mf.u, img), img; axis=:x)
    @test tau_mf > 1
    @test isapprox(tau_mf, tau_asm; rtol=1e-4)

    # The apply's summation order is fixed and every output is owned by one
    # thread, so two identical solves agree to the last bit on device too.
    sol_again = solve(Tortuosity.LinearProblem(op, copy(b_mf)), KrylovJL_CG(); reltol=1e-8)
    tau_again = tortuosity(reconstruct_field(sol_again.u, img), img; axis=:x)
    @test Array(sol_again.u) == Array(sol_mf.u)
    @test tau_again == tau_mf

    release!(b_asm, b_mf, A, op, d_img)
end

# --- Interface ---

@testset "the device operator carries Int32 pore ordinals and releases them" begin
    img = ones(Bool, 16, 16, 16)
    nnodes = count(img)
    d_img = CuArray(img)
    op, b = build_steady_operator(d_img; nnodes=nnodes, axis=:x, T=Float32)

    # Int32 is unconditional on device: the index array is the operator's whole
    # state, and halving its traffic is what the apply spends its time on.
    @test op isa MaskedLaplacian
    @test op.idx isa CuArray
    @test eltype(op.idx) === Int32
    @test eltype(op) === Float32
    @test eltype(b) === Float32
    @test Tortuosity._async_return_safe(op.idx) == (pkgversion(CUDA) >= v"5.4")
    @test Tortuosity._steady_workgroup(op.idx) == (32, 2, 2)
    @test Tortuosity._steady_workgroup(CUDA.ones(Bool, 4, 4, 1)) == (64, 4, 1)
    @test Tortuosity._gpu_min_nodes(CUDABackend()) == 20_000
    @test Tortuosity._precond_min_nodes(b) == 3_000
    @test _gpu_max_coarse(249_999_999) == 14_000
    @test _gpu_max_coarse(250_000_000) == 16_000
    @test _gpu_max_coarse(499_999_999) == 16_000
    @test _gpu_max_coarse(500_000_000) == 32_000
    @test size(op) == (nnodes, nnodes)
    @test size(op.idx) == size(img)
    @test _resolve_max_coarse(op, nothing, (25, 25, 25)) == DEFAULT_GPU_MAX_COARSE
    @test _resolve_max_coarse(op, nothing, (1, 1, 14_001)) == Tortuosity.DEFAULT_MAX_COARSE
    @test _resolve_max_coarse(op, 321, (1, 1, 14_001)) == 321

    @test op.owns_D === false
    @test _free!(op) === nothing
    release!(b, d_img)
end

@testset "the constructor hands the operator the device D it made for it" begin
    img = ones(Bool, 20, 16, 16)
    D = fill(1.5, size(img))

    # `D` arrives on the host, so construction makes a device copy that only the
    # operator will ever reference. Without ownership that copy is a leak, and
    # it is grid-sized — the one array where leaking it matters.
    sim = SteadyDiffusionProblem(img; axis=:x, D=D, gpu=true, matrixfree=true)
    op = sim.prob.A
    @test op.D isa CuArray
    @test op.D !== D
    @test op.owns_D === true
    @test _free!(op) === nothing

    # Handing in a device array does not avoid the copy — the adapter allocates
    # a fresh one even for a `CuArray` that is already the right element type —
    # so the operator owns that one too, and the caller's array is left alone.
    d_D = CuArray(Float32.(D))
    kept = SteadyDiffusionProblem(img; axis=:x, D=d_D, gpu=true, matrixfree=true)
    @test kept.prob.A.D !== d_D
    @test kept.prob.A.owns_D === true
    @test Array(d_D) == Float32.(D)

    host = SteadyDiffusionProblem(img; axis=:x, D=d_D, gpu=false, matrixfree=true)
    @test host.prob.A.D isa Array
    @test host.D0 == 1.5f0

    D_lazy = reshape(range(0.5f0, 1.5f0; length=length(img)), size(img))
    lazy = SteadyDiffusionProblem(img; axis=:x, D=D_lazy, gpu=true, matrixfree=true)
    @test lazy.D0 == 1.5f0
    @test isfinite(tortuosity(solve(lazy).u, lazy))

    u = solve(kept.prob, KrylovJL_CG(); reltol=1.0f-8).u
    c = reconstruct_field(u, img)
    tau_grid = tortuosity(c, img; axis=:x, D=Float32.(D))
    @test kept.D0 == 1.5f0
    @test tortuosity(u, kept) ≈ tau_grid rtol = 1e-4

    @test _free!(kept.prob.A) === nothing
    release!(d_D)
end
