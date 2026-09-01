# End-to-end GPU pipeline tests. test_gpu_parity.jl stops at assembled
# matrices; this file exercises the full path — build with gpu=true, solve,
# reshape via reconstruct_field, compute tortuosity — and verifies the final value
# matches analytical expectations or the CPU run. Catches bugs where every
# intermediate is correct but the struct has a stale device array, or a
# postprocessing helper can't cope with a GPU mask.
#
# Backend-agnostic: caller (runtests.jl) must ensure *some* GPU backend is
# loaded and functional (CUDA on Linux x64, Metal on macOS arm64, etc.).
# We never reference a concrete device array type here.

using Test
using Random
using LinearAlgebra
using Tortuosity
using Tortuosity: PortableSparseCSC, Imaginator, _on_gpu, _gpu_adapt, reconstruct_slice

# ---------------------------------------------------------------------------
# Steady-state
# ---------------------------------------------------------------------------

@testset "automatic GPU selection follows the registered backend crossover" begin
    threshold = Tortuosity._gpu_min_nodes(Tortuosity._preferred_gpu_backend[])
    n = ceil(Int, cbrt(threshold))
    img = ones(Bool, n, n, n)
    sim = SteadyDiffusionProblem(img; axis=:x, matrixfree=true, warn_nonpercolating=false)
    @test count(img) >= threshold
    @test _on_gpu(sim.prob.b)
    @test !isnothing(Tortuosity._resolve_precond(:auto, sim, false))

    thin = ones(Bool, 8, 50, 50)
    thin_sim = SteadyDiffusionProblem(
        thin; axis=:x, gpu=true, matrixfree=true, warn_nonpercolating=false,
    )
    @test _on_gpu(thin_sim.prob.b)
    @test isnothing(Tortuosity._resolve_precond(:auto, thin_sim, false))
end

# The small sizes are a regression guard, not padding: small-box GPU runs once
# produced τ ≈ 0.73 instead of 1.0 on Metal because histogram_connections_kernel!
# interleaved a per-bucket atomic with a shared-counter atomic, and the latter
# silently lost updates under contention. The undercount only matters in absolute
# terms (~24 missing entries in the connectivity list), so 16³ and 24³ absorbed
# it within atol=1e-3 — it only became visible once the box was small enough that
# 24 lost edges was a meaningful fraction of the total.
@testset "open space $(n)^3 · axis=$(ax)" for n in (4, 6, 16, 24), ax in (:x, :y, :z)
    img = ones(Bool, n, n, n)
    sim = SteadyDiffusionProblem(img; axis=ax, gpu=true)

    # Invariant: .img stays on the host even when gpu=true, otherwise every
    # CPU-only postprocessing helper downstream is broken.
    @test sim.img isa Array{Bool,3}
    @test sim.prob.A isa PortableSparseCSC

    sol = solve(sim.prob, KrylovJL_CG(); reltol=1.0f-6)
    c_grid = reconstruct_field(sol.u, sim.img)
    @test tortuosity(c_grid, sim.img; axis=ax) ≈ 1.0 atol = 1e-3
end

@testset "half-channel 16^3 (x-axis)" begin
    img = ones(Bool, 16, 16, 16)
    img[:, :, 1:8] .= false
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=true)
    sol = solve(sim.prob, KrylovJL_CG(); reltol=1.0f-6)
    c_grid = reconstruct_field(sol.u, sim.img)
    # Half the cross-section blocked: τ = 1 (no tortuous path), FF = 2.
    @test tortuosity(c_grid, sim.img; axis=:x) ≈ 1.0 atol = 1e-3
    @test formation_factor(c_grid, sim.img; axis=:x) ≈ 2.0 atol = 1e-3
end

@testset "reconstruct_field copies a device mask even behind a wrapper" begin
    # `reconstruct_field` decides whether to copy the mask by asking whether it
    # is host-indexable (`isa Union{Array,BitArray}`), not by asking `_on_gpu`.
    # The difference only shows up here: a device array behind a
    # `PermutedDimsArray` reports `_on_gpu == false`, so an `_on_gpu`-based guard
    # would hand it straight to the logical index and iterate it one voxel at a
    # time from the host. That is not a wrong answer — it is the right answer at
    # 512M device round-trips on an 800³ image — so a value check cannot see it
    # and this asserts the absence of scalar indexing instead.
    mask = Bool[true false true; false true true; true true false]
    dev = _gpu_adapt[](mask)
    wrapped = PermutedDimsArray(dev, (2, 1))
    u = collect(1.0:count(mask))

    # The trap, stated: the wrapper hides the device array from `_on_gpu`.
    @test !_on_gpu(wrapped)
    @test !(wrapped isa Union{Array,BitArray})

    # GPUArraysCore is what every backend routes scalar indexing through, and
    # `ScalarDisallowed` turns a scalar read into an error instead of a warning.
    # Reached by UUID rather than by `using`, so this stays backend-agnostic and
    # needs no new test dependency.
    gac = Base.loaded_modules[Base.PkgId(
        Base.UUID("46192b85-c4d5-4398-a991-12ede77f4527"), "GPUArraysCore",
    )]
    # `task_local_storage(f, key, value)` scopes the setting to the call, so
    # nothing here leaks into the rest of the suite.
    no_scalar(f) = task_local_storage(f, :ScalarIndexing, gac.ScalarDisallowed)

    # `isequal`, not `==`: solid voxels come back as NaN.
    @test isequal(no_scalar(() -> reconstruct_field(u, dev)),
                  reconstruct_field(u, mask))
    @test isequal(no_scalar(() -> reconstruct_field(u, wrapped)),
                  reconstruct_field(u, permutedims(mask, (2, 1))))
end

@testset "CPU/GPU parity on blobs (seed=$(seed))" for seed in (1, 42, 100)
    img = Array{Bool}(
        Imaginator.blobs(; shape=(32, 32, 32), porosity=0.55f0, blobiness=1, seed=seed)
    )
    # Skip degenerate images that don't connect inlet to outlet
    (any(img[1, :, :]) && any(img[end, :, :])) || return

    sim_cpu = SteadyDiffusionProblem(img; axis=:x, gpu=false)
    sol_cpu = solve(sim_cpu.prob, KrylovJL_CG(); reltol=1.0e-8)
    tau_cpu = tortuosity(reconstruct_field(sol_cpu.u, sim_cpu.img), sim_cpu.img; axis=:x)

    sim_gpu = SteadyDiffusionProblem(img; axis=:x, gpu=true, checkpoint_readout=true)
    sol_gpu = solve(sim_gpu.prob, KrylovJL_CG(); reltol=1.0f-6)
    tau_gpu = tortuosity(reconstruct_field(sol_gpu.u, sim_gpu.img), sim_gpu.img; axis=:x)

    # Float32 vs Float64, same geometry → loose rtol absorbs the precision gap
    @test isfinite(tau_gpu)
    @test tau_gpu > 1
    @test tau_cpu ≈ tau_gpu rtol = 1e-3
    @test tortuosity(sol_gpu.u, sim_gpu) ≈ tau_gpu rtol = 1e-3
    @test tortuosity(Array(sol_gpu.u), sim_gpu) ≈ tau_gpu rtol = 1e-3
    u_gpu = Tortuosity._gpu_adapt[](sol_cpu.u)
    @test tortuosity(u_gpu, sim_cpu) ≈ tau_cpu rtol = 1e-3
    Tortuosity._free!(u_gpu)

    partial = 0.5f0 .* sol_gpu.u
    partial_c = reconstruct_field(partial, sim_gpu.img)
    @test Tortuosity._checkpoint_tortuosity(partial, sim_gpu) ≈
          tortuosity(partial_c, sim_gpu.img; axis=:x) rtol = 1e-4
    Tortuosity._free!(partial)
end

# The two-level preconditioner has to give the same tortuosity on the device as
# without it. It runs the coarse solve on the host in Float64 while the fine
# problem is Float32, so this is also the only test that exercises that split.
@testset "two-level preconditioner on GPU (seed=$(seed))" for seed in (1, 42)
    img = Array{Bool}(
        Imaginator.blobs(; shape=(48, 48, 48), porosity=0.55f0, blobiness=1, seed=seed)
    )
    (any(img[1, :, :]) && any(img[end, :, :])) || return

    sim = SteadyDiffusionProblem(img; axis=:x, gpu=true)
    Pl = Tortuosity.two_level_preconditioner(sim; block=8)
    @test Pl isa Tortuosity.TwoLevelPreconditioner
    @test _on_gpu(Pl.agg)

    plain = solve(sim.prob, KrylovJL_CG(); reltol=1.0f-6)
    prec = solve(sim.prob, KrylovJL_CG(); Pl=Pl, reltol=1.0f-6)
    tau_plain = tortuosity(reconstruct_field(plain.u, sim.img), sim.img; axis=:x)
    tau_prec = tortuosity(reconstruct_field(prec.u, sim.img), sim.img; axis=:x)
    @test tau_prec ≈ tau_plain rtol = 1e-3
    @test prec.iters < plain.iters
end

@testset "loose refinement corrections meet the true-residual contract" begin
    img = Imaginator.blobs(
        ; shape=(20, 20, 20), porosity=0.5f0, blobiness=1, seed=42,
    )
    img = Array{Bool}(Imaginator.trim_nonpercolating_paths(img; axis=:x))
    D = ones(Float32, size(img))
    D[2:2:end, :, :] .= 0.1f0
    D[.!img] .= 0.0f0
    sim = SteadyDiffusionProblem(
        img; axis=:x, D=D, gpu=true, warn_nonpercolating=false,
    )
    P = Tortuosity.two_level_preconditioner(sim)
    sol = solve(sim; precond=P)

    @test Symbol(sol.retcode) === :Success
    @test sol.resid[] <= 1.0f-6
end

@testset "a scalar D narrows to the device element type" begin
    # `_gpu_adapt` narrows a diffusivity array to Float32 on the way to the
    # device; a scalar has to take the same narrowing, or `D0` stays Float64 and
    # drags the whole operator's element type up with it.
    img = ones(Bool, 16, 12, 12)
    img[6:11, 4:8, 4:8] .= false
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=true, D=2.0)
    @test eltype(sim.prob.A) === Float32
    @test eltype(sim.prob.b) === Float32

    c = reconstruct_field(Array(solve(sim.prob, KrylovJL_CG(); reltol=1.0f-8).u), sim.img)
    cpu = SteadyDiffusionProblem(img; axis=:x, gpu=false, D=2.0)
    c_cpu = reconstruct_field(solve(cpu.prob, KrylovJL_CG(); reltol=1e-12).u, img)
    @test effective_diffusivity(c, sim.img; axis=:x, D=2.0) ≈
          effective_diffusivity(c_cpu, img; axis=:x, D=2.0) rtol = 1e-4
end

@testset "the preconditioner returns the same bits every time" begin
    # One image, deliberately: bit-determinism is a property of the reduction,
    # not of the geometry, and the second 48³ blob this used to run is
    # statistically the same image at three seconds of device time.
    #
    # Regression guard. `_restrict_kernel!` used to sum each coarse cell's fine
    # values with an atomic, and thread-block arrival order is not fixed between
    # launches, so the non-associative float sum gave a different coarse
    # residual — and therefore a different τ — on every run of the same image.
    # It runs once per CG iteration, so the scatter was re-injected continuously
    # rather than once per solve; the measured spread grew from 0.008% at 100³
    # to 0.094% at 400³, past the accuracy target the benchmark selects on.
    #
    # The restriction is a gather over a fixed adjacency now, so equality here
    # is exact rather than approximate. A tolerance would not express the
    # property and would pass at this size even with the atomic back.
    img = Array{Bool}(
        Imaginator.blobs(; shape=(48, 48, 48), porosity=0.55f0, blobiness=1, seed=1)
    )
    @test any(img[1, :, :]) && any(img[end, :, :])

    sim = SteadyDiffusionProblem(img; axis=:x, gpu=true, warn_nonpercolating=false)
    Pl = Tortuosity.two_level_preconditioner(sim; block=8)

    # The application itself, which is where the schedule-dependent sum lived.
    x = copy(sim.prob.b)
    reference = Array(ldiv!(similar(x), Pl, x))
    for _ in 1:8
        @test Array(ldiv!(similar(x), Pl, x)) == reference
    end

    # And end to end: repeated solves of one image agree to the last bit, which
    # is the property a caller actually depends on.
    taus = map(1:3) do _
        sol = solve(sim.prob, KrylovJL_CG(); Pl=Pl, reltol=1.0f-6)
        tortuosity(reconstruct_field(sol.u, sim.img), sim.img; axis=:x)
    end
    @test allequal(taus)
end

# The image that forces `Ti=Int64` on its own has >306M pore voxels and needs
# ~27 GB of matrix before the solve allocates, so the wide path is exercised by
# asking for it at a size that fits instead. Everything below the type is
# identical either way, which is the property worth pinning: the wide build must
# be the narrow one widened, not a differently-computed one.
@testset "Ti=Int64 on GPU matches the narrow build (seed=$(seed))" for seed in (1, 42)
    img = Array{Bool}(
        Imaginator.blobs(; shape=(32, 32, 32), porosity=0.6f0, blobiness=1, seed=seed)
    )
    (any(img[1, :, :]) && any(img[end, :, :])) || return

    narrow = SteadyDiffusionProblem(img; axis=:x, gpu=true)
    wide = SteadyDiffusionProblem(img; axis=:x, gpu=true, Ti=Int64)

    @test eltype(narrow.prob.A.colptr) === Int32
    @test eltype(wide.prob.A.colptr) === Int64
    @test eltype(wide.prob.A.rowval) === Int64
    # The symmetry claim is what selects the CSR fast path; losing it on the
    # wide build would silently cost a gather-for-scatter downgrade.
    @test wide.prob.A.symmetric

    @test Array(narrow.prob.A.colptr) == Array(wide.prob.A.colptr)
    @test Array(narrow.prob.A.rowval) == Array(wide.prob.A.rowval)
    @test Array(narrow.prob.A.nzval) == Array(wide.prob.A.nzval)
    @test Array(narrow.prob.b) == Array(wide.prob.b)

    tau(sim) = tortuosity(
        reconstruct_field(solve(sim.prob, KrylovJL_CG(); reltol=1.0f-8).u, sim.img),
        sim.img; axis=:x,
    )
    @test tau(wide) ≈ tau(narrow) rtol = 1e-4

    # The preconditioner reads `colptr`/`rowval` straight out of the matrix, so
    # it has to cope with the wide index too.
    Pl = Tortuosity.two_level_preconditioner(wide; block=8)
    @test Pl isa Tortuosity.TwoLevelPreconditioner
    prec = solve(wide.prob, KrylovJL_CG(); Pl=Pl, reltol=1.0f-8)
    @test tortuosity(reconstruct_field(prec.u, wide.img), wide.img; axis=:x) ≈
          tau(narrow) rtol = 1e-3
end

@testset "CPU/GPU parity with variable D (seed=$(seed))" for seed in (3, 17)
    img = Array{Bool}(
        Imaginator.blobs(; shape=(24, 24, 24), porosity=0.6f0, blobiness=1, seed=seed)
    )
    (any(img[1, :, :]) && any(img[end, :, :])) || return

    # Spatially-varying diffusivity over [0.5, 1.5] inside pores; zero in
    # solid voxels so the constructor's subdomain-count assertion holds.
    rng = Random.MersenneTwister(seed)
    D = zeros(Float32, size(img))
    D[img] .= 0.5f0 .+ rand(rng, Float32, count(img))

    sim_cpu = SteadyDiffusionProblem(img; axis=:x, gpu=false, D=Float64.(D))
    sol_cpu = solve(sim_cpu.prob, KrylovJL_CG(); reltol=1.0e-8)
    tau_cpu = tortuosity(
        reconstruct_field(sol_cpu.u, sim_cpu.img), sim_cpu.img; axis=:x, D=Float64.(D),
    )

    sim_gpu = SteadyDiffusionProblem(img; axis=:x, gpu=true, D=D)
    sol_gpu = solve(sim_gpu.prob, KrylovJL_CG(); reltol=1.0f-6)
    tau_gpu = tortuosity(reconstruct_field(sol_gpu.u, sim_gpu.img), sim_gpu.img; axis=:x, D=D)

    @test isfinite(tau_gpu)
    @test tau_gpu > 1
    @test tau_cpu ≈ tau_gpu rtol = 2e-3
    @test tortuosity(sol_gpu.u, sim_gpu) ≈ tau_gpu rtol = 1e-3
end

# ---------------------------------------------------------------------------
# Transient (closes the gap noted in docs/design.md § open issues)
# ---------------------------------------------------------------------------

@testset "TransientDiffusionProblem + solve end-to-end on GPU" begin
    img = Array{Bool}(
        Imaginator.blobs(; shape=(24, 24, 24), porosity=0.6f0, blobiness=1, seed=7)
    )
    (any(img[:, :, 1]) && any(img[:, :, end])) || return

    prob = TransientDiffusionProblem(img; axis=:z, gpu=true)
    @test prob.img isa AbstractArray{Bool}
    @test !_on_gpu(prob.img)
    @test prob.A isa PortableSparseCSC

    sol = solve(prob, ROCK4(); saveat=0.05, tspan=(0.0, 0.2))
    @test sol.t[end] >= 0.2
    @test length(sol.u) == length(sol.t)
    # sol.u lives on CPU even though the solver ran on GPU
    @test all(u isa Vector{Float32} for u in sol.u)
    @test all(all(isfinite, u) for u in sol.u)
    @test all(length(u) == count(prob.img) for u in sol.u)

    # `reconstruct_slice` gathers on whichever device `u` lives on, so that the
    # stop conditions can read one face of a device solution without dragging the
    # whole vector to the host. `sol.u` is always host-resident, so the device
    # branch is only reached from inside the integrator — nothing else in the
    # suite passes it a device vector.
    u_dev = _gpu_adapt[](sol.u[end])
    @test _on_gpu(u_dev)
    for k in (1, 12, 24)
        # isequal, not ==: solid voxels come back as NaN.
        @test isequal(reconstruct_slice(u_dev, prob, k),
                      reconstruct_slice(sol.u[end], prob, k))
    end
end

@testset "TransientDiffusionProblem CPU/GPU parity (scalar snapshot)" begin
    img = Array{Bool}(
        Imaginator.blobs(; shape=(24, 24, 24), porosity=0.6f0, blobiness=1, seed=11)
    )
    (any(img[:, :, 1]) && any(img[:, :, end])) || return

    prob_cpu = TransientDiffusionProblem(img; axis=:z, gpu=false)
    sol_cpu = solve(prob_cpu, ROCK4(); saveat=0.05, tspan=(0.0, 0.15))
    c_mean_cpu = sum(sol_cpu.u[end]) / length(sol_cpu.u[end])

    prob_gpu = TransientDiffusionProblem(img; axis=:z, gpu=true)
    sol_gpu = solve(prob_gpu, ROCK4(); saveat=0.05, tspan=(0.0, 0.15))
    c_mean_gpu = sum(sol_gpu.u[end]) / length(sol_gpu.u[end])

    # Different integrator tolerances + Float32/Float64 → loose check
    @test isapprox(c_mean_cpu, c_mean_gpu; atol=1e-2)
end

# ---------------------------------------------------------------------------
# Iterative refinement of the Float32 path
# ---------------------------------------------------------------------------

# `Float32` CG stops on a recursively-updated residual that drifts away from
# `b - A*x`, so it reports success while the answer is still wrong. The package
# refines against a `Float64` residual before returning. The oracle here is the
# CPU operator, which is `Float64` and uses the same pore ordering, so it checks
# the device path against something that shares none of its arithmetic.
@testset "Float32 solves are refined against a true residual (seed=$(seed))" for seed in (1, 42)
    img = Array{Bool}(
        Imaginator.blobs(; shape=(48, 48, 48), porosity=0.4f0, blobiness=1, seed=seed)
    )
    (any(img[1, :, :]) && any(img[end, :, :])) || return

    sim = SteadyDiffusionProblem(img; axis=:x, gpu=true)
    ref = SteadyDiffusionProblem(img; axis=:x, gpu=false)
    A, b = ref.prob.A, ref.prob.b
    true_resid(u) = norm(b .- A * Float64.(Array(u))) / norm(b)

    plain = solve(sim, KrylovJL_CG(); refine=false)
    refined = solve(sim, KrylovJL_CG())

    # The whole point: the residual the solve reports is the true one, not the
    # recurrence it stopped on. Reporting the recurrence is what hid the defect.
    @test refined.resid[] ≈ true_resid(refined.u) rtol = 1e-5
    # And it is reported the same way an unrefined solve reports it, so a caller
    # never has to branch on the working precision to read the field.
    @test refined.resid isa Base.RefValue

    # Refinement must never hand back a worse answer than the solve it repairs.
    @test true_resid(refined.u) <= true_resid(plain.u)

    # It costs iterations, and they are counted rather than hidden.
    @test refined.iters > plain.iters

    # `stats` must not be the workspace Krylov mutates in place: that object holds
    # the last correction round by the time refinement returns, so a caller
    # reading `stats.niter` would see a handful of iterations for a solve that
    # took hundreds. It is a copy describing the base solve, and it says so.
    @test refined.stats !== refined.cache.cacheval.stats
    @test refined.stats.niter == plain.iters
    @test occursin("refined", refined.stats.status)
    # A refined solve that reaches the requested tolerance reports success, and
    # the two places that say so must agree.
    @test Symbol(refined.retcode) === :Success
    @test refined.stats.solved

    # Both still solve the physics.
    @test tortuosity(reconstruct_field(refined.u, sim.img), sim.img; axis=:x) ≈
          tortuosity(reconstruct_field(plain.u, sim.img), sim.img; axis=:x) rtol = 1e-2
end

# Refinement is keyed on the working precision, not on the residual: `Float64`
# CG overshoots its own `reltol` by the same factor `Float32` does, so no ratio
# test could separate them, and refining a `Float64` solve buys nothing.
@testset "refinement is Float32-only" begin
    @test Tortuosity._refines_by_default(Float32)
    @test !Tortuosity._refines_by_default(Float64)
end
