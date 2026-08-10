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
using Tortuosity
using Tortuosity: PortableSparseCSC, Imaginator, _on_gpu, _gpu_adapt, reconstruct_slice

# ---------------------------------------------------------------------------
# Steady-state
# ---------------------------------------------------------------------------

@testset "open space $(n)^3 · axis=$(ax)" for n in (16, 24), ax in (:x, :y, :z)
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

# Regression: small-box GPU runs once produced τ ≈ 0.73 instead of 1.0 on
# Metal because histogram_connections_kernel! interleaved a per-bucket atomic
# with a shared-counter atomic, and the latter silently lost updates under
# contention. The undercount only matters in absolute terms (~24 missing
# entries in the connectivity list), so the existing 16³/24³ tests above
# absorbed it within atol=1e-3 — the bug only became visible once the box
# was small enough that 24 lost edges was a meaningful fraction of total.
@testset "open space $(n)^3 (small-box atomic regression) · axis=$(ax)" for n in (4, 6),
    ax in (:x, :y, :z)

    img = ones(Bool, n, n, n)
    sim = SteadyDiffusionProblem(img; axis=ax, gpu=true)
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

@testset "CPU/GPU parity on blobs (seed=$(seed))" for seed in (1, 42, 100)
    img = Array{Bool}(
        Imaginator.blobs(; shape=(32, 32, 32), porosity=0.55f0, blobiness=1, seed=seed)
    )
    # Skip degenerate images that don't connect inlet to outlet
    (any(img[1, :, :]) && any(img[end, :, :])) || return

    sim_cpu = SteadyDiffusionProblem(img; axis=:x, gpu=false)
    sol_cpu = solve(sim_cpu.prob, KrylovJL_CG(); reltol=1.0e-8)
    tau_cpu = tortuosity(reconstruct_field(sol_cpu.u, sim_cpu.img), sim_cpu.img; axis=:x)

    sim_gpu = SteadyDiffusionProblem(img; axis=:x, gpu=true)
    sol_gpu = solve(sim_gpu.prob, KrylovJL_CG(); reltol=1.0f-6)
    tau_gpu = tortuosity(reconstruct_field(sol_gpu.u, sim_gpu.img), sim_gpu.img; axis=:x)

    # Float32 vs Float64, same geometry → loose rtol absorbs the precision gap
    @test isfinite(tau_gpu)
    @test tau_gpu > 1
    @test tau_cpu ≈ tau_gpu rtol = 1e-3
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
    tau_cpu = tortuosity(reconstruct_field(sol_cpu.u, sim_cpu.img), sim_cpu.img; axis=:x)

    sim_gpu = SteadyDiffusionProblem(img; axis=:x, gpu=true, D=D)
    sol_gpu = solve(sim_gpu.prob, KrylovJL_CG(); reltol=1.0f-6)
    tau_gpu = tortuosity(reconstruct_field(sol_gpu.u, sim_gpu.img), sim_gpu.img; axis=:x)

    @test isfinite(tau_gpu)
    @test tau_gpu > 1
    @test tau_cpu ≈ tau_gpu rtol = 2e-3
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
