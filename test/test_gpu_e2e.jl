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
using Tortuosity: PortableSparseCSC, Imaginator, _on_gpu

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

    prob = TransientDiffusionProblem(img; axis=:z, gpu=true, dtype=Float32)
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
end

@testset "TransientDiffusionProblem CPU/GPU parity (scalar snapshot)" begin
    img = Array{Bool}(
        Imaginator.blobs(; shape=(24, 24, 24), porosity=0.6f0, blobiness=1, seed=11)
    )
    (any(img[:, :, 1]) && any(img[:, :, end])) || return

    prob_cpu = TransientDiffusionProblem(img; axis=:z, gpu=false, dtype=Float64)
    sol_cpu = solve(prob_cpu, ROCK4(); saveat=0.05, tspan=(0.0, 0.15))
    c_mean_cpu = sum(sol_cpu.u[end]) / length(sol_cpu.u[end])

    prob_gpu = TransientDiffusionProblem(img; axis=:z, gpu=true, dtype=Float32)
    sol_gpu = solve(prob_gpu, ROCK4(); saveat=0.05, tspan=(0.0, 0.15))
    c_mean_gpu = sum(sol_gpu.u[end]) / length(sol_gpu.u[end])

    # Different integrator tolerances + Float32/Float64 → loose check
    @test isapprox(c_mean_cpu, c_mean_gpu; atol=1e-2)
end
