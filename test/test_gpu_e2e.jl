# End-to-end GPU pipeline tests. test_gpu_parity.jl stops at assembled
# matrices; this file exercises the full path — build with gpu=true, solve,
# reshape via vec_to_grid, compute tortuosity — and verifies the final value
# matches analytical expectations or the CPU run. Catches bugs where every
# intermediate is correct but the struct has a stale device array, or a
# postprocessing helper can't cope with a GPU mask.
#
# Backend-agnostic: caller (runtests.jl) must ensure *some* GPU backend is
# loaded and functional (CUDA on Linux x64, Metal on macOS arm64, etc.).
# We never reference a concrete device array type here.

using Test
using Tortuosity
using Tortuosity: PortableSparseCSC, Imaginator, _on_gpu

# ---------------------------------------------------------------------------
# Steady-state
# ---------------------------------------------------------------------------

@testset "open space $(n)^3 · axis=$(ax)" for n in (16, 24), ax in (:x, :y, :z)
    img = ones(Bool, n, n, n)
    sim = TortuositySimulation(img; axis=ax, gpu=true)

    # Invariant: .img stays on the host even when gpu=true, otherwise every
    # CPU-only postprocessing helper downstream is broken.
    @test sim.img isa Array{Bool,3}
    @test sim.prob.A isa PortableSparseCSC

    sol = solve(sim.prob, KrylovJL_CG(); reltol=1.0f-6)
    c_grid = vec_to_grid(sol.u, sim.img)
    @test tortuosity(c_grid, sim.img; axis=ax) ≈ 1.0 atol = 1e-3
end

@testset "half-channel 16^3 (x-axis)" begin
    img = ones(Bool, 16, 16, 16)
    img[:, :, 1:8] .= false
    sim = TortuositySimulation(img; axis=:x, gpu=true)
    sol = solve(sim.prob, KrylovJL_CG(); reltol=1.0f-6)
    c_grid = vec_to_grid(sol.u, sim.img)
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

    sim_cpu = TortuositySimulation(img; axis=:x, gpu=false)
    sol_cpu = solve(sim_cpu.prob, KrylovJL_CG(); reltol=1.0e-8)
    tau_cpu = tortuosity(vec_to_grid(sol_cpu.u, sim_cpu.img), sim_cpu.img; axis=:x)

    sim_gpu = TortuositySimulation(img; axis=:x, gpu=true)
    sol_gpu = solve(sim_gpu.prob, KrylovJL_CG(); reltol=1.0f-6)
    tau_gpu = tortuosity(vec_to_grid(sol_gpu.u, sim_gpu.img), sim_gpu.img; axis=:x)

    # Float32 vs Float64, same geometry → loose rtol absorbs the precision gap
    @test isfinite(tau_gpu)
    @test tau_gpu > 1
    @test tau_cpu ≈ tau_gpu rtol = 1e-3
end

# ---------------------------------------------------------------------------
# Transient (closes the gap noted in docs/design.md § open issues)
# ---------------------------------------------------------------------------

@testset "TransientProblem + solve! end-to-end on GPU" begin
    img = Array{Bool}(
        Imaginator.blobs(; shape=(24, 24, 24), porosity=0.6f0, blobiness=1, seed=7)
    )
    (any(img[:, :, 1]) && any(img[:, :, end])) || return

    prob = TransientProblem(img, 0.05; axis=:z, gpu=true, dtype=Float32)
    @test prob.img isa AbstractArray{Bool}
    @test !_on_gpu(prob.img)
    @test prob.A isa PortableSparseCSC

    state = init_state(prob)
    @test length(state.t) == 1
    @test state.t[1] == 0.0
    @test length(state.C[1]) == count(prob.img)
    @test all(isfinite, state.C[1])

    solve!(state, prob, stop_at_time(0.2))
    @test state.t[end] >= 0.2
    @test length(state.C) == length(state.t)
    @test all(all(isfinite, C) for C in state.C)
end

@testset "TransientProblem CPU/GPU parity (scalar snapshot)" begin
    img = Array{Bool}(
        Imaginator.blobs(; shape=(24, 24, 24), porosity=0.6f0, blobiness=1, seed=11)
    )
    (any(img[:, :, 1]) && any(img[:, :, end])) || return

    prob_cpu = TransientProblem(img, 0.05; axis=:z, gpu=false, dtype=Float64)
    state_cpu = init_state(prob_cpu)
    solve!(state_cpu, prob_cpu, stop_at_time(0.15))
    c_mean_cpu = sum(state_cpu.C[end]) / length(state_cpu.C[end])

    prob_gpu = TransientProblem(img, 0.05; axis=:z, gpu=true, dtype=Float32)
    state_gpu = init_state(prob_gpu)
    solve!(state_gpu, prob_gpu, stop_at_time(0.15))
    c_mean_gpu = sum(state_gpu.C[end]) / length(state_gpu.C[end])

    # Different integrator tolerances + Float32/Float64 → loose check
    @test isapprox(c_mean_cpu, c_mean_gpu; atol=1e-2)
end
