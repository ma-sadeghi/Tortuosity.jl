# Tests for transient diffusion functionality via the TransientProblem public API
using Test
using SparseArrays
using LinearAlgebra: norm, tr
using Tortuosity
using Tortuosity:
    Imaginator,
    apply_boundaries!,
    find_boundary_nodes,
    slab_concentration,
    slab_mass_uptake,
    slab_flux,
    slab_cumulative_flux

# --- Analytical solutions (pure math, no ODE solve) ---

@testset "slab_concentration — boundary conditions" begin
    # C(x=0, t) = C1 = 1 and C(x=L, t) = C2 = 0 for any t
    for t in [0.01, 0.1, 1.0, 10.0]
        @test slab_concentration(1.0, 0.0, t) ≈ 1.0 atol = 1e-8
        @test slab_concentration(1.0, 1.0, t) ≈ 0.0 atol = 1e-8
    end
end

@testset "slab_concentration — steady state is linear" begin
    # At large t, C(x) → C1 + (C2-C1)*x/L = 1 - x
    for x in [0.2, 0.3, 0.5, 0.7, 0.9]
        @test slab_concentration(1.0, x, 10.0) ≈ (1.0 - x) atol = 1e-8
    end
end

@testset "slab_concentration — D scaling" begin
    # Higher D reaches steady state faster: |C - C_ss| should be smaller
    c_slow = slab_concentration(0.1, 0.5, 0.5)
    c_fast = slab_concentration(10.0, 0.5, 0.5)
    @test abs(c_fast - 0.5) < abs(c_slow - 0.5)
end

@testset "slab_concentration — vector of times" begin
    ts = [0.1, 0.5, 1.0, 5.0]
    result = slab_concentration(1.0, 0.5, ts)
    @test length(result) == 4
    # Last time point should be near steady state
    @test result[end] ≈ 0.5 atol = 1e-6
end

@testset "slab_mass_uptake — limits" begin
    # Series converges slowly at t=0; with 100 terms the residual is ~0.002
    @test slab_mass_uptake(1.0, 0.0) ≈ 0.0 atol = 0.01
    @test slab_mass_uptake(1.0, 10.0) ≈ 1.0 atol = 1e-6
    @test slab_mass_uptake(1.0, 100.0) ≈ 1.0 atol = 1e-8
end

@testset "slab_mass_uptake — monotonically increasing" begin
    ts = collect(range(0.0, 5.0; length=50))
    ms = slab_mass_uptake(1.0, ts)
    @test all(diff(ms) .>= 0)
end

@testset "slab_flux — steady state" begin
    # At steady state, flux = D*(C1-C2)/L = 1.0 everywhere
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]
        @test slab_flux(1.0, x, 10.0) ≈ 1.0 atol = 1e-6
    end
end

@testset "slab_flux — zero at t=0" begin
    @test slab_flux(1.0, 0.5, 0.0) == 0.0
    ts = [0.0, 0.1, 1.0]
    result = slab_flux(1.0, 0.5, ts)
    @test result[1] == 0.0
end

@testset "slab_cumulative_flux — cumulative" begin
    @test slab_cumulative_flux(1.0, 0.0) ≈ 0.0 atol = 1e-8
    # Monotonically increasing
    ts = collect(range(0.01, 5.0; length=50))
    Qs = slab_cumulative_flux(1.0, ts)
    @test all(diff(Qs) .>= 0)
    # At large t, slope approaches D*(C1-C2)/L = 1.0
    Q1 = slab_cumulative_flux(1.0, 100.0)
    Q2 = slab_cumulative_flux(1.0, 101.0)
    @test (Q2 - Q1) ≈ 1.0 atol = 1e-4
end

# --- TransientProblem construction ---

blob_8 = BitArray(Imaginator.blobs(; shape=(8, 8, 8), porosity=0.5, blobiness=1, seed=42))

@testset "TransientProblem — default parameters" begin
    prob = TransientProblem(blob_8, 0.1; gpu=false)
    @test prob.axis == :z
    @test prob.bc_inlet == Float32(1)
    @test prob.bc_outlet == Float32(0)
    @test size(prob.A, 1) == count(blob_8)
    @test prob.dt == 0.1
end

@testset "TransientProblem — custom parameters" begin
    prob = TransientProblem(blob_8, 0.05; axis=:x, bc_inlet=2, bc_outlet=0, D=0.5, dtype=Float64, gpu=false)
    @test prob.axis == :x
    @test prob.bc_inlet == 2.0
    @test prob.D == 0.5
end

@testset "TransientProblem — insulated outlet" begin
    prob = TransientProblem(blob_8, 0.1; bc_inlet=1, bc_outlet=nothing, gpu=false)
    @test isnothing(prob.bc_outlet)
end

@testset "TransientProblem — time-dependent boundary" begin
    f_inlet = t -> sin(2π * t)
    prob = TransientProblem(blob_8, 0.1; bc_inlet=f_inlet, bc_outlet=0, gpu=false)
    @test prob.bc_inlet isa Function
    @test prob.bc_outlet == Float32(0)
end

# --- Operator structure ---

@testset "Operator — Dirichlet rows are zeroed" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientProblem(img, 0.1; axis=:z, bc_inlet=1, bc_outlet=0, gpu=false, dtype=Float64)
    A = prob.A
    bc_nodes = Tortuosity.find_boundary_nodes(prob.img, :bottom)
    append!(bc_nodes, Tortuosity.find_boundary_nodes(prob.img, :top))
    for node in bc_nodes
        @test all(A[node, :] .== 0)
    end
end

@testset "Operator — insulated boundary rows are NOT zeroed" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientProblem(img, 0.1; axis=:z, bc_inlet=1, bc_outlet=nothing, gpu=false, dtype=Float64)
    A = prob.A
    outlet_nodes = Tortuosity.find_boundary_nodes(prob.img, :top)
    for node in outlet_nodes
        @test !all(A[node, :] .== 0)
    end
end

# --- Boundary application ---

@testset "apply_boundaries! — Dirichlet on both faces" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientProblem(img, 0.1; axis=:z, bc_inlet=1, bc_outlet=0, gpu=false, dtype=Float64)
    C0 = zeros(Float64, size(img))
    apply_boundaries!(C0, prob)
    @test all(C0[:, :, 1] .== 1.0)
    @test all(C0[:, :, 4] .== 0.0)
end

@testset "apply_boundaries! — insulated outlet" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientProblem(img, 0.1; axis=:z, bc_inlet=1, bc_outlet=nothing, gpu=false, dtype=Float64)
    C0 = zeros(Float64, size(img))
    apply_boundaries!(C0, prob)
    @test all(C0[:, :, 1] .== 1.0)
    @test all(C0[:, :, 4] .== 0.0) # untouched
end

@testset "apply_boundaries! — time-dependent inlet" begin
    img = ones(Bool, (4, 4, 4))
    f_inlet = t -> 0.5 + t
    prob = TransientProblem(img, 0.1; axis=:z, bc_inlet=f_inlet, bc_outlet=0, gpu=false, dtype=Float64)
    C0 = zeros(Float64, size(img))
    apply_boundaries!(C0, prob)
    @test all(C0[:, :, 1] .== 0.5) # f(0) = 0.5
    @test all(C0[:, :, 4] .== 0.0)
end

# --- init_state / solve! ---

@testset "init_state — produces valid state" begin
    prob = TransientProblem(blob_8, 0.1; gpu=false, dtype=Float64)
    state = init_state(prob)
    @test length(state.t) == 1
    @test state.t[1] == 0.0
    @test length(state.C) == 1
    @test length(state.C[1]) == count(blob_8)
end

@testset "solve! — time progresses" begin
    prob = TransientProblem(blob_8, 0.1; gpu=false, dtype=Float64)
    state = init_state(prob)
    solve!(state, prob, stop_at_time(0.5))
    @test state.t[end] >= 0.5
    @test length(state.t) > 1
    @test length(state.C) == length(state.t)
end

@testset "solve! — max_iter warning" begin
    prob = TransientProblem(blob_8, 0.1; gpu=false, dtype=Float64)
    state = init_state(prob)
    @test_logs (:warn, r"max_iter") solve!(state, prob, (t, C) -> false; max_iter=3)
    @test length(state.t) == 4 # initial + 3 steps
end

# --- Stop conditions ---

@testset "stop_at_time" begin
    cond = stop_at_time(1.0)
    @test cond([0.5], []) == false
    @test cond([1.0], []) == true
    @test cond([1.5], []) == true
end

@testset "stop_at_avg_concentration" begin
    img = ones(Bool, (4, 4, 4))
    cond = Tortuosity.stop_at_avg_concentration(0.5, img)
    dummy_C_low = [fill(0.3, count(img))]
    dummy_C_high = [fill(0.6, count(img))]
    @test cond([0.0], dummy_C_low) == false
    @test cond([0.0], dummy_C_high) == true
end

#=
open_16 = ones(Bool, (16, 16, 16))

@testset "Open space transient — $(ax)-axis" for ax in (:x, :y)
    prob = TransientProblem(open_16, 0.05; axis=ax, bc_inlet=1, bc_outlet=0, gpu=false, dtype=Float64)
    state = init_state(prob)
    solve!(state, prob, stop_at_delta_flux(0.01, prob); max_iter=500)
    C_final = state.C[end]
    mid = get_slice_conc(C_final, prob.img, prob.axis, size(open_16, 1) ÷ 2; grid_to_vec=prob.grid_to_vec)
    @test mid ≈ 0.5 atol = 0.05
    J_in = compute_flux(C_final, prob.D, prob.dx, prob.img, prob.axis; ind=1, grid_to_vec=prob.grid_to_vec)
    J_out = compute_flux(C_final, prob.D, prob.dx, prob.img, prob.axis; ind=:end, grid_to_vec=prob.grid_to_vec)
    @test J_in ≈ J_out atol = 0.02
end

prob_z = TransientProblem(open_16, 0.05; axis=:z, bc_inlet=1, bc_outlet=0, gpu=false, dtype=Float64)

@testset "Open space transient — z-axis" begin
    state = init_state(prob_z)
    solve!(state, prob_z, stop_at_delta_flux(0.01, prob_z); max_iter=500)
    C_final = state.C[end]
    mid = get_slice_conc(C_final, prob_z.img, prob_z.axis, size(open_16, 3) ÷ 2; grid_to_vec=prob_z.grid_to_vec)
    @test mid ≈ 0.5 atol = 0.05
    J_in = compute_flux(C_final, prob_z.D, prob_z.dx, prob_z.img, prob_z.axis; ind=1, grid_to_vec=prob_z.grid_to_vec)
    J_out = compute_flux(C_final, prob_z.D, prob_z.dx, prob_z.img, prob_z.axis; ind=:end, grid_to_vec=prob_z.grid_to_vec)
    @test J_in ≈ J_out atol = 0.02
end

@testset "Stop conditions" begin
    state_t = init_state(prob_z)
    solve!(state_t, prob_z, stop_at_time(1.0); max_iter=100)
    @test state_t.t[end] >= 1.0

    state_c = init_state(prob_z)
    cond = Tortuosity.stop_at_avg_concentration(0.4, prob_z)
    solve!(state_c, prob_z, cond; max_iter=500)
    avg = sum(state_c.C[end]) / length(state_c.C[end])
    @test avg >= 0.4

    prob_insulated = TransientProblem(open_16, 0.1; axis=:z, bc_inlet=1, bc_outlet=nothing, gpu=false, dtype=Float64)
    state = init_state(prob_insulated)
    solve!(state, prob, stop_at_time(3.0); max_iter=100)
    avg = sum(state.C[end]) / length(state.C[end])
    @test avg ≈ 1.0 atol = 0.1
end

@testset verbose = true "fit_effective_diffusivity — open space" begin
    state_fit = init_state(prob_z)
    solve!(state_fit, prob_z, stop_at_delta_flux(0.01, prob_z); max_iter=500)

    @testset "method = :mass" begin
        τ, D_eff, _, _, _, _ = fit_effective_diffusivity(state_fit, prob_z, :mass)
        @test τ ≈ 1.0 atol = 0.1
        @test D_eff ≈ 1.0 atol = 0.1
    end

    @testset "method = :conc" begin
        τ, D_eff, _, _, _, _ = fit_effective_diffusivity(state_fit, prob_z, :conc; depth=0.5)
        @test D_eff ≈ 1.0 atol = 0.1
    end

    @testset "method = :flux" begin
        τ, D_eff, _, _, _, _ = fit_effective_diffusivity(state_fit, prob_z, :flux; depth=0.5)
        @test D_eff ≈ 1.0 atol = 0.25
    end
end
=#
