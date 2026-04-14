# Tests for transient diffusion functionality via the TransientDiffusionProblem public API
using Test
using SparseArrays
using LinearAlgebra: norm, tr
using Tortuosity
using Tortuosity:
    Imaginator,
    apply_boundaries!,
    find_boundary_nodes,
    slice_concentration,
    slab_concentration,
    slab_mass_uptake,
    slab_flux,
    slab_cumulative_flux

# --- Analytical solutions (pure math, no ODE solve) ---

@testset "slab_concentration — boundary conditions" begin
    # c(x=0, t) = c1 = 1 and c(x=L, t) = c2 = 0 for any t
    for t in [0.01, 0.1, 1.0, 10.0]
        @test slab_concentration(1.0, 0.0, t) ≈ 1.0 atol = 1e-8
        @test slab_concentration(1.0, 1.0, t) ≈ 0.0 atol = 1e-8
    end
end

@testset "slab_concentration — steady state is linear" begin
    # At large t, c(x) → c1 + (c2 - c1)·x/L = 1 - x
    for x in [0.2, 0.3, 0.5, 0.7, 0.9]
        @test slab_concentration(1.0, x, 10.0) ≈ (1.0 - x) atol = 1e-8
    end
end

@testset "slab_concentration — D scaling" begin
    # Higher D reaches steady state faster: |c - c_ss| should be smaller
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
    # At steady state, flux = D·(c1 - c2)/L = 1.0 everywhere
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
    ts = collect(range(0.01, 5.0; length=50))
    Qs = slab_cumulative_flux(1.0, ts)
    @test all(diff(Qs) .>= 0)
    Q1 = slab_cumulative_flux(1.0, 100.0)
    Q2 = slab_cumulative_flux(1.0, 101.0)
    @test (Q2 - Q1) ≈ 1.0 atol = 1e-4
end

# --- TransientDiffusionProblem construction ---

blob_8 = BitArray(Imaginator.blobs(; shape=(8, 8, 8), porosity=0.5, blobiness=1, seed=42))

@testset "TransientDiffusionProblem — default parameters" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false)
    @test prob.axis == :z
    @test prob.bc_inlet == Float32(1)
    @test prob.bc_outlet == Float32(0)
    @test size(prob.A, 1) == count(blob_8)
end

@testset "TransientDiffusionProblem — custom parameters" begin
    prob = TransientDiffusionProblem(blob_8; axis=:x, bc_inlet=2, bc_outlet=0, D=0.5, dtype=Float64, gpu=false)
    @test prob.axis == :x
    @test prob.bc_inlet == 2.0
    @test prob.D == 0.5
end

@testset "TransientDiffusionProblem — insulated outlet" begin
    prob = TransientDiffusionProblem(blob_8; bc_inlet=1, bc_outlet=nothing, gpu=false)
    @test isnothing(prob.bc_outlet)
end

@testset "TransientDiffusionProblem — time-dependent boundary" begin
    f_inlet = t -> sin(2π * t)
    prob = TransientDiffusionProblem(blob_8; bc_inlet=f_inlet, bc_outlet=0, gpu=false)
    @test prob.bc_inlet isa Function
    @test prob.bc_outlet == Float32(0)
end

@testset "TransientDiffusionProblem — show" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false)
    io = IOBuffer()
    show(io, prob)
    s = String(take!(io))
    @test occursin("TransientDiffusionProblem", s)
    @test occursin("shape=(8, 8, 8)", s)
    @test occursin("axis=z", s)
end

# --- Operator structure ---

@testset "Operator — Dirichlet rows are zeroed" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=0, gpu=false, dtype=Float64)
    A = prob.A
    bc_nodes = find_boundary_nodes(prob.img, :bottom)
    append!(bc_nodes, find_boundary_nodes(prob.img, :top))
    for node in bc_nodes
        @test all(A[node, :] .== 0)
    end
end

@testset "Operator — insulated boundary rows are NOT zeroed" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=nothing, gpu=false, dtype=Float64)
    A = prob.A
    outlet_nodes = find_boundary_nodes(prob.img, :top)
    for node in outlet_nodes
        @test !all(A[node, :] .== 0)
    end
end

# --- Boundary application ---

@testset "apply_boundaries! — Dirichlet on both faces" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=0, gpu=false, dtype=Float64)
    c0 = zeros(Float64, size(img))
    apply_boundaries!(c0, prob)
    @test all(c0[:, :, 1] .== 1.0)
    @test all(c0[:, :, 4] .== 0.0)
end

@testset "apply_boundaries! — insulated outlet" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=nothing, gpu=false, dtype=Float64)
    c0 = zeros(Float64, size(img))
    apply_boundaries!(c0, prob)
    @test all(c0[:, :, 1] .== 1.0)
    @test all(c0[:, :, 4] .== 0.0)  # untouched
end

@testset "apply_boundaries! — time-dependent inlet" begin
    img = ones(Bool, (4, 4, 4))
    f_inlet = t -> 0.5 + t
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=f_inlet, bc_outlet=0, gpu=false, dtype=Float64)
    c0 = zeros(Float64, size(img))
    apply_boundaries!(c0, prob)
    @test all(c0[:, :, 1] .== 0.5)  # f(0) = 0.5
    @test all(c0[:, :, 4] .== 0.0)
end

# --- Solve + TransientSolution ---

@testset "solve — basic time progression" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false, dtype=Float64)
    sol = solve(prob, ROCK4(); saveat=0.1, tspan=(0.0, 0.5))
    @test sol.t[end] >= 0.5
    @test length(sol.t) > 1
    @test length(sol.u) == length(sol.t)
    @test all(length(u) == count(blob_8) for u in sol.u)
end

@testset "solve — sol.u is CPU even under GPU-style code path" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false, dtype=Float64)
    sol = solve(prob, ROCK4(); saveat=0.1, tspan=(0.0, 0.5))
    @test all(u isa Vector{Float64} for u in sol.u)
end

@testset "TransientSolution — show" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false, dtype=Float64)
    sol = solve(prob, ROCK4(); saveat=0.2, tspan=(0.0, 0.5))
    io = IOBuffer()
    show(io, sol)
    s = String(take!(io))
    @test occursin("TransientSolution", s)
    @test occursin("snapshots=", s)
    @test occursin("retcode=", s)
end

# --- Stop conditions ---

@testset "StopAtSteadyState — terminates on small du/dt" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false, dtype=Float64)
    sol = solve(prob, ROCK4();
        saveat=0.1,
        callback=StopAtSteadyState(abstol=1e-4, reltol=1e-3),
        tspan=(0.0, 20.0))
    @test sol.retcode == :Terminated
    @test sol.t[end] < 20.0  # terminated before hitting the end
end

@testset "StopAtSaturation — terminates at target mean" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false, dtype=Float64)
    target = 0.3
    sol = solve(prob, ROCK4();
        saveat=0.05,
        callback=StopAtSaturation(target),
        tspan=(0.0, 20.0))
    @test sol.retcode == :Terminated
    mean_final = sum(sol.u[end]) / length(sol.u[end])
    @test mean_final >= target - 1e-2  # rootfinding + default reltol gives ~1e-3 slack
end

@testset "StopAtFluxBalance — terminates near steady state" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false, dtype=Float64)
    sol = solve(prob, ROCK4();
        saveat=0.1,
        callback=StopAtFluxBalance(prob; abstol=0.05),
        tspan=(0.0, 20.0))
    @test sol.retcode == :Terminated
    # At termination, inlet and outlet flux should agree within tolerance
    c_final = sol.u[end]
    j_in = flux(c_final, prob.D, prob.voxel_size, prob.img, prob.axis; ind=1, pore_index=prob.pore_index)
    j_out = flux(c_final, prob.D, prob.voxel_size, prob.img, prob.axis; ind=:end, pore_index=prob.pore_index)
    @test abs(j_in - j_out) <= 0.05
end

@testset "StopAtPeriodicState — detects periodic steady state" begin
    img = trues(1, 1, 16)
    freq = 0.5
    prob = TransientDiffusionProblem(img;
        axis=:z,
        bc_inlet=t -> (sin(2π * freq * t) + 1) / 2,
        bc_outlet=nothing,
        gpu=false, dtype=Float64)
    sol = solve(prob, ROCK4();
        saveat=0.05,
        callback=StopAtPeriodicState(freq, prob; reltol=1e-3),
        tspan=(0.0, 100.0),
        u0=fill(0.5, 1, 1, 16))
    @test sol.retcode == :Terminated
    @test sol.t[end] < 100.0
end

# --- Composed callbacks ---

@testset "CallbackSet — compose stop condition with tspan cap" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false, dtype=Float64)
    # Extremely tight tolerance so the callback shouldn't fire before tspan end
    sol = solve(prob, ROCK4();
        saveat=0.1,
        callback=StopAtSteadyState(abstol=1e-20, reltol=1e-20),
        tspan=(0.0, 0.5))
    @test sol.retcode == :Success  # hit tspan[2] first
    @test sol.t[end] >= 0.4
end

# --- End-to-end open space ---

open_16 = ones(Bool, (16, 16, 16))

@testset "Open space transient — $(ax)-axis" for ax in (:x, :y, :z)
    prob = TransientDiffusionProblem(open_16;
        axis=ax, bc_inlet=1, bc_outlet=0, gpu=false, dtype=Float64)
    sol = solve(prob, ROCK4();
        saveat=0.05,
        callback=StopAtFluxBalance(prob; abstol=0.01),
        tspan=(0.0, 50.0))
    @test sol.retcode == :Terminated
    c_final = sol.u[end]
    mid = slice_concentration(c_final, prob.img, prob.axis, size(open_16, 1) ÷ 2; pore_index=prob.pore_index)
    @test mid ≈ 0.5 atol = 0.05
    j_in = flux(c_final, prob.D, prob.voxel_size, prob.img, prob.axis; ind=1, pore_index=prob.pore_index)
    j_out = flux(c_final, prob.D, prob.voxel_size, prob.img, prob.axis; ind=:end, pore_index=prob.pore_index)
    @test j_in ≈ j_out atol = 0.02
end

# --- Fitting effective diffusivity ---
#
# These are smoke tests for the TransientSolution adapter method of
# fit_effective_diffusivity. Tolerances are loose because the fit quality on a
# small 16³ open-space cube is limited by (1) the discrete grid resolution and
# (2) a known mismatch between the :mass discretisation and the continuous
# `slab_mass_uptake` analytical — the simulation subtracts the initial face
# contribution from mass_uptake while the analytical does not. Strict
# effective-diffusivity accuracy tests belong on a larger image anyway, so we
# just verify the wrapper plumbing works end-to-end here.

@testset "fit_effective_diffusivity — TransientSolution wrapper" begin
    prob = TransientDiffusionProblem(open_16;
        axis=:z, bc_inlet=1, bc_outlet=0, gpu=false, dtype=Float64)
    sol = solve(prob, ROCK4();
        saveat=0.02,
        callback=StopAtFluxBalance(prob; abstol=0.001),
        tspan=(0.0, 50.0))

    # All three methods should return finite, positive results
    for method in (:mass, :conc, :flux)
        τ, D_eff, xdata, ydata, fit, model = fit_effective_diffusivity(sol, prob, method; depth=0.5)
        @test isfinite(τ) && τ > 0
        @test isfinite(D_eff) && D_eff > 0
        @test length(xdata) == length(ydata)
        @test length(xdata) > 0
    end

    # :flux is the tightest method for this setup; check the value itself
    _, D_eff_flux, _, _, _, _ = fit_effective_diffusivity(sol, prob, :flux; depth=0.5)
    @test D_eff_flux ≈ 1.0 atol = 0.25
end
