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
    prob = TransientDiffusionProblem(blob_8; axis=:x, bc_inlet=2, bc_outlet=0, D=0.5, gpu=false)
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
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=0, gpu=false)
    A = prob.A
    bc_nodes = find_boundary_nodes(prob.img, :bottom)
    append!(bc_nodes, find_boundary_nodes(prob.img, :top))
    for node in bc_nodes
        @test all(A[node, :] .== 0)
    end
end

@testset "Operator — insulated boundary rows are NOT zeroed" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=nothing, gpu=false)
    A = prob.A
    outlet_nodes = find_boundary_nodes(prob.img, :top)
    for node in outlet_nodes
        @test !all(A[node, :] .== 0)
    end
end

# --- Boundary application ---

@testset "apply_boundaries! — Dirichlet on both faces" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=0, gpu=false)
    c0 = zeros(Float64, size(img))
    apply_boundaries!(c0, prob)
    @test all(c0[:, :, 1] .== 1.0)
    @test all(c0[:, :, 4] .== 0.0)
end

@testset "apply_boundaries! — insulated outlet" begin
    img = ones(Bool, (4, 4, 4))
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=nothing, gpu=false)
    c0 = zeros(Float64, size(img))
    apply_boundaries!(c0, prob)
    @test all(c0[:, :, 1] .== 1.0)
    @test all(c0[:, :, 4] .== 0.0)  # untouched
end

@testset "apply_boundaries! — time-dependent inlet" begin
    img = ones(Bool, (4, 4, 4))
    f_inlet = t -> 0.5 + t
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=f_inlet, bc_outlet=0, gpu=false)
    c0 = zeros(Float64, size(img))
    apply_boundaries!(c0, prob)
    @test all(c0[:, :, 1] .== 0.5)  # f(0) = 0.5
    @test all(c0[:, :, 4] .== 0.0)
end

# --- Solve + TransientSolution ---

@testset "solve — basic time progression" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false)
    sol = solve(prob, ROCK4(); saveat=0.1, tspan=(0.0, 0.5))
    @test sol.t[end] >= 0.5
    @test length(sol.t) > 1
    @test length(sol.u) == length(sol.t)
    @test all(length(u) == count(blob_8) for u in sol.u)
end

@testset "solve — sol.u is CPU even under GPU-style code path" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false)
    sol = solve(prob, ROCK4(); saveat=0.1, tspan=(0.0, 0.5))
    @test all(u isa Vector{Float64} for u in sol.u)
end

@testset "TransientSolution — show" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false)
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
    prob = TransientDiffusionProblem(blob_8; gpu=false)
    sol = solve(prob, ROCK4();
        saveat=0.1,
        callback=StopAtSteadyState(abstol=1e-4, reltol=1e-3),
        tspan=(0.0, 20.0))
    @test sol.retcode == :Terminated
    @test sol.t[end] < 20.0  # terminated before hitting the end
end

@testset "StopAtSaturation — terminates at target mean" begin
    prob = TransientDiffusionProblem(blob_8; gpu=false)
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
    prob = TransientDiffusionProblem(blob_8; gpu=false)
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
        gpu=false)
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
    prob = TransientDiffusionProblem(blob_8; gpu=false)
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
        axis=ax, bc_inlet=1, bc_outlet=0, gpu=false)
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
# Smoke tests for the TransientSolution adapter + accuracy tests on a
# fully-open slab. For a homogeneous open slab the discrete FD solver is
# solving exactly the same PDE as the analytical slab solutions, so
# fit_effective_diffusivity must recover D_eff = D_pore to numerical
# precision on a sufficiently refined grid — :conc and :flux over the full
# trajectory, :mass over a late-time window that skips the early-time
# (t ≲ dx²/D) finite-boundary-cell discretization artifact (see
# fit_effective_diffusivity docstring).

@testset "fit_effective_diffusivity — TransientSolution wrapper" begin
    prob = TransientDiffusionProblem(open_16;
        axis=:z, bc_inlet=1, bc_outlet=0, gpu=false)
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
end

@testset "fit_effective_diffusivity — fully-open slab recovers D to machine precision" begin
    # A 1-voxel-wide "slab" in z with fine axial resolution. Every voxel is
    # pore, so the discrete solver is exactly the continuous 1D slab and
    # fit_effective_diffusivity with :conc or :flux must recover D = 1 to
    # within ODE-integrator precision. This would fail by O(dx/L) if the
    # depth_actual ↔ voxel_index map were off by half a voxel.
    N = 65
    img = trues(1, 1, N)
    prob = TransientDiffusionProblem(img;
        axis=:z, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
    sol = solve(prob, ROCK4();
        saveat=0.005,
        callback=StopAtFluxBalance(prob; abstol=1e-5),
        tspan=(0.0, 5.0),
        reltol=1e-8, abstol=1e-10)

    for depth in (0.25, 0.5, 0.75)
        τ_c, D_eff_c, _, _, _, _ = fit_effective_diffusivity(sol, prob, :conc; depth=depth)
        @test D_eff_c ≈ 1.0 atol = 1e-6
        @test τ_c    ≈ 1.0 atol = 1e-6

        τ_f, D_eff_f, _, _, _, _ = fit_effective_diffusivity(sol, prob, :flux; depth=depth)
        @test D_eff_f ≈ 1.0 atol = 1e-5
        @test τ_f    ≈ 1.0 atol = 1e-5
    end
end

@testset "fit_effective_diffusivity — depth endpoints hit Dirichlet values" begin
    # Pin the analytical ↔ discrete coordinate mapping at the boundaries:
    # voxel 1 must sit at x=0 (where slab_concentration returns c1) and
    # voxel N at x=L (where it returns c2). A half-voxel offset in
    # depth_actual would shift these by dx/2 and break the mapping.
    N = 33
    img = trues(1, 1, N)
    prob = TransientDiffusionProblem(img;
        axis=:z, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
    sol = solve(prob, ROCK4();
        saveat=0.01,
        callback=StopAtFluxBalance(prob; abstol=1e-5),
        tspan=(0.0, 5.0),
        reltol=1e-8, abstol=1e-10)

    # At depth=0 / depth=1 the ydata is literally the clamped value, so if
    # the model matches c1 / c2 at x=0 / x=L then the fit reduces to a
    # trivial constant and τ = 1 exactly.
    τ0, D0, _, _, _, _ = fit_effective_diffusivity(sol, prob, :conc; depth=0.0)
    @test D0 ≈ 1.0 atol = 1e-6

    τ1, D1, _, _, _, _ = fit_effective_diffusivity(sol, prob, :conc; depth=1.0)
    @test D1 ≈ 1.0 atol = 1e-6
end

@testset "fit_effective_diffusivity — :mass with auto-late t_fit default" begin
    # Regression test for the mass_uptake reference-state fix AND the
    # auto-late t_fit default for :mass. Before the fix, mass_uptake
    # subtracted sum(c_hist[1]) as reference, which included the Dirichlet-
    # clamped inlet face contribution and biased :mass fits by O(1/N).
    # After the fix, the prob-aware overload defaults to c0_total=0 and
    # fit_effective_diffusivity auto-picks a late t_fit window that skips
    # the early-time finite-cell discretization — users don't need to know
    # about either pathology to get accurate :mass fits.
    for N in (17, 33, 65, 129)
        img = trues(1, 1, N)
        prob = TransientDiffusionProblem(img;
            axis=:z, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
        sol = solve(prob, ROCK4();
            saveat=0.01,
            tspan=(0.0, 10.0),
            reltol=1e-12, abstol=1e-14)

        # No t_fit passed — should auto-pick a window past the first-
        # eigenmode timescale and recover D to effectively machine precision.
        τ, D_eff, xdata, _, _, _ = fit_effective_diffusivity(sol, prob, :mass)
        @test D_eff ≈ 1.0 atol = 1e-6
        @test τ    ≈ 1.0 atol = 1e-6
        # Sanity check that the window actually starts late
        @test xdata[1] >= 1.0
    end
end

@testset "fit_effective_diffusivity — t_fit auto-selection rules" begin
    # Pin the defaulting behavior of _resolve_t_fit through the public API:
    # - :conc / :flux → full trajectory
    # - :mass         → starts at min(1.5·L²/D_pore, 0.5·(t_end - t_start))
    # - Explicit t_fit always wins
    N = 33
    img = trues(1, 1, N)
    prob = TransientDiffusionProblem(img;
        axis=:z, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
    sol = solve(prob, ROCK4(); saveat=0.01, tspan=(0.0, 10.0),
                reltol=1e-10, abstol=1e-12)

    # :conc / :flux default → full trajectory
    _, _, xd_c, _, _, _ = fit_effective_diffusivity(sol, prob, :conc; depth=0.5)
    @test xd_c[1] == sol.t[1]
    @test xd_c[end] == sol.t[end]

    _, _, xd_f, _, _, _ = fit_effective_diffusivity(sol, prob, :flux; depth=0.5)
    @test xd_f[1] == sol.t[1]
    @test xd_f[end] == sol.t[end]

    # :mass default → starts at ~1.5·L²/D_pore = 1.5 for L=1, D_pore=1
    _, _, xd_m, _, _, _ = fit_effective_diffusivity(sol, prob, :mass)
    @test xd_m[1] ≈ 1.5 atol = 0.02  # one saveat slack
    @test xd_m[end] == sol.t[end]

    # Explicit t_fit always wins, even for :mass
    _, _, xd_m_exp, _, _, _ = fit_effective_diffusivity(sol, prob, :mass; t_fit=(0.5, 2.0))
    @test xd_m_exp[1] ≈ 0.5 atol = 0.02
    @test xd_m_exp[end] ≈ 2.0 atol = 0.02

    # Short simulation → :mass late cap should fall back to 0.5·(t_end-t_start)
    sol_short = solve(prob, ROCK4(); saveat=0.01, tspan=(0.0, 1.0),
                      reltol=1e-10, abstol=1e-12)
    _, _, xd_short, _, _, _ = fit_effective_diffusivity(sol_short, prob, :mass)
    @test xd_short[1] ≈ 0.5 atol = 0.02  # capped at half the trajectory
end

@testset "mass_uptake — c0_total reference behavior" begin
    # Direct test of the mass_uptake API: the prob-aware overload must
    # default to c0_total=0 (the true pre-clamp initial state for the
    # default solve path), not nansum(c_hist[1]) — which is the post-clamp
    # state carrying the Dirichlet face contribution.
    N = 33
    img = trues(1, 1, N)
    prob = TransientDiffusionProblem(img;
        axis=:z, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
    sol = solve(prob, ROCK4();
        saveat=0.01,
        callback=StopAtFluxBalance(prob; abstol=1e-8),
        tspan=(0.0, 20.0),
        reltol=1e-12, abstol=1e-14)

    # Prob overload default: reference is 0, so the asymptote is (c1+c2)/2.
    m_prob = Tortuosity.mass_uptake(sol.u, prob)
    @test m_prob[end] ≈ 0.5 atol = 1e-4

    # Explicit c0_total override still works.
    m_override = Tortuosity.mass_uptake(sol.u, prob; c0_total=0.0)
    @test m_override == m_prob

    # Primitive img overload preserves legacy behavior (subtract c_hist[1]).
    m_img = Tortuosity.mass_uptake(sol.u, prob.img)
    @test m_img[1] == 0.0  # legacy: first entry subtracts itself
    @test m_img[end] ≈ 0.5 - 1/N atol = 1e-4  # O(1/N) biased, as documented
end
