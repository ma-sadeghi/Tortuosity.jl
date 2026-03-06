# Tests for transient diffusion functionality via the TransientProblem public API
using Test
using SparseArrays
using LinearAlgebra: norm, tr
using Tortuosity
using Tortuosity:
    Imaginator,
    apply_boundaries!,
    find_boundary_nodes,
    analytic_conc,
    analytic_mass,
    analytic_flux,
    analytic_∑flux

# --- Analytical solutions (pure math, no ODE solve) ---

@testset "analytic_conc — boundary conditions" begin
    # C(x=0, t) = C1 = 1 and C(x=L, t) = C2 = 0 for any t
    for t in [0.01, 0.1, 1.0, 10.0]
        @test analytic_conc(1.0, 0.0, t) ≈ 1.0 atol = 1e-8
        @test analytic_conc(1.0, 1.0, t) ≈ 0.0 atol = 1e-8
    end
end

@testset "analytic_conc — steady state is linear" begin
    # At large t, C(x) → C1 + (C2-C1)*x/L = 1 - x
    for x in [0.2, 0.3, 0.5, 0.7, 0.9]
        @test analytic_conc(1.0, x, 10.0) ≈ (1.0 - x) atol = 1e-8
    end
end

@testset "analytic_conc — D scaling" begin
    # Higher D reaches steady state faster: |C - C_ss| should be smaller
    c_slow = analytic_conc(0.1, 0.5, 0.5)
    c_fast = analytic_conc(10.0, 0.5, 0.5)
    @test abs(c_fast - 0.5) < abs(c_slow - 0.5)
end

@testset "analytic_conc — vector of times" begin
    ts = [0.1, 0.5, 1.0, 5.0]
    result = analytic_conc(1.0, 0.5, ts)
    @test length(result) == 4
    # Last time point should be near steady state
    @test result[end] ≈ 0.5 atol = 1e-6
end

@testset "analytic_mass — limits" begin
    # Series converges slowly at t=0; with 100 terms the residual is ~0.002
    @test analytic_mass(1.0, 0.0) ≈ 0.0 atol = 0.01
    @test analytic_mass(1.0, 10.0) ≈ 1.0 atol = 1e-6
    @test analytic_mass(1.0, 100.0) ≈ 1.0 atol = 1e-8
end

@testset "analytic_mass — monotonically increasing" begin
    ts = collect(range(0.0, 5.0; length=50))
    ms = analytic_mass(1.0, ts)
    @test all(diff(ms) .>= 0)
end

@testset "analytic_flux — steady state" begin
    # At steady state, flux = D*(C1-C2)/L = 1.0 everywhere
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]
        @test analytic_flux(1.0, x, 10.0) ≈ 1.0 atol = 1e-6
    end
end

@testset "analytic_∑flux — cumulative" begin
    @test analytic_∑flux(1.0, 0.0) ≈ 0.0 atol = 1e-8
    # Monotonically increasing
    ts = collect(range(0.01, 5.0; length=50))
    Qs = analytic_∑flux(1.0, ts)
    @test all(diff(Qs) .>= 0)
    # At large t, slope approaches D*(C1-C2)/L = 1.0
    Q1 = analytic_∑flux(1.0, 100.0)
    Q2 = analytic_∑flux(1.0, 101.0)
    @test (Q2 - Q1) ≈ 1.0 atol = 1e-4
end

# --- Operator tests via TransientProblem ---

open_8 = ones(Bool, (8, 8, 8))
blob_8 = BitArray(Imaginator.blobs(; shape=(8, 8, 8), porosity=0.5, blobiness=1, seed=42))

# Helper: build reverse map from pore-vector index → (i,j,k) grid position
function vec_to_pos(grid_to_vec, img)
    v2p = Dict{Int,Tuple{Int,Int,Int}}()
    for ci in CartesianIndices(img)
        idx = grid_to_vec[ci]
        idx > 0 && (v2p[idx] = Tuple(ci))
    end
    return v2p
end

@testset "TransientProblem — open space, $(ax)-axis" for ax in (:x, :y, :z)
    prob = TransientProblem(
        open_8, 0.1; axis=ax, bound_mode=(1.0, 0.0), gpu=false, dtype=Float64
    )
    A = prob.A
    n = count(open_8)  # 512
    coeff = prob.D / prob.dx^2  # = 49.0

    @test size(A) == (n, n)

    # Boundary rows (Dirichlet) should be all zeros
    inlet_face, outlet_face = Dict(
        :x => (:left, :right), :y => (:front, :back), :z => (:bottom, :top)
    )[ax]
    inlet_nodes = find_boundary_nodes(prob.img, inlet_face)
    outlet_nodes = find_boundary_nodes(prob.img, outlet_face)
    bc_nodes = vcat(inlet_nodes, outlet_nodes)
    for node in bc_nodes
        @test nnz(A[node, :]) == 0
    end

    # Build reverse map from prob.grid_to_vec
    v2pos = vec_to_pos(prob.grid_to_vec, prob.img)

    # Check exact values for every interior node
    interior = setdiff(1:n, bc_nodes)
    for node in interior
        i, j, k = v2pos[node]
        pos = [i, j, k]
        # Count neighbors: each of ±1 in each dimension, if within [1,8]
        neighbors = 0
        for dim in 1:3, delta in (-1, 1)
            nb = copy(pos)
            nb[dim] += delta
            1 <= nb[dim] <= 8 && (neighbors += 1)
        end
        # Exact diagonal value: -neighbors * D/dx²
        @test A[node, node] ≈ -neighbors * coeff atol = 1e-10
        # Every off-diagonal nonzero should be exactly D/dx²
        row = A[node, :]
        off_diag = [row[j] for j in findnz(row)[1] if j != node]
        @test length(off_diag) == neighbors
        @test all(v -> v ≈ coeff, off_diag)
        # Row sum = 0 (mass conservation)
        @test abs(sum(row)) < 1e-10
    end
end

@testset "TransientProblem — insulated boundary (1, NaN)" begin
    prob = TransientProblem(
        open_8, 0.1; axis=:z, bound_mode=(1.0, NaN), gpu=false, dtype=Float64
    )
    A = prob.A
    coeff = prob.D / prob.dx^2

    # Only inlet rows should be zeroed; outlet rows should be intact
    inlet_nodes = find_boundary_nodes(prob.img, :bottom)
    outlet_nodes = find_boundary_nodes(prob.img, :top)
    for node in inlet_nodes
        @test nnz(A[node, :]) == 0
    end
    for node in outlet_nodes
        @test nnz(A[node, :]) > 0
    end

    # Outlet nodes: exact values
    for node in outlet_nodes
        row = A[node, :]
        off_diag = [row[j] for j in findnz(row)[1] if j != node]
        @test all(v -> v ≈ coeff, off_diag)
        @test A[node, node] ≈ -length(off_diag) * coeff atol = 1e-10
        @test abs(sum(row)) < 1e-10
    end
end

@testset "TransientProblem — seeded blob" begin
    prob = TransientProblem(
        blob_8, 0.1; axis=:z, bound_mode=(1.0, 0.0), gpu=false, dtype=Float64
    )
    A = prob.A
    n = size(A, 1)
    coeff = prob.D / prob.dx^2

    @test size(A) == (n, n)

    # Boundary rows should be zeroed
    inlet_nodes = find_boundary_nodes(prob.img, :bottom)
    outlet_nodes = find_boundary_nodes(prob.img, :top)
    bc_nodes = vcat(inlet_nodes, outlet_nodes)
    for node in bc_nodes
        @test nnz(A[node, :]) == 0
    end

    # Separate interior into connected (has neighbors) and isolated (no pore neighbors)
    interior = setdiff(1:n, bc_nodes)
    connected = [node for node in interior if nnz(A[node, :]) > 0]
    isolated = [node for node in interior if nnz(A[node, :]) == 0]

    # Isolated pore voxels (surrounded by solid) should have all-zero rows
    for node in isolated
        @test A[node, node] == 0.0
    end

    # Connected interior: exact values — off-diag = coeff, diag = -k*coeff
    for node in connected
        row = A[node, :]
        off_diag = [row[j] for j in findnz(row)[1] if j != node]
        k = length(off_diag)
        @test 1 <= k <= 6
        @test all(v -> v ≈ coeff, off_diag)
        @test A[node, node] ≈ -k * coeff atol = 1e-10
        @test abs(sum(row)) < 1e-10
    end
end

# --- apply_boundaries! (user-facing, called before init_state) ---

@testset "apply_boundaries! — Dirichlet (1, 0)" begin
    prob = TransientProblem(
        open_8, 0.1; axis=:z, bound_mode=(1, 0), gpu=false, dtype=Float64
    )
    C0 = zeros(Float64, 8, 8, 8)
    apply_boundaries!(C0, prob)
    @test all(C0[:, :, 1] .== 1.0)
    @test all(C0[:, :, 8] .== 0.0)
    @test all(C0[:, :, 4] .== 0.0)  # interior untouched
end

@testset "apply_boundaries! — insulated (1, NaN)" begin
    prob = TransientProblem(
        open_8, 0.1; axis=:z, bound_mode=(1.0, NaN), gpu=false, dtype=Float64
    )
    C0 = fill(0.5, 8, 8, 8)
    apply_boundaries!(C0, prob)
    @test all(C0[:, :, 1] .== 1.0)       # inlet set to C1
    @test all(C0[:, :, 8] .== 0.5)       # outlet untouched (insulated)
    @test all(C0[:, :, 4] .== 0.5)       # interior untouched
end

# --- Integration tests: matrix-level properties via TransientProblem ---

@testset "TransientProblem — integration properties (open 8³)" begin
    # Hardcoded values verified analytically: for 8³ open cube, D=1, dx=1/7,
    # 384 interior nodes with 3-6 neighbors each, coeff = D/dx² = 49.
    # trace = Σ(-k_i * 49) over interior, norm = sqrt(Σ(k_i² + k_i) * 49²)
    prob = TransientProblem(
        open_8, 0.1; axis=:z, bound_mode=(1.0, 0.0), gpu=false, dtype=Float64
    )
    A = prob.A

    @test size(A, 1) == 512
    @test nnz(A) == 2496
    @test norm(A) ≈ 5771.193290819499 atol = 1e-8
    @test tr(Matrix(A)) ≈ -103488.0 atol = 1e-8

    # D scaling: doubling D doubles the norm
    prob_2D = TransientProblem(
        open_8, 0.1; axis=:z, bound_mode=(1.0, 0.0), D=2.0, gpu=false, dtype=Float64
    )
    @test norm(prob_2D.A) ≈ 2.0 * 5771.193290819499 atol = 1e-8

    # dx scaling: halving dx quadruples D/dx²
    dx_half = prob.dx / 2
    prob_fine = TransientProblem(
        open_8, 0.1; axis=:z, bound_mode=(1.0, 0.0), dx=dx_half, gpu=false, dtype=Float64
    )
    @test norm(prob_fine.A) ≈ 4.0 * 5771.193290819499 atol = 1e-8

    # Axis symmetry: open cube gives identical norm and nnz regardless of axis
    prob_x = TransientProblem(
        open_8, 0.1; axis=:x, bound_mode=(1.0, 0.0), gpu=false, dtype=Float64
    )
    prob_y = TransientProblem(
        open_8, 0.1; axis=:y, bound_mode=(1.0, 0.0), gpu=false, dtype=Float64
    )
    @test norm(prob_x.A) ≈ 5771.193290819499 atol = 1e-8
    @test norm(prob_y.A) ≈ 5771.193290819499 atol = 1e-8
    @test nnz(prob_x.A) == 2496
    @test nnz(prob_y.A) == 2496
end

@testset "TransientProblem — integration properties (insulated 8³)" begin
    # Insulated outlet adds 64 active rows vs Dirichlet
    prob = TransientProblem(
        open_8, 0.1; axis=:z, bound_mode=(1.0, NaN), gpu=false, dtype=Float64
    )
    A = prob.A
    @test nnz(A) == 2848
    @test norm(A) ≈ 6096.513757878351 atol = 1e-8
    @test tr(Matrix(A)) ≈ -117600.0 atol = 1e-8
end

@testset "TransientProblem — integration properties (blob 8³)" begin
    prob = TransientProblem(
        blob_8, 0.1; axis=:z, bound_mode=(1.0, 0.0), gpu=false, dtype=Float64
    )
    A = prob.A

    @test size(A, 1) == 251
    @test nnz(A) == 639
    @test norm(A) ≈ 2101.8658377736674 atol = 1e-8
    @test tr(Matrix(A)) ≈ -22491.0 atol = 1e-8

    # D scaling
    prob_2D = TransientProblem(
        blob_8, 0.1; axis=:z, bound_mode=(1.0, 0.0), D=2.0, gpu=false, dtype=Float64
    )
    @test norm(prob_2D.A) ≈ 2.0 * 2101.8658377736674 atol = 1e-8
end

@testset "TransientProblem — A * C0 initial RHS" begin
    # For open 8³ with C0=1 on z=1 face and 0 elsewhere:
    # only the 64 nodes at k=2 see a nonzero neighbor (the inlet at C=1)
    # each gets rhs = coeff * 1 = 49, all others get 0
    prob = TransientProblem(
        open_8, 0.1; axis=:z, bound_mode=(1, 0), gpu=false, dtype=Float64
    )
    C0 = zeros(Float64, 8, 8, 8)
    apply_boundaries!(C0, prob)
    c0_vec = C0[prob.img]
    rhs = prob.A * c0_vec

    @test norm(rhs) ≈ 392.0 atol = 1e-8
    @test sum(rhs) ≈ 3136.0 atol = 1e-8
    @test maximum(rhs) ≈ 49.0 atol = 1e-8
    @test minimum(rhs) ≈ 0.0 atol = 1e-10

    # BC nodes have exactly zero RHS
    bc_nodes = vcat(
        find_boundary_nodes(prob.img, :bottom), find_boundary_nodes(prob.img, :top)
    )
    @test all(rhs[bc_nodes] .== 0.0)

    # k=2 layer nodes all have rhs = 49.0 (one inlet neighbor at C=1)
    g2v = prob.grid_to_vec
    k2_nodes = [g2v[i, j, 2] for i in 1:8, j in 1:8 if g2v[i, j, 2] > 0]
    @test all(rhs[k2_nodes] .≈ 49.0)
end

# --- grid_to_vec via TransientProblem ---

@testset "TransientProblem — grid_to_vec consistency" begin
    prob = TransientProblem(
        blob_8, 0.1; axis=:z, bound_mode=(1.0, 0.0), gpu=false, dtype=Float64
    )
    g = prob.grid_to_vec
    @test size(g) == size(prob.img)
    @test all(g[.!prob.img] .== 0)                       # solid → 0
    @test minimum(g[prob.img]) == 1                        # pore indices start at 1
    @test maximum(g[prob.img]) == count(prob.img)          # max = number of pores
    @test length(unique(g[prob.img])) == count(prob.img)   # all unique
end

# --- Solver and fitting tests (commented out pending refactor) ---
#=
open_16 = ones(Bool, (16, 16, 16))

@testset "Open space transient — $(ax)-axis" for ax in (:x, :y)
    prob = TransientProblem(open_16, 0.05; axis=ax, bound_mode=(1, 0), gpu=false, dtype=Float64)
    state = init_state(prob)
    solve!(state, prob, stop_at_delta_flux(0.01, prob); max_iter=500)
    C_final = state.C[end]
    mid = get_slice_conc(C_final, prob, size(open_16, 1) ÷ 2)
    @test mid ≈ 0.5 atol = 0.05
    J_in = get_flux(C_final, prob; ind=1)
    J_out = get_flux(C_final, prob; ind=:end)
    @test J_in ≈ J_out atol = 0.02
end

prob_z = TransientProblem(open_16, 0.05; axis=:z, bound_mode=(1, 0), gpu=false, dtype=Float64)

@testset "Open space transient — z-axis" begin
    state = init_state(prob_z)
    solve!(state, prob_z, stop_at_delta_flux(0.01, prob_z); max_iter=500)
    C_final = state.C[end]
    mid = get_slice_conc(C_final, prob_z, size(open_16, 3) ÷ 2)
    @test mid ≈ 0.5 atol = 0.05
    J_in = get_flux(C_final, prob_z; ind=1)
    J_out = get_flux(C_final, prob_z; ind=:end)
    @test J_in ≈ J_out atol = 0.02
end

@testset "Stop conditions" begin
    state_t = init_state(prob_z)
    solve!(state_t, prob_z, stop_at_time(0.2); max_iter=100)
    @test state_t.t[end] >= 0.2
    @test length(state_t.C) == length(state_t.t)
    avg = sum(state_t.C[end]) / length(state_t.C[end])
    @test avg > 0
end

@testset "Insulated boundary (1, NaN)" begin
    prob = TransientProblem(open_16, 0.1; axis=:z, bound_mode=(1.0, NaN), gpu=false, dtype=Float64)
    state = init_state(prob)
    solve!(state, prob, stop_at_time(3.0); max_iter=100)
    avg = sum(state.C[end]) / length(state.C[end])
    @test avg ≈ 1.0 atol = 0.1
end

@testset verbose = true "effective_diffusivity — open space" begin
    state_fit = init_state(prob_z)
    solve!(state_fit, prob_z, stop_at_delta_flux(0.01, prob_z); max_iter=500)

    @testset "method = :mass" begin
        D_eff, φ, _, _, _, _ = effective_diffusivity(state_fit, prob_z, :mass)
        @test D_eff ≈ 1.0 atol = 0.1
        @test φ ≈ 1.0 atol = 0.15
    end

    @testset "method = :conc" begin
        D_eff, _, _, _, _, _ = effective_diffusivity(state_fit, prob_z, :conc; depth=0.5)
        @test D_eff ≈ 1.0 atol = 0.1
    end

    @testset "method = :flux" begin
        D_eff, _, _, _, _, _ = effective_diffusivity(state_fit, prob_z, :flux; depth=0.5)
        @test D_eff ≈ 1.0 atol = 0.25
    end
end
=#
