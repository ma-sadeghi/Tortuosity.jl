# Physics invariants of the transient solver, and consistency of the observables.
#
# Two ideas drive this file:
#
#  1. The transient ODE path and the steady linear-solve path share almost no
#     code (ROCK4 + a scaled Laplacian vs. LinearSolve/CG on a Dirichlet-
#     eliminated system), yet they must agree in the t → ∞ limit. Cross-checking
#     them is the cheapest way to catch a regression that a single-path test
#     would rationalise as "close enough".
#
#  2. Every observable has a pore-vector overload and a full-grid overload,
#     plus vector-of-snapshots and problem-aware convenience wrappers. Those
#     exist purely to avoid materialising the 3D field, so they are prime
#     targets for optimisation — and must stay numerically identical.

using Test
using LinearAlgebra
using Tortuosity
using Tortuosity:
    Imaginator,
    axis_dim,
    effective_diffusivity,
    find_boundary_nodes,
    fit_voxel_diffusivity,
    flux,
    mass_uptake,
    reconstruct_field,
    reconstruct_slice,
    slice_concentration,
    slice_indices,
    tortuosity

steady_field(img; axis, D=nothing) = begin
    sim = SteadyDiffusionProblem(img; axis=axis, gpu=false, D=D)
    reconstruct_field(solve(sim.prob, KrylovJL_CG(); reltol=1e-12).u, img)
end

function transient_fixtures()
    imgs = Tuple{String,Array{Bool,3}}[]
    push!(imgs, ("open 10x6x6", ones(Bool, 10, 6, 6)))
    let img = ones(Bool, 12, 8, 8)
        img[4:8, 3:6, 3:6] .= false
        push!(imgs, ("obstructed 12x8x8", img))
    end
    let img = Array{Bool}(Imaginator.blobs(; shape=(12, 12, 12), porosity=0.65, blobiness=1, seed=5))
        img = Array{Bool}(Imaginator.trim_nonpercolating_paths(img; axis=:x))
        count(img) >= 50 && push!(imgs, ("trimmed blob 12^3", img))
    end
    return imgs
end

const TRANSIENT_IMAGES = transient_fixtures()

# --- Transient ↔ steady agreement ---

@testset "transient relaxes to the steady solution — $(label)" for (label, img) in TRANSIENT_IMAGES
    prob = TransientDiffusionProblem(img; axis=:x, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
    # The default voxel_size normalises the axis to [0, 1], so the slowest mode
    # decays like exp(-π²·t/τ). Integrating to t = 20 leaves nothing but the
    # integrator's own error — no stop-condition tolerance to confound the
    # comparison. (`StopAtSteadyState` itself is covered in test_transient.jl.)
    sol = solve(prob, ROCK4(); saveat=5.0, tspan=(0.0, 20.0), reltol=1e-10, abstol=1e-12)
    @test sol.retcode == :Success

    c_transient = reconstruct_field(sol.u[end], prob.img)
    c_steady = steady_field(img; axis=:x)
    @test maximum(abs, (c_transient .- c_steady)[prob.img]) < 1e-7

    # The derived transport properties must agree too, not just the raw field.
    @test tortuosity(c_transient, prob.img; axis=:x) ≈
          tortuosity(c_steady, img; axis=:x) rtol = 1e-7
end

@testset "transient with variable D relaxes to the variable-D steady solution" begin
    img = ones(Bool, 10, 6, 6)
    D = fill(1.0, size(img))
    D[4:7, :, :] .= 0.15                    # a low-diffusivity band across the domain
    prob = TransientDiffusionProblem(img; axis=:x, bc_inlet=1.0, bc_outlet=0.0, D=D, gpu=false)
    # The low-D band slows the slowest mode by roughly D_eff, so integrate longer.
    sol = solve(prob, ROCK4(); saveat=10.0, tspan=(0.0, 60.0), reltol=1e-10, abstol=1e-12)
    @test sol.retcode == :Success

    c_transient = reconstruct_field(sol.u[end], prob.img)
    c_steady = steady_field(img; axis=:x, D=D)
    @test maximum(abs, (c_transient .- c_steady)[prob.img]) < 1e-7
end

# --- Conservation in a closed system ---

@testset "an insulated domain conserves total mass and homogenises" begin
    # With both faces insulated no rows are zeroed, so A is the plain scaled
    # Laplacian: its columns sum to zero and every explicit RK stage therefore
    # leaves Σu untouched. The steady limit is the initial mean.
    img = ones(Bool, 12, 6, 6)
    img[5:8, 2:4, 2:5] .= false
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=nothing, bc_outlet=nothing, gpu=false)

    u0 = zeros(Float64, size(img))
    u0[1:4, :, :] .= 1.0
    u0[9:12, :, :] .= 0.25

    sol = solve(prob, ROCK4(); saveat=1.0, tspan=(0.0, 20.0), u0=u0,
                reltol=1e-10, abstol=1e-12)
    @test sol.retcode == :Success

    totals = [sum(u) for u in sol.u]
    expected_total = sum(u0[prob.img])
    @test maximum(abs, totals .- expected_total) < 1e-8 * max(1.0, expected_total)

    # Diffusion in a sealed box drives everything to the volume-average.
    c_mean = expected_total / count(prob.img)
    @test maximum(abs, sol.u[end] .- c_mean) < 1e-6
end

@testset "an insulated domain leaves a uniform field untouched" begin
    img = ones(Bool, 8, 5, 5)
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=nothing, bc_outlet=nothing, gpu=false)
    sol = solve(prob, ROCK4(); saveat=0.5, tspan=(0.0, 2.0), u0=fill(0.37, size(img)),
                reltol=1e-10, abstol=1e-12)
    # A constant field is in the kernel of the operator, so only roundoff in the
    # row sums may move it.
    for u in sol.u
        @test maximum(abs, u .- 0.37) < 1e-6
    end
end

# --- Operator structure ---

@testset "transient operator structure" begin
    img = ones(Bool, 6, 5, 5)
    img[3, 2:4, 2:4] .= false

    @testset "insulated operator is symmetric with zero row sums" begin
        prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=nothing, bc_outlet=nothing, gpu=false)
        A = Array(prob.A)
        @test issymmetric(A)
        @test maximum(abs, vec(sum(A; dims=2))) < 1e-8
        # dc/dt = A·c must be a *diffusion* operator: negative diagonal, positive
        # coupling. A sign flip here would run the equation backwards in time.
        @test all(diag(A) .<= 0)
        @test all((A - Diagonal(diag(A))) .>= 0)
    end

    @testset "Dirichlet zeroes only the boundary rows" begin
        insulated = TransientDiffusionProblem(img; axis=:z, bc_inlet=nothing, bc_outlet=nothing, gpu=false)
        clamped = TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=0, gpu=false)
        bc = union(find_boundary_nodes(clamped.img, :bottom), find_boundary_nodes(clamped.img, :top))
        free = setdiff(1:count(clamped.img), bc)
        A_ins = Array(insulated.A)
        A_bc = Array(clamped.A)
        @test all(iszero, A_bc[bc, :])
        @test A_bc[free, :] ≈ A_ins[free, :]
    end

    @testset "operator entries scale as 1/voxel_size²" begin
        a = TransientDiffusionProblem(img; axis=:z, bc_inlet=nothing, bc_outlet=nothing,
                                      voxel_size=1.0, gpu=false)
        b = TransientDiffusionProblem(img; axis=:z, bc_inlet=nothing, bc_outlet=nothing,
                                      voxel_size=0.5, gpu=false)
        @test Array(b.A) ≈ 4 .* Array(a.A)
    end

    @testset "default voxel_size spans the unit interval along the axis" begin
        for ax in (:x, :y, :z)
            prob = TransientDiffusionProblem(img; axis=ax, gpu=false)
            @test prob.voxel_size ≈ 1 / (size(img, axis_dim(ax)) - 1)
        end
    end

    @testset "CPU problems are Float64 end to end" begin
        prob = TransientDiffusionProblem(img; axis=:z, gpu=false)
        @test eltype(prob.A) === Float64
        @test prob.D isa Float64
        sol = solve(prob, ROCK4(); saveat=0.5, tspan=(0.0, 1.0))
        @test all(u isa Vector{Float64} for u in sol.u)
    end
end

# --- Observable overload consistency ---

@testset "observables agree between pore-vector and full-grid forms" begin
    img = TRANSIENT_IMAGES[2][2]
    prob = TransientDiffusionProblem(img; axis=:x, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
    sol = solve(prob, ROCK4(); saveat=0.05, tspan=(0.0, 0.4), reltol=1e-10, abstol=1e-12)
    grids = [reconstruct_field(u, prob.img) for u in sol.u]
    N = size(img, 1)

    @testset "slice_concentration" begin
        for ind in (1, 2, N ÷ 2, N), pore_only in (false, true)
            from_vec = slice_concentration(sol.u[end], prob.img, :x, ind;
                                           pore_index=prob.pore_index, pore_only=pore_only)
            from_grid = slice_concentration(grids[end], prob.img, :x, ind; pore_only=pore_only)
            @test from_vec ≈ from_grid rtol = 1e-12
        end
    end

    @testset "flux" begin
        for ind in (1, 2, N ÷ 2, N - 1)
            from_vec = flux(sol.u[end], prob.D, prob.voxel_size, prob.img, :x;
                            ind=ind, pore_index=prob.pore_index)
            from_grid = flux(grids[end], prob.D, prob.voxel_size, prob.img, :x; ind=ind)
            @test from_vec ≈ from_grid rtol = 1e-12
        end
        # `:end` resolves to size(img, axis) - 1
        @test flux(sol.u[end], prob.D, prob.voxel_size, prob.img, :x;
                   ind=:end, pore_index=prob.pore_index) ≈
              flux(sol.u[end], prob.D, prob.voxel_size, prob.img, :x;
                   ind=N - 1, pore_index=prob.pore_index)
    end

    @testset "snapshot-vector overloads map the scalar ones" begin
        @test slice_concentration(sol.u, prob.img, :x, 3; pore_index=prob.pore_index) ≈
              [slice_concentration(u, prob.img, :x, 3; pore_index=prob.pore_index) for u in sol.u]
        @test flux(sol.u, prob.D, prob.voxel_size, prob.img, :x; ind=3, pore_index=prob.pore_index) ≈
              [flux(u, prob.D, prob.voxel_size, prob.img, :x; ind=3) for u in grids]
    end

    @testset "problem-aware wrappers unpack the same arguments" begin
        @test slice_concentration(sol.u[end], prob, 4) ≈
              slice_concentration(sol.u[end], prob.img, :x, 4; pore_index=prob.pore_index)
        @test slice_concentration(sol.u[end], prob, 4; pore_only=true) ≈
              slice_concentration(sol.u[end], prob.img, :x, 4;
                                  pore_index=prob.pore_index, pore_only=true)
        @test flux(sol.u[end], prob; ind=3) ≈
              flux(sol.u[end], prob.D, prob.voxel_size, prob.img, :x; ind=3,
                   pore_index=prob.pore_index)
        @test mass_uptake(sol.u, prob) ≈ mass_uptake(sol.u, prob.img; c0_total=0)
        @test isequal(reconstruct_slice(sol.u[end], prob, 4),
                      reconstruct_slice(sol.u[end], prob.pore_index, :x, 4))
        @test slice_indices(prob, 4) == slice_indices(prob.pore_index, :x, 4)
    end

    @testset "pore_only rescales by the pore fraction of the slice" begin
        ind = N ÷ 2
        n_pore = count(selectdim(prob.img, 1, ind))
        n_total = length(selectdim(prob.img, 1, ind))
        full = slice_concentration(sol.u[end], prob, ind; pore_only=false)
        pores = slice_concentration(sol.u[end], prob, ind; pore_only=true)
        @test full * n_total ≈ pores * n_pore rtol = 1e-12
    end
end

# --- Exact observable values on a uniform grid ---

@testset "flux on the steady open box equals D·Δc/L exactly" begin
    # voxel_size defaults to 1/(N-1) so the domain spans [0, 1]; with a unit
    # concentration drop and D = 1 the flux is exactly 1 at every face. This
    # pins the voxel_size bookkeeping inside `flux`.
    N = 9
    img = ones(Bool, N, 5, 5)
    prob = TransientDiffusionProblem(img; axis=:x, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
    c = steady_field(img; axis=:x)
    for ind in 1:(N - 1)
        @test flux(c, prob.D, prob.voxel_size, prob.img, :x; ind=ind) ≈ 1.0 atol = 1e-9
    end
    @test effective_diffusivity(c, img; axis=:x, voxel_size=prob.voxel_size) ≈ 1.0 atol = 1e-9
end

@testset "flux normalises by the full slice area, not the pore count" begin
    # A prismatic duct of open fraction φ carries flux exactly φ: the per-voxel
    # drop is 1/(N-1) across φ·M pore voxels, cancelled by dividing by
    # voxel_size, then divided by the *full* slice area M.
    #
    # This is the only absolute flux value in this file measured on a geometry
    # where the pore count and the slice area differ. Every other flux test here
    # compares one overload against another, so a change of normalisation would
    # move both sides together and go unnoticed.
    N = 9
    img = falses(N, 8, 8)
    img[:, 3:6, 3:6] .= true
    φ = 16 / 64
    @test φ != 1                       # otherwise the distinction is vacuous
    prob = TransientDiffusionProblem(img; axis=:x, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
    c = steady_field(img; axis=:x)
    u = c[prob.img]
    for ind in 1:(N - 1)
        @test flux(c, prob.D, prob.voxel_size, prob.img, :x; ind=ind) ≈ φ atol = 1e-9
        @test flux(u, prob.D, prob.voxel_size, prob.img, :x;
                   ind=ind, pore_index=prob.pore_index) ≈ φ atol = 1e-9
    end
end

@testset "mass_uptake rises monotonically to φ·(c1+c2)/2" begin
    img = ones(Bool, 12, 6, 6)
    img[5:8, 2:4, 2:4] .= false
    φ = count(img) / length(img)
    # The asymptote is the plain porosity-weighted mean only because the
    # obstruction is mirror-symmetric about the transport axis; without that the
    # steady field is still linear in resistance but not in position, and the
    # volume average drifts off (c1+c2)/2. Assert the precondition rather than
    # relying on the fixture happening to have it.
    @test img == reverse(img; dims=1)
    for (c1, c2) in ((1.0, 0.0), (2.0, 1.0))
        prob = TransientDiffusionProblem(img; axis=:x, bc_inlet=c1, bc_outlet=c2, gpu=false)
        sol = solve(prob, ROCK4();
            saveat=0.25,
            callback=StopAtSteadyState(abstol=1e-6, reltol=1e-6),
            tspan=(0.0, 100.0), reltol=1e-10, abstol=1e-12)
        @test sol.retcode == :Terminated
        m = mass_uptake(sol.u, prob)
        @test all(diff(m) .>= -1e-9)
        @test m[end] ≈ φ * (c1 + c2) / 2 atol = 1e-4
    end
end

# --- Boundary-condition closure branches ---

@testset "constant f(t) boundaries reproduce the numeric-boundary solution" begin
    # `build_rhs` has four closures (neither / inlet / outlet / both are
    # functions). Driving all four with constant functions must land on the
    # same trajectory as plain numbers — a cheap way to keep every branch live.
    img = ones(Bool, 10, 5, 5)
    reference = nothing
    cases = (
        ("numbers", 1.0, 0.0),
        ("inlet f(t)", t -> 1.0, 0.0),
        ("outlet f(t)", 1.0, t -> 0.0),
        ("both f(t)", t -> 1.0, t -> 0.0),
    )
    for (label, bc_in, bc_out) in cases
        prob = TransientDiffusionProblem(img; axis=:x, bc_inlet=bc_in, bc_outlet=bc_out, gpu=false)
        sol = solve(prob, ROCK4(); saveat=0.1, tspan=(0.0, 0.6), reltol=1e-10, abstol=1e-12)
        if reference === nothing
            reference = sol.u
        else
            # `zip` truncates silently, so pin the snapshot count first —
            # otherwise a branch that produced fewer snapshots would compare a
            # shorter prefix and still pass.
            @test length(sol.u) == length(reference)
            @test maximum(maximum(abs, a .- b) for (a, b) in zip(sol.u, reference)) < 1e-6
        end
    end
end

@testset "an oscillating inlet drives an oscillating interior response" begin
    freq = 1.0
    img = trues(1, 1, 24)
    prob = TransientDiffusionProblem(img;
        axis=:z, bc_inlet=t -> (sin(2π * freq * t) + 1) / 2, bc_outlet=nothing, gpu=false)
    sol = solve(prob, ROCK4(); saveat=0.02, tspan=(0.0, 6.0),
                u0=fill(0.5, 1, 1, 24), reltol=1e-8, abstol=1e-10)
    mid = [slice_concentration(u, prob, 12) for u in sol.u]
    tail = mid[(length(mid) ÷ 2):end]
    amplitude = maximum(tail) - minimum(tail)
    # Drive amplitude is 0.5; at this depth and frequency the diffusive skin
    # attenuates it to ~0.42. A threshold of 1e-3 would still pass with the
    # response 400× too small, which is indistinguishable from a broken
    # time-dependent boundary that only leaks numerical noise inward.
    @test 0.2 < amplitude < 0.5
    @test all(-0.05 .<= tail .<= 1.05)
    # …and it must oscillate at the driving frequency, not merely wander:
    # over the 3 s tail there should be ~3 crossings of the mid-level in each
    # direction, so count sign changes about the tail mean.
    level = (maximum(tail) + minimum(tail)) / 2
    crossings = count(i -> (tail[i] - level) * (tail[i + 1] - level) < 0, 1:(length(tail) - 1))
    @test crossings ≈ 2 * freq * (sol.t[end] - sol.t[length(mid) ÷ 2]) atol = 2
end

# --- Solve plumbing ---

@testset "solve honours u0, saveat and tspan" begin
    img = ones(Bool, 8, 4, 4)
    prob = TransientDiffusionProblem(img; axis=:x, bc_inlet=1.0, bc_outlet=0.0, gpu=false)

    @testset "snapshots are ordered and inside tspan" begin
        sol = solve(prob, ROCK4(); saveat=0.05, tspan=(0.0, 0.3))
        @test issorted(sol.t)
        @test all(0.0 .<= sol.t .<= 0.3 + 1e-12)
        @test length(sol.u) == length(sol.t)
        @test maximum(abs, diff(sol.t) .- 0.05) < 1e-9
    end

    @testset "u0 is applied, then overwritten on the Dirichlet faces" begin
        u0 = fill(0.42, size(img))
        sol = solve(prob, ROCK4(); saveat=0.5, tspan=(0.0, 1.0), u0=u0)
        first_snapshot = reconstruct_field(sol.u[1], prob.img)
        @test all(first_snapshot[1, :, :] .≈ 1.0)     # inlet clamped
        @test all(first_snapshot[end, :, :] .≈ 0.0)   # outlet clamped
        @test first_snapshot[4, 2, 2] ≈ 0.42 atol = 1e-6
    end

    @testset "the raw ODESolution is exposed for diagnostics" begin
        sol = solve(prob, ROCK4(); saveat=0.5, tspan=(0.0, 1.0))
        @test sol.prob === prob
        @test sol.ode_sol !== nothing
        @test Symbol(sol.ode_sol.retcode) == sol.retcode
    end
end

# --- Voxel-wise fitting ---

@testset "fit_voxel_diffusivity recovers τ = 1 in an open slab" begin
    # Every voxel on the sampled slice sees the same 1D response, so each
    # independent fit must return the pore diffusivity.
    N = 17
    img = ones(Bool, 4, 4, N)
    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1.0, bc_outlet=0.0, gpu=false)
    sol = solve(prob, ROCK4(); saveat=0.01, tspan=(0.0, 2.0), reltol=1e-10, abstol=1e-12)

    n_samples = 8
    taus, SE_taus, voxels = fit_voxel_diffusivity(sol, prob; depth=0.5, n_samples=n_samples)
    @test length(taus) == n_samples
    @test length(SE_taus) == n_samples
    @test length(voxels) == n_samples
    @test all(isapprox.(taus, 1.0; atol=1e-3))
    @test all(SE_taus .< 1e-2)

    # `voxels` is built by indexing into the slice, so a subset check cannot
    # fail. Pin the actual sampling rule instead: `n_samples` distinct voxels
    # spread evenly across the slice, spanning it end to end.
    depth_idx = round(Int, 1 + 0.5 * (N - 1))
    slice = slice_indices(prob, depth_idx)
    @test voxels == slice[round.(Int, LinRange(1, length(slice), n_samples))]
    @test length(unique(voxels)) == n_samples
    @test first(voxels) == first(slice)
    @test last(voxels) == last(slice)

    @testset "fit_depth=true also recovers the sampling depth" begin
        taus2, xs, SE_taus2, SE_xs, voxels2 = fit_voxel_diffusivity(
            sol, prob; depth=0.5, n_samples=4, fit_depth=true,
        )
        @test length(xs) == 4
        expected_depth = (depth_idx - 1) * prob.voxel_size
        @test all(isapprox.(xs, expected_depth; atol=1e-3))
        @test all(isapprox.(taus2, 1.0; atol=1e-3))
        @test voxels2 ⊆ slice_indices(prob, depth_idx)
        @test length(SE_taus2) == length(SE_xs) == 4
    end
end
