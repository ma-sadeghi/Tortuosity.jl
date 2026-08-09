# The two-level preconditioner is the one piece of this solver that can be
# wrong without failing: an inaccurate coarse solve makes CG stall and return a
# plausible tortuosity rather than an error. These tests pin the two properties
# that rule that out — the operator really is `W'AW` for the `W` the aggregates
# describe, and applying it is symmetric positive definite, which is what makes
# preconditioned CG converge to the same solution as unpreconditioned CG.

using Test
using LinearAlgebra
using SparseArrays
using Tortuosity
using Tortuosity:
    Imaginator,
    TwoLevelPreconditioner,
    two_level_preconditioner,
    DEFAULT_COARSE_SHIFT,
    _choose_block,
    reconstruct_field,
    tortuosity

# The prolongation matrix the preconditioner's aggregates describe, built here
# independently of anything in src/.
function prolongation(P, n)
    agg = Array(P.agg)
    rows = [i for i in 1:n if agg[i] > 0]
    return sparse(rows, [Int(agg[i]) for i in rows], ones(length(rows)), n, P.nc)
end

precond_fixtures() = [
    ("open 24^3", ones(Bool, 24, 24, 24)),
    ("untrimmed blob 32^3 seed=42",
     Array{Bool}(Imaginator.blobs(; shape=(32, 32, 32), porosity=0.5, blobiness=1, seed=42))),
    ("untrimmed blob 32^3 seed=7",
     Array{Bool}(Imaginator.blobs(; shape=(32, 32, 32), porosity=0.45, blobiness=2, seed=7))),
    ("slab plus detached cube", let img = falses(24, 24, 24)
        img[:, 1:8, 1:8] .= true            # spans the x axis
        img[8:14, 14:20, 14:20] .= true     # reaches neither face
        img
    end),
    ("hollow shell 12^3", let img = ones(Bool, 12, 12, 12)
        img[2:11, 2:11, 2:11] .= false
        img
    end),
    # Thinner than one block along y, so the block grid is one deep there and
    # the offset between a block and its z-neighbour collides with the offset
    # to its y-neighbour. Couplings still have to land on the right pair.
    ("thin slab 24x3x24", ones(Bool, 24, 3, 24)),
    ("thin slab 3x24x24", ones(Bool, 3, 24, 24)),
]

const PRECOND_IMAGES = precond_fixtures()

@testset "block size honours the coarse-size ceiling" begin
    @test _choose_block(64, 64, 64, 32_000) == 8       # 8^3 = 512 blocks, well under
    @test _choose_block(800, 800, 800, 32_000) == 26   # 31^3 = 29791
    for max_coarse in (500, 5_000, 32_000)
        bs = _choose_block(200, 200, 200, max_coarse)
        @test cld(200, bs)^3 <= max_coarse
        @test bs == 8 || cld(200, bs - 1)^3 > max_coarse   # smallest that fits
    end
end

@testset "coarse operator is W'AW — $(label)" for (label, img) in PRECOND_IMAGES
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    A = sim.prob.A
    n = size(A, 1)
    P = two_level_preconditioner(sim; block=4)
    @test P isa TwoLevelPreconditioner
    @test length(P.agg) == n
    @test all(0 .<= Array(P.agg) .<= P.nc)

    W = prolongation(P, n)
    Ac = Matrix(W' * A * W)
    # A block is kept exactly when its coarse diagonal is positive, so no kept
    # block may have a zero one.
    @test all(diag(Ac) .> 0)
    Ac_shifted = Ac + DEFAULT_COARSE_SHIFT * Diagonal(diag(Ac))

    # `P.fact` is the factorisation of that same matrix: applying it and then
    # multiplying back has to return what went in.
    v = collect(range(-1.0, 1.0; length=P.nc))
    @test Ac_shifted * (P.fact \ v) ≈ v rtol = 1e-8

    # And a voxel whose block was dropped is one whose whole coarse basis
    # function lies in the null space of A.
    agg = Array(P.agg)
    dropped = findall(iszero, agg)
    if !isempty(dropped)
        e = zeros(n)
        e[dropped] .= 1.0
        @test norm(A * e) <= 1e-12 * max(norm(e), 1)
    end
end

@testset "application is symmetric positive definite — $(label)" for (label, img) in
                                                                    PRECOND_IMAGES
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    n = size(sim.prob.A, 1)
    P = two_level_preconditioner(sim; block=4)
    y = zeros(n)

    # Symmetry: x'(M\z) == z'(M\x) for the operator ldiv! applies.
    for seed in (1, 2)
        x = [sin(seed * i) for i in 1:n]
        z = [cos(seed * i / 3) for i in 1:n]
        Mx = ldiv!(similar(y), P, x)
        Mz = ldiv!(similar(y), P, z)
        @test dot(z, Mx) ≈ dot(x, Mz) rtol = 1e-10
        # Definiteness: the smoother term alone already makes this strict.
        @test dot(x, Mx) > 0
    end

    # The `x / λmax` term divides by a Gershgorin bound, which must actually
    # bound: no column of A may have a larger absolute sum.
    @test 1 / P.inv_lambda >= maximum(sum(abs, sim.prob.A; dims=1))
end

@testset "the Gershgorin bound is above the true spectral radius" begin
    # Small enough to diagonalise, so the cheap column-sum check above is
    # anchored to the quantity it stands in for at least once.
    img = ones(Bool, 6, 6, 6)
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    P = two_level_preconditioner(sim; block=3)
    @test 1 / P.inv_lambda >= eigmax(Matrix(sim.prob.A))
end

@testset "preconditioned CG lands on the same solution — $(label)" for (label, img) in
                                                                      PRECOND_IMAGES
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    P = two_level_preconditioner(sim; block=4)
    # `abstol` has to be forced down as well: LinearSolve defaults it to
    # sqrt(eps(Float64)) = 1.5e-8, and Krylov stops at `abstol + reltol·‖r₀‖`,
    # so at a tight `reltol` the absolute term is what actually ends both runs —
    # and it ends them at different places, because a preconditioned CG measures
    # the residual in the M⁻¹ norm rather than the 2-norm.
    plain = solve(sim.prob, KrylovJL_CG(); reltol=1e-12, abstol=1e-14, verbose=false)
    prec = solve(sim.prob, KrylovJL_CG(); Pl=P, reltol=1e-12, abstol=1e-14, verbose=false)

    tau_plain = tortuosity(reconstruct_field(plain.u, img), img; axis=:x)
    tau_prec = tortuosity(reconstruct_field(prec.u, img), img; axis=:x)
    @test tau_prec ≈ tau_plain rtol = 1e-8

    # Compare the flux-carrying part of the field, not the raw vectors: two
    # equally converged solutions differ freely on clusters that reach neither
    # Dirichlet face, because nothing in the residual constrains them.
    trimmed = Array{Bool}(Imaginator.trim_nonpercolating_paths(img; axis=:x))
    live = trimmed[img]
    if any(live)
        @test prec.u[live] ≈ plain.u[live] rtol = 1e-6
    end
end

# What the coarse space buys is not a constant factor: unpreconditioned CG needs
# more iterations as the image gets larger, and the coarse correction is what
# stops that growth. Two sizes are the smallest evidence of it, and this is the
# property that would silently disappear if the coarse operator were built from
# the wrong matrix while still being definite enough to converge.
@testset "the coarse space removes the growth in iteration count" begin
    counts = map((48, 64)) do n
        img = Array{Bool}(Imaginator.blobs(; shape=(n, n, n), porosity=0.5, blobiness=1, seed=42))
        sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
        P = two_level_preconditioner(sim; block=8)
        plain = solve(sim.prob, KrylovJL_CG(); reltol=1e-8, verbose=false)
        prec = solve(sim.prob, KrylovJL_CG(); Pl=P, reltol=1e-8, verbose=false)
        return (plain.iters, prec.iters)
    end
    (plain_48, prec_48), (plain_64, prec_64) = counts
    @test plain_64 > 1.25 * plain_48        # measured 428 -> 574
    @test prec_64 < 1.10 * prec_48          # measured 215 -> 222
    @test prec_64 < plain_64 / 2
end

@testset "a shift is required" begin
    img = ones(Bool, 16, 16, 16)
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    @test_throws AssertionError two_level_preconditioner(sim; block=4, shift=0.0)
end

@testset "degenerate pore spaces yield no coarse space" begin
    # Interior voxels with no neighbour at all: every column of A is empty, so
    # there is no coarse unknown anywhere and no preconditioner to build.
    img = falses(9, 9, 9)
    img[3:2:7, 3:2:7, 3:2:7] .= true
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    @test nnz(sim.prob.A) == 0
    @test two_level_preconditioner(sim; block=4) === nothing

    # Move the same voxels onto the inlet face and they become Dirichlet nodes
    # with a pinned unit diagonal, which is a coarse unknown after all.
    img2 = falses(9, 9, 9)
    img2[1, 3:2:7, 3:2:7] .= true
    sim2 = SteadyDiffusionProblem(img2; axis=:x, gpu=false, warn_nonpercolating=false)
    @test two_level_preconditioner(sim2; block=4) isa TwoLevelPreconditioner
end

@testset "variable diffusivity" begin
    img = Array{Bool}(Imaginator.blobs(; shape=(32, 32, 32), porosity=0.5, blobiness=1, seed=42))
    D = zeros(size(img))
    D[img] .= range(0.5, 2.0; length=count(img))
    sim = SteadyDiffusionProblem(img; axis=:x, D=D, gpu=false, warn_nonpercolating=false)
    P = two_level_preconditioner(sim; block=4)
    W = prolongation(P, size(sim.prob.A, 1))
    @test Matrix(W' * sim.prob.A * W) ≈ Matrix(W' * sim.prob.A * W)' rtol = 1e-12

    plain = solve(sim.prob, KrylovJL_CG(); reltol=1e-12, abstol=1e-14, verbose=false)
    prec = solve(sim.prob, KrylovJL_CG(); Pl=P, reltol=1e-12, abstol=1e-14, verbose=false)
    @test Tortuosity.effective_diffusivity(reconstruct_field(prec.u, img), img; axis=:x) ≈
          Tortuosity.effective_diffusivity(reconstruct_field(plain.u, img), img; axis=:x) rtol =
        1e-8
end

@testset "show" begin
    img = ones(Bool, 16, 16, 16)
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    P = two_level_preconditioner(sim; block=4)
    @test occursin("TwoLevelPreconditioner", sprint(show, P))
    @test occursin("nc=$(P.nc)", sprint(show, P))
    @test size(P) == (size(sim.prob.A, 1), size(sim.prob.A, 1))
    @test eltype(P) === Float64
end
