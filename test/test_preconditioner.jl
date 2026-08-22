# The two-level preconditioner is the one piece of this solver that can be
# wrong without failing: an inaccurate coarse solve makes CG stall and return a
# plausible tortuosity rather than an error. These tests pin the two properties
# that rule that out — the operator really is `W'AW` for the `W` the aggregates
# describe, and applying it is symmetric positive definite, which is what makes
# preconditioned CG converge to the same solution as unpreconditioned CG.

using Test
using LinearAlgebra
using Random
using SparseArrays
using Tortuosity
using Tortuosity:
    Imaginator,
    TwoLevelPreconditioner,
    two_level_preconditioner,
    DEFAULT_COARSE_SHIFT,
    DEFAULT_COARSE_BLOCK,
    COARSE_RATIO,
    _COARSE_SLOTS,
    _coarse_operator,
    _coarse_diagonal_floor,
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

# The coarse-size ceiling used to be met by growing the block, which is what made
# the method's iteration count track the image edge. It is met by adding grids
# under the coarse space instead, and the block edge is now the same at every
# size. These pin that the ceiling is still honoured the new way.
@testset "the ceiling is met by depth, not by a coarser block" begin
    img = ones(Bool, 24, 24, 24)
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)

    # Default block, whatever the image: nothing reads the image to choose it.
    for shape in ((24, 24, 24), (12, 24, 36))
        s = SteadyDiffusionProblem(ones(Bool, shape...); axis=:x, gpu=false,
                                   warn_nonpercolating=false)
        @test two_level_preconditioner(s).block == DEFAULT_COARSE_BLOCK
    end

    # A coarse space over the ceiling keeps its size and gains levels below it,
    # each one `COARSE_RATIO` coarser per edge, down to a direct solve that fits.
    P = two_level_preconditioner(sim; block=2, max_coarse=8)
    @test P.block == 2
    @test P.nc > 8                                  # the coarse space itself is not shrunk
    @test !isempty(P.levels)
    sizes = [size(L.A, 1) for L in P.levels]
    @test sizes[1] == P.nc
    @test issorted(sizes; rev=true) && allunique(sizes)
    @test size(P.fact, 1) <= 8                      # the direct solve honours the ceiling
    for (a, b) in zip(sizes, [sizes[2:end]; size(P.fact, 1)])
        @test b < a                                 # every level actually coarsens
        @test b >= a ÷ (2 * COARSE_RATIO^3)         # and not by more than the ratio allows
    end

    # Under the ceiling nothing is interposed at all, which is the two-level
    # method exactly as it was.
    @test isempty(two_level_preconditioner(sim; block=2, max_coarse=32_000).levels)
end

# Each level's operator has to be the Galerkin product of the one above it. A
# hierarchy built from the wrong matrix still converges — more slowly — so
# nothing else here would catch it.
@testset "each coarse level is W'AW of the level above" begin
    img = Array{Bool}(Imaginator.blobs(; shape=(32, 32, 32), porosity=0.5, blobiness=1, seed=42))
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    P = two_level_preconditioner(sim; block=2, max_coarse=32)
    @test length(P.levels) >= 2

    for (l, L) in enumerate(P.levels)
        parent = L.parent
        rows = [i for i in eachindex(parent) if parent[i] > 0]
        nnext = l < length(P.levels) ? size(P.levels[l + 1].A, 1) : size(P.fact, 1)
        W = sparse(rows, [Int(parent[i]) for i in rows], ones(length(rows)),
                   length(parent), nnext)
        Anext = W' * L.A * W
        if l < length(P.levels)
            @test Matrix(Anext) ≈ Matrix(P.levels[l + 1].A) rtol = 1e-10
        else
            # The coarsest operator is only held as its factorisation, so it is
            # checked the way `P.fact` is above: solve with it and multiply back.
            v = collect(range(-1.0, 1.0; length=nnext))
            @test Anext * (P.fact \ v) ≈ v rtol = 1e-8
        end
    end
end

# A block holding nothing but a cluster enclosed within it has a coarse diagonal
# that cancels to *exactly* zero, so in floating point it lands on a residue whose
# sign is whichever way the threads happened to race. Keeping such a block is not
# a rounding-sized mistake: its coarse row has a ~1e-16 diagonal, and the coarse
# solve then amplifies along that direction by ~1e16. `> 0` is therefore not a
# safe test, and this pins the floor that replaced it.
@testset "a coarse diagonal at round-off is not a coarse unknown" begin
    bs, maximum_diagonal = 2, 25.0
    floor = _coarse_diagonal_floor(bs, maximum_diagonal)

    # Well clear of both the residue it must reject and the smallest diagonal a
    # real coarse unknown can carry, which is one edge weight.
    @test 1e-13 < floor < 1e-6

    nbx, nby, nbz = 2, 2, 1
    nc0 = nbx * nby * nbz
    S = zeros(Float64, _COARSE_SLOTS * nc0)
    diagonals = [10.0,          # an ordinary block
                 2.220446e-16,  # the measured residue of a cancelling sum
                 0.0,           # an empty block
                 2 * floor]     # small, but genuinely there
    for (a, d) in enumerate(diagonals)
        S[(a - 1) * _COARSE_SLOTS + 1] = d
    end

    Ac, remap = _coarse_operator(S, nc0, nbx, nbx * nby, DEFAULT_COARSE_SHIFT, floor)
    @test remap == Int32[1, 0, 0, 2]        # blocks 2 and 3 carry no coarse unknown
    @test size(Ac) == (2, 2)
    @test diag(Ac) ≈ [10.0, 2 * floor] .* (1 + DEFAULT_COARSE_SHIFT)

    # The residue is rejected on its magnitude, not its sign: the same block with
    # the sign the other path happened to produce is dropped too.
    S[_COARSE_SLOTS + 1] = -2.220446e-16
    @test _coarse_operator(S, nc0, nbx, nbx * nby, DEFAULT_COARSE_SHIFT, floor)[2] ==
          Int32[1, 0, 0, 2]
end

# The same property end to end. Which side of zero the residue lands on is not
# something a fixture can pin down — here it comes out negative, so `> 0` would
# have dropped this block too — so this guards the invariant rather than
# reproducing the defect. The discriminating cases are the testset above and the
# `block=2` parity case in `test_matrixfree_precond.jl`.
@testset "an enclosed cluster leaves no near-null coarse row" begin
    img = falses(16, 16, 16)
    img[:, 1:4, 1:4] .= true            # spans x, so it carries the flux
    img[10:12, 10:12, 10:12] .= true    # detached, and inside the block 9:16^3
    # Variable D so the weights differ and the cancelling sum genuinely rounds.
    D = [0.5 + 0.11i + 0.023j + 0.0031k for i in 1:16, j in 1:16, k in 1:16]
    D[.!img] .= 0.0
    sim = SteadyDiffusionProblem(img; axis=:x, D=D, gpu=false, warn_nonpercolating=false)
    n = size(sim.prob.A, 1)
    P = two_level_preconditioner(sim; block=8)

    W = prolongation(P, n)
    Ac = Matrix(W' * sim.prob.A * W)
    @test minimum(diag(Ac)) > _coarse_diagonal_floor(P.block, maximum(diag(sim.prob.A)))

    # The detached cube is in the null space of A, so every one of its voxels
    # must have been dropped rather than given a coarse unknown of its own.
    agg = Array(P.agg)
    cube = falses(16, 16, 16)
    cube[10:12, 10:12, 10:12] .= true
    @test all(iszero, agg[cube[img]])

    x = [sin(3.0i) for i in 1:n]
    y = ldiv!(zeros(n), P, x)
    @test norm(y, Inf) < 1e3 * norm(x, Inf)   # was 1e15 times larger when kept
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

# A V-cycle is only a valid CG preconditioner while it stays symmetric positive
# definite, and it is easy to lose: an unsymmetric cycle, or a smoother weight
# past the stability bound, both still "work" in the sense of returning a vector.
# CG then either stalls or rejects the operator outright.
@testset "a hierarchical coarse solve stays SPD — $(label)" for (label, img) in
                                                                PRECOND_IMAGES
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    n = size(sim.prob.A, 1)
    P = two_level_preconditioner(sim; block=2, max_coarse=8)
    @test !isempty(P.levels)

    for seed in (1, 2)
        x = [sin(seed * i) for i in 1:n]
        z = [cos(seed * i / 3) for i in 1:n]
        Mx = ldiv!(zeros(n), P, x)
        Mz = ldiv!(zeros(n), P, z)
        @test dot(z, Mx) ≈ dot(x, Mz) rtol = 1e-10
        @test dot(x, Mx) > 0
    end

    # And it still has to be the same solve: CG with it lands where CG without
    # it does.
    plain = solve(sim.prob, KrylovJL_CG(); reltol=1e-12, abstol=1e-14, verbose=false)
    prec = solve(sim.prob, KrylovJL_CG(); Pl=P, reltol=1e-12, abstol=1e-14, verbose=false)
    @test tortuosity(reconstruct_field(prec.u, img), img; axis=:x) ≈
          tortuosity(reconstruct_field(plain.u, img), img; axis=:x) rtol = 1e-8
end

# The reason the hierarchy exists. The coarse space is held at a fixed ratio to
# the fine grid, so it grows with the image, and the grids below it are what keep
# solving it affordable. Growing the block instead — which is what this replaced
# — costs iterations in proportion to the image edge.
#
# `max_coarse` is lowered so that two tractable sizes straddle the ceiling the
# way 400³ and 800³ straddle the released one.
@testset "iteration count does not track the image edge" begin
    counts = map((48, 96)) do n
        img = Array{Bool}(Imaginator.blobs(; shape=(n, n, n), porosity=0.5, blobiness=1, seed=42))
        sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
        P = two_level_preconditioner(sim; max_coarse=64)
        @test P.block == DEFAULT_COARSE_BLOCK       # the block did not grow with n
        @test !isempty(P.levels)
        return solve(sim.prob, KrylovJL_CG(); Pl=P, reltol=1e-8, verbose=false).iters
    end
    small, large = counts
    # Doubling the edge used to roughly double this. Measured 217 -> 219.
    @test large < 1.25 * small
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

# The aggregate inversion reserves each thread-chunk's output positions before
# writing, so that a cell's member nodes land in ascending order no matter which
# thread files them. That property is what lets the chunk count be capped for
# memory without changing a single number, so it is pinned here rather than
# trusted: the scratch tables are `nc x nchunks`, which on a large image and a
# many-threaded host would otherwise reach a gigabyte of host memory.
@testset "aggregate inversion does not depend on the chunk count" begin
    rng = MersenneTwister(20260821)
    nc = 37
    agg = rand(rng, 0:nc, 5000)          # 0 marks a node in no aggregate

    # Reference: group node indices by cell, ascending, with no chunking at all.
    want = [sort([i for i in eachindex(agg) if agg[i] == a]) for a in 1:nc]

    for budget in (1, 64, 1024, 256 * 1024 * 1024)   # 1 forces a single chunk
        offsets, fine = Tortuosity._invert_aggregates(agg, nc; max_scratch_bytes=budget)
        @test length(offsets) == nc + 1
        @test offsets[1] == 1
        @test Int(offsets[end]) - 1 == count(>(0), agg)
        got = [Int.(fine[offsets[a]:(offsets[a + 1] - 1)]) for a in 1:nc]
        @test got == want
    end
end

# The coarsest solve runs once per CG iteration, so what it allocates is
# multiplied by the iteration count. `_vcycle!`'s base case writes the solution
# into the caller's vector with `ldiv!`; the obvious alternative, `e .= fact \ r`,
# is bit-identical but allocates a coarse-sized vector to throw away on every
# iteration. The reference below is the same factorisation solving the same
# system, so the bound calibrates itself rather than naming a byte count that
# would drift with SuiteSparse.
@testset "the coarsest solve allocates nothing per iteration of its own" begin
    img = Array{Bool}(Imaginator.blobs(; shape=(48, 48, 48), porosity=0.6, blobiness=1, seed=42))
    sim = SteadyDiffusionProblem(img; axis=:x, gpu=false, warn_nonpercolating=false)
    P = two_level_preconditioner(sim; block=6)
    # With no interposed grids the cycle *is* the base case, which is the branch
    # under test; a hierarchy would put smoothing allocations in the way.
    @test isempty(P.levels)

    r = randn(MersenneTwister(20260821), P.nc)
    e = similar(r)
    ldiv!(e, P.fact, r)                                  # warm both paths
    Tortuosity._vcycle!(e, P.levels, 1, r, P.fact)
    direct = @allocated ldiv!(e, P.fact, r)
    cycle = @allocated Tortuosity._vcycle!(e, P.levels, 1, r, P.fact)
    @test cycle <= 2 * direct

    # And the result is the coarse solve, not something cheaper that happens to
    # allocate less.
    @test Tortuosity._vcycle!(similar(r), P.levels, 1, r, P.fact) == P.fact \ r
end
