# The two-level preconditioner has to compose with the matrix-free operator, and
# the assembled path is the executable specification of what it must build: the
# same coarse space, the same Gershgorin bound, and — the property that actually
# decides whether CG converges — the same correction applied to the same
# residual. CPU only; the GPU backends have their own file.

using Test
using LinearAlgebra
using Random
using SparseArrays
using Tortuosity
using Tortuosity:
    Imaginator,
    TwoLevelPreconditioner,
    two_level_preconditioner,
    build_steady_operator,
    build_steady_system,
    reconstruct_field,
    tortuosity

# Percolating blobs, a duct, a slab with a detached cube whose blocks carry no
# coarse unknown, and the zero-degree fixture the elimination convention is
# delicate on. Every testset forces `block=4` so that even the smallest of them
# is aggregated into more than one block — a single aggregate would make the
# coarse operator a scalar and hide any slot arithmetic that went wrong.
function mfp_fixtures()
    imgs = Tuple{String,Array{Bool,3}}[]
    for (porosity, blobiness, seed) in ((0.5, 1, 42), (0.45, 2, 7))
        img = Array{Bool}(
            Imaginator.blobs(;
                shape=(32, 32, 32), porosity=porosity, blobiness=blobiness, seed=seed
            ),
        )
        push!(imgs, ("blob 32^3 seed=$(seed)", img))
    end
    push!(imgs, ("duct 24^3", let img = falses(24, 24, 24)
        img[:, 9:16, 9:16] .= true
        img
    end))
    push!(imgs, ("slab plus detached cube", let img = falses(24, 24, 24)
        img[:, 1:8, 1:8] .= true            # spans the x axis
        img[8:14, 14:20, 14:20] .= true     # reaches neither face
        img
    end))
    push!(imgs, ("isolated voxels 8x6x6", let img = zeros(Bool, 8, 6, 6)
        img[:, 3:4, 3:4] .= true
        img[4, 1, 1] = true
        img[1, 6, 6] = true
        img[8, 1, 6] = true
        img
    end))
    return imgs
end

# Smooth, strictly positive and different in every voxel, so no two harmonic
# means coincide by accident.
mfp_diffusivity(img) =
    [0.5 + 0.1i + 0.02j + 0.003k for i in 1:size(img, 1), j in 1:size(img, 2), k in 1:size(img, 3)]

mfp_diffusivity_cases(img) = (("uniform D", nothing), ("variable D", mfp_diffusivity(img)))

# The pair of preconditioners a fixture yields, one per representation.
function mfp_pair(img, axis, D; kwargs...)
    nnodes = count(img)
    A, _ = build_steady_system(img; nnodes=nnodes, axis=axis, D=D)
    op, _ = build_steady_operator(img; nnodes=nnodes, axis=axis, D=D)
    P_asm = two_level_preconditioner(A, img; kwargs...)
    P_mf = two_level_preconditioner(op, img; kwargs...)
    return A, op, P_asm, P_mf
end

const MFP_IMAGES = mfp_fixtures()
const MFP_AXES = (:x, :y, :z)

@testset "aggregates into more than one block — $(label)" for (label, img) in MFP_IMAGES
    _, _, _, P = mfp_pair(img, :x, nothing; block=4)
    @test P isa TwoLevelPreconditioner
    @test P.nc > 1
end

# --- Coarse-space parity ---
#
# The coarse operator is accumulated with atomics in both paths, and the grid
# pass reaches a given slot in a different thread order than the pass over
# stored entries does, so the two stencils agree to rounding rather than to the
# last bit. Measured worst case over these fixtures is 1.6e-13 relative on the
# applied correction, dominated by the detached cube whose coarse rows are
# nearly null.

@testset "the coarse space matches the assembled path" begin
    for (label, img) in MFP_IMAGES
        for (dlabel, D) in mfp_diffusivity_cases(img), axis in MFP_AXES
            @testset "$(label) — $(dlabel) — axis=$(axis)" begin
                A, op, P_asm, P_mf = mfp_pair(img, axis, D; block=4)
                @test P_mf isa TwoLevelPreconditioner
                @test P_mf.nc == P_asm.nc
                @test P_mf.block == P_asm.block
                @test Array(P_mf.agg) == Array(P_asm.agg)
                @test P_mf.inv_lambda ≈ P_asm.inv_lambda rtol = 1e-15

                n = size(A, 1)
                rng = MersenneTwister(2718)
                for _ in 1:3
                    x = randn(rng, n)
                    y_asm = ldiv!(zeros(n), P_asm, x)
                    y_mf = ldiv!(zeros(n), P_mf, x)
                    @test isapprox(y_mf, y_asm; rtol=1e-12)
                end
            end
        end
    end
end

# The case that exposed the round-off floor, kept as its regression test.
#
# A block holding nothing but an enclosed cluster has a coarse diagonal that
# cancels to exactly zero, and the two paths sum it in different orders. At
# `block=2` on this fixture the assembled path landed on `+2.22e-16` and the
# matrix-free path on `0.0`, so a `> 0` test kept the block on one path and
# dropped it on the other — leaving the assembled path with a coarse row whose
# diagonal was 1e-16 and a correction 1e15 times its input. The two paths
# disagreed by 18 orders of magnitude on the same residual.
#
# Small blocks are what make this reachable: they are the ones that can hold a
# cluster and nothing else. Hence `block=2` here where the rest of the file uses 4.
@testset "an enclosed cluster is dropped by both paths — $(dlabel)" for (dlabel, D) in
    mfp_diffusivity_cases(MFP_IMAGES[2][2])

    img = MFP_IMAGES[2][2]                      # blob 32^3 seed=7
    A, _, P_asm, P_mf = mfp_pair(img, :x, D; block=2, max_coarse=10^9)
    @test P_mf.nc == P_asm.nc
    @test Array(P_mf.agg) == Array(P_asm.agg)

    n = size(A, 1)
    x = randn(MersenneTwister(1), n)
    y_asm = ldiv!(zeros(n), P_asm, x)
    y_mf = ldiv!(zeros(n), P_mf, x)
    @test isapprox(y_asm, y_mf; rtol=1e-10)
    # A near-null coarse row shows up here and nowhere else: bounded, not 1e15.
    @test norm(y_asm, Inf) < 1e3 * norm(x, Inf)
end

# The hierarchy is built from the coarse operator, which the two paths accumulate
# differently, so it is the part of the coarse space that could agree at the top
# and drift apart further down. Same `block=4` as everything else here; the
# ceiling is what is lowered, so several levels come out of these small fixtures.
@testset "a hierarchical coarse space matches the assembled path" begin
    for (label, img) in MFP_IMAGES
        for (dlabel, D) in mfp_diffusivity_cases(img)
            @testset "$(label) — $(dlabel)" begin
                A, _, P_asm, P_mf = mfp_pair(img, :x, D; block=4, max_coarse=8)
                # The smallest fixtures have a coarse space under the ceiling
                # already, and are here for the parity check rather than the depth.
                @test isempty(P_mf.levels) == (P_mf.nc <= 8)
                @test length(P_mf.levels) == length(P_asm.levels)
                for (Lm, La) in zip(P_mf.levels, P_asm.levels)
                    @test Lm.parent == La.parent
                    @test size(Lm.A) == size(La.A)
                    @test Lm.A ≈ La.A rtol = 1e-12
                end

                n = size(A, 1)
                rng = MersenneTwister(2718)
                for _ in 1:3
                    x = randn(rng, n)
                    @test isapprox(ldiv!(zeros(n), P_mf, x), ldiv!(zeros(n), P_asm, x);
                                   rtol=1e-12)
                end
            end
        end
    end
end

# The smoother term divides by a Gershgorin bound the assembled path reads off
# its stored values. The operator stores none and folds a maximum over the node
# diagonals into the coarse pass instead, which is the same number because the
# diagonals are the only positive entries of the matrix it stands for.
@testset "inv_lambda is the assembled bound — $(label)" for (label, img) in MFP_IMAGES
    for (dlabel, D) in mfp_diffusivity_cases(img)
        A, _, _, P_mf = mfp_pair(img, :x, D; block=4)
        @test P_mf.inv_lambda ≈ 1 / (2 * maximum(nonzeros(A))) rtol = 1e-15
    end
end

# --- Solving with it ---

@testset "preconditioned CG behaves identically on both paths" begin
    # Large enough that the coarse space is worth its cost: at 48³ it only just
    # halves the iteration count, which leaves the guard below nothing to see.
    img = Array{Bool}(
        Imaginator.blobs(; shape=(64, 64, 64), porosity=0.5, blobiness=1, seed=42)
    )
    nnodes = count(img)
    A, b_asm = build_steady_system(img; nnodes=nnodes, axis=:x)
    op, b_mf = build_steady_operator(img; nnodes=nnodes, axis=:x)
    P_asm = two_level_preconditioner(A, img; block=8)
    P_mf = two_level_preconditioner(op, img; block=8)

    sol_asm = solve(
        Tortuosity.LinearProblem(A, copy(b_asm)), KrylovJL_CG();
        Pl=P_asm, reltol=1e-10, abstol=1e-14, verbose=false,
    )
    sol_mf = solve(
        Tortuosity.LinearProblem(op, copy(b_mf)), KrylovJL_CG();
        Pl=P_mf, reltol=1e-10, abstol=1e-14, verbose=false,
    )
    @test Symbol(sol_asm.retcode) === :Success
    @test Symbol(sol_mf.retcode) === :Success
    # A coarse space that differed anywhere structurally would move this by far
    # more than one iteration.
    @test abs(sol_mf.iters - sol_asm.iters) <= 1

    tau_asm = tortuosity(reconstruct_field(sol_asm.u, img), img; axis=:x)
    tau_mf = tortuosity(reconstruct_field(sol_mf.u, img), img; axis=:x)
    @test tau_mf ≈ tau_asm rtol = 1e-8

    # An inert preconditioner — a coarse correction that is zero, or one the
    # `x / λmax` term swamps — would still converge, to the same answer, in the
    # unpreconditioned iteration count. That is the failure this rules out.
    sol_plain = solve(
        Tortuosity.LinearProblem(op, copy(b_mf)), KrylovJL_CG();
        reltol=1e-10, abstol=1e-14, verbose=false,
    )
    @test Symbol(sol_plain.retcode) === :Success
    @test sol_mf.iters < sol_plain.iters / 2
    tau_plain = tortuosity(reconstruct_field(sol_plain.u, img), img; axis=:x)
    @test tau_mf ≈ tau_plain rtol = 1e-8
end

# --- Degenerate pore spaces ---

@testset "degenerate pore spaces yield no coarse space on either path" begin
    # Interior voxels with no neighbour at all: every column is empty, so no
    # block carries a coarse unknown. The assembled path sees `nnz(A) == 0` up
    # front; the matrix-free one finds it out when the coarse operator is empty.
    img = falses(9, 9, 9)
    img[3:2:7, 3:2:7, 3:2:7] .= true
    A, op, P_asm, P_mf = mfp_pair(img, :x, nothing; block=4)
    @test nnz(A) == 0
    @test P_asm === nothing
    @test P_mf === nothing

    # An all-solid image is an operator of order zero.
    solid = falses(8, 8, 8)
    A0, _ = build_steady_system(solid; nnodes=0, axis=:x)
    op0, _ = build_steady_operator(solid; nnodes=0, axis=:x)
    @test size(op0) == (0, 0)
    @test two_level_preconditioner(A0, solid; block=4) === nothing
    @test two_level_preconditioner(op0, solid; block=4) === nothing

    # Move the same isolated voxels onto the inlet face and they become
    # Dirichlet nodes with a pinned unit diagonal, which is a coarse unknown
    # after all — both paths have to notice.
    face = falses(9, 9, 9)
    face[1, 3:2:7, 3:2:7] .= true
    _, _, Pf_asm, Pf_mf = mfp_pair(face, :x, nothing; block=4)
    @test Pf_asm isa TwoLevelPreconditioner
    @test Pf_mf isa TwoLevelPreconditioner
    @test Pf_mf.nc == Pf_asm.nc
    @test Pf_mf.inv_lambda == Pf_asm.inv_lambda
end

@testset "a shift is required of the matrix-free path too" begin
    img = ones(Bool, 16, 16, 16)
    op, _ = build_steady_operator(img; nnodes=count(img), axis=:x)
    @test_throws AssertionError two_level_preconditioner(op, img; block=4, shift=0.0)
end
