# Parity tests for the matrix-free steady operator. `build_steady_system` is the
# executable specification: `MaskedLaplacian`'s apply must reproduce its matrix's
# action and `build_steady_operator` must reproduce its right-hand side. CPU
# only — the GPU backends have their own file.

using Test
using Random
using SparseArrays
using LinearAlgebra
using Tortuosity
using Tortuosity: Imaginator,
    MaskedLaplacian,
    build_steady_operator,
    build_steady_system,
    _cg_workspace,
    _free!

# A duct that percolates, plus three deliberately isolated voxels: one in the
# interior (its column is dropped entirely), one on the `:x` inlet face and one
# on the `:x` outlet face (both pinned with a unit diagonal instead of their
# zero degree). Mirrors `test_impl_parity.jl`'s fixture of the same shape.
function isolated_voxel_image()
    img = zeros(Bool, 8, 6, 6)
    img[:, 3:4, 3:4] .= true
    img[4, 1, 1] = true
    img[1, 6, 6] = true
    img[8, 1, 6] = true
    return img
end

# The seven images the assembled path is pinned bit-identical on in
# `test_impl_parity.jl`, repeated here so this file stands alone. Between them
# they carry zero-degree pore voxels in the interior and on both Dirichlet
# faces, which is where the elimination convention is delicate.
function matrixfree_fixtures()
    imgs = Tuple{String,Array{Bool,3}}[]
    push!(imgs, ("open 10^3", ones(Bool, 10, 10, 10)))
    push!(imgs, ("open 16^3", ones(Bool, 16, 16, 16)))
    let img = ones(Bool, 12, 12, 12)
        img[:, :, 1:6] .= false
        push!(imgs, ("half 12^3", img))
    end
    for seed in (1, 7, 42)
        img = Array{Bool}(
            Imaginator.blobs(;
                shape=(16, 16, 16), porosity=0.55f0, blobiness=1, seed=seed
            ),
        )
        count(img) >= 4 || continue
        push!(imgs, ("blob 16^3 seed=$seed", img))
    end
    push!(imgs, ("isolated voxels 8x6x6", isolated_voxel_image()))
    return imgs
end

# Smooth, strictly positive and different in every voxel, so no two harmonic
# means coincide by accident.
node_diffusivity_grid(img) =
    [0.5 + 0.1i + 0.02j + 0.003k for i in 1:size(img, 1), j in 1:size(img, 2), k in 1:size(img, 3)]

diffusivity_cases(img) = (("uniform D", nothing), ("variable D", node_diffusivity_grid(img)))

# Six-face degree of every pore voxel, used to locate the zero-degree nodes
# rather than hardcoding their coordinates.
function zero_degree_voxels(img)
    nx, ny, nz = size(img)
    out = CartesianIndex{3}[]
    for c in CartesianIndices(img)
        img[c] || continue
        i, j, k = c.I
        deg = 0
        for (di, dj, dk) in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
            ii, jj, kk = i + di, j + dj, k + dk
            (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
            img[ii, jj, kk] && (deg += 1)
        end
        iszero(deg) && push!(out, c)
    end
    return out
end

matrixfree_images = matrixfree_fixtures()
transport_axes = (:x, :y, :z)

@testset "the fixture suite really contains the awkward nodes" begin
    @test length(matrixfree_images) == 7
    iso = zero_degree_voxels(isolated_voxel_image())
    @test any(c -> c[1] == 1, iso)                   # zero-degree inlet node
    @test any(c -> c[1] == 8, iso)                   # zero-degree outlet node
    @test any(c -> 1 < c[1] < 8, iso)                # zero-degree interior node
    @test any(!isempty(zero_degree_voxels(img)) for (_, img) in matrixfree_images)
end

# --- Apply parity ---
#
# The assembled CSC `mul!` accumulates a row in ascending column order, which is
# the order the stencil kernel walks `_NEIGHBOURS` in, so the two agree far
# tighter than the physics requires. They are not bit-identical: the diagonal
# sits between the two halves of that order but is only known once both have
# been walked, so the upper half is summed on its own and folded in as one term.
#
# Both claims are asserted on the same applies: the loose one is what the physics
# needs, the rounding-level one is what the shared summation order buys. If only
# the second goes red, that order drifted rather than the operator.

@testset "apply parity vs the assembled matrix" begin
    for (label, img) in matrixfree_images
        nnodes = count(img)
        nnodes >= 4 || continue
        for (dlabel, D) in diffusivity_cases(img), axis in transport_axes
            @testset "$(label) — $(dlabel) — axis=$(axis)" begin
                A, _ = build_steady_system(img; nnodes=nnodes, axis=axis, D=D)
                op, _ = build_steady_operator(img; nnodes=nnodes, axis=axis, D=D)
                rng = MersenneTwister(4711)
                for _ in 1:3
                    x = randn(rng, nnodes)
                    y_asm = mul!(zeros(nnodes), A, x)
                    y_mf = mul!(zeros(nnodes), op, x)
                    @test isapprox(y_mf, y_asm; rtol=1e-14)
                    @test isapprox(y_mf, y_asm; rtol=1e-15)
                end
            end
        end
    end
end

@testset "5-argument mul! applies alpha and beta like the assembled path" begin
    for (label, img) in matrixfree_images
        nnodes = count(img)
        nnodes >= 4 || continue
        for (dlabel, D) in diffusivity_cases(img)
            @testset "$(label) — $(dlabel)" begin
                A, _ = build_steady_system(img; nnodes=nnodes, axis=:x, D=D)
                op, _ = build_steady_operator(img; nnodes=nnodes, axis=:x, D=D)
                rng = MersenneTwister(1301)
                x = randn(rng, nnodes)
                y0 = randn(rng, nnodes)

                for (alpha, beta) in ((1.0, 0.0), (2.5, -0.75), (0.0, 3.0), (1.0, 1.0))
                    y_asm = mul!(copy(y0), A, x, alpha, beta)
                    y_mf = mul!(copy(y0), op, x, alpha, beta)
                    @test isapprox(y_mf, y_asm; rtol=1e-12)
                end

                # beta = 0 must not read `y`, so a dirty buffer cannot leak into
                # the answer — and a NaN one advertises the leak loudly.
                y_mf = mul!(fill(NaN, nnodes), op, x, 2.0, 0.0)
                @test !any(isnan, y_mf)
                y_asm = mul!(zeros(nnodes), A, x, 2.0, 0.0)
                @test isapprox(y_mf, y_asm; rtol=1e-12)

                # Integer scalars must not widen the arithmetic away from eltype(y).
                y_int = mul!(zeros(nnodes), op, x, 1, 0)
                @test y_int == mul!(zeros(nnodes), op, x)
            end
        end
    end
end

# --- Right-hand side ---
#
# `b` is a sum of edge weights taken in a fixed order in both paths, so nothing
# weaker than `==` is warranted on CPU Float64.

@testset "RHS parity: b matches build_steady_system exactly" begin
    for (label, img) in matrixfree_images
        nnodes = count(img)
        nnodes >= 4 || continue
        for (dlabel, D) in diffusivity_cases(img), axis in transport_axes
            @testset "$(label) — $(dlabel) — axis=$(axis)" begin
                _, b_asm = build_steady_system(img; nnodes=nnodes, axis=axis, D=D)
                _, b_mf = build_steady_operator(img; nnodes=nnodes, axis=axis, D=D)
                @test eltype(b_mf) === eltype(b_asm)
                @test length(b_mf) == nnodes
                @test b_mf == b_asm
            end
        end
    end
end

@testset "b follows the element-type keyword" begin
    img = ones(Bool, 8, 8, 8)
    nnodes = count(img)
    _, b_mf = build_steady_operator(img; nnodes=nnodes, axis=:x, T=Float32)
    _, b_asm = build_steady_system(img; nnodes=nnodes, axis=:x, T=Float32)
    @test eltype(b_mf) === Float32
    @test b_mf == b_asm
end

# --- Interface ---

@testset "MaskedLaplacian interface" begin
    img = ones(Bool, 6, 6, 6)
    nnodes = count(img)
    op, b = build_steady_operator(img; nnodes=nnodes, axis=:x)

    @test op isa MaskedLaplacian
    @test op isa AbstractMatrix
    @test size(op) == (nnodes, nnodes)
    @test size(op, 1) == nnodes
    @test size(op, 2) == nnodes
    @test eltype(op) === Float64
    @test !Tortuosity._async_return_safe(op.idx)
    @test Tortuosity._steady_workgroup(op.idx) == (64, 4, 1)
    @test length(b) == nnodes
    @test op.nnodes == nnodes
    @test size(op.idx) == size(img)

    # An operator built with variable D follows D's element type, as the
    # assembled matrix does.
    D32 = Float32.(node_diffusivity_grid(img))
    op32, _ = build_steady_operator(img; nnodes=nnodes, axis=:x, D=D32, T=Float32)
    @test eltype(op32) === Float32

    @test_throws ErrorException op[1, 1]

    x = randn(MersenneTwister(31), nnodes)
    y = mul!(zeros(nnodes), op, x)
    @test op * x == y
    # A fixed summation order per output makes repeated applies bit-identical.
    @test mul!(zeros(nnodes), op, x) == y
    @test eltype(op * Float32.(x)) === Float64

    # A dirty buffer must be overwritten, not accumulated into: a kernel that
    # missed this would corrupt every Krylov iteration after the first.
    @test mul!(fill(1e6, nnodes), op, x) == y

    io = IOBuffer()
    show(io, op)
    s = String(take!(io))
    @test occursin("MaskedLaplacian", s)
    @test occursin("$(nnodes)×$(nnodes)", s)
    @test occursin("6×6×6", s)
    @test occursin("uniform", s)

    @test _free!(op) === nothing
end

@testset "an all-solid image gives a zero-sized operator" begin
    img = zeros(Bool, 4, 4, 4)
    op, b = build_steady_operator(img; nnodes=0, axis=:x)
    @test size(op) == (0, 0)
    @test isempty(b)
    @test isempty(mul!(Float64[], op, Float64[]))
    @test isempty(op * Float64[])
end

# --- Zero-degree nodes ---
#
# Elimination leaves a boundary node its original diagonal, except where that
# diagonal is zero, in which case the node is pinned with a unit one. A free
# node with no neighbours instead loses its column entirely, hence its row.

@testset "zero-degree nodes follow the assembled convention" begin
    img = isolated_voxel_image()
    nnodes = count(img)
    A, _ = build_steady_system(img; nnodes=nnodes, axis=:x)
    op, _ = build_steady_operator(img; nnodes=nnodes, axis=:x)
    dense = Array(A)
    ordinals = cumsum(vec(img))
    lin = LinearIndices(img)

    iso = zero_degree_voxels(img)
    @test length(iso) == 3
    interior = filter(c -> 1 < c[1] < size(img, 1), iso)
    faces = filter(c -> c[1] == 1 || c[1] == size(img, 1), iso)
    @test length(interior) == 1
    @test length(faces) == 2

    x = randn(MersenneTwister(19), nnodes)
    y_mf = mul!(zeros(nnodes), op, x)
    y_asm = mul!(zeros(nnodes), A, x)

    p = ordinals[lin[interior[1]]]
    @test all(iszero, dense[p, :])           # the assembled row really is empty
    @test all(iszero, dense[:, p])
    @test y_asm[p] == 0.0
    @test y_mf[p] == 0.0

    for c in faces
        q = ordinals[lin[c]]
        @test dense[q, q] == 1.0             # unit diagonal, not the zero degree
        @test count(!iszero, dense[q, :]) == 1
        @test y_mf[q] == x[q]
        @test y_mf[q] == y_asm[q]
    end

    # Every other node whose row is a lone diagonal is a pure rescaling too —
    # the assembled matrix arbitrates the value, so nothing is hardcoded.
    lone_diagonals = 0
    for q in 1:nnodes
        count(!iszero, dense[q, :]) == 1 && !iszero(dense[q, q]) || continue
        @test y_mf[q] == dense[q, q] * x[q]
        lone_diagonals += 1
    end
    @test lone_diagonals > length(faces)
end

# --- LinearSolve integration ---

@testset "_cg_workspace mirrors Krylov's CgWorkspace for a MaskedLaplacian" begin
    # The operator plugs into the same `init_cacheval` hook `PortableSparseCSC`
    # uses, so it inherits the same dependency on `CgWorkspace`'s field list: a
    # new length-n vector would arrive empty and the solver would read off the
    # end of it. Compare against a constructor-built workspace so a Krylov
    # upgrade that adds one is caught here rather than at the frontier sizes.
    Krylov = Tortuosity.LinearSolve.Krylov
    img = ones(Bool, 4, 4, 4)
    n = count(img)
    op, b = build_steady_operator(img; nnodes=n, axis=:x)
    u = zeros(n)
    ours = _cg_workspace(op, b, u)
    theirs = Krylov.CgWorkspace(n, n, Vector{Float64})

    @test typeof(ours) === typeof(theirs)
    @test ours.x === u
    @test (ours.m, ours.n) == (theirs.m, theirs.n)
    @test (ours.m, ours.n) == size(op)
    for f in fieldnames(typeof(theirs))
        getfield(theirs, f) isa AbstractVector || continue
        @test length(getfield(ours, f)) == length(getfield(theirs, f))
    end
end

@testset "solving through LinearSolve leaves the solution in the cache's own u" begin
    # Proves the `init_cacheval` specialization is the one that fires: the
    # generic path allocates the workspace's `x` and lets LinearSolve replace it
    # afterwards, so `cacheval.x === cache.u` holds only when ours built it.
    img = ones(Bool, 8, 8, 8)
    n = count(img)
    op, b = build_steady_operator(img; nnodes=n, axis=:x)
    prob = Tortuosity.LinearProblem(op, b)
    cache = Tortuosity.LinearSolve.init(prob, KrylovJL_CG(); reltol=1e-10)
    sol = solve!(cache)
    @test cache.cacheval.x === cache.u
    @test Symbol(sol.retcode) === :Success
end

# --- End to end ---

@testset "end-to-end solve agrees with the assembled path" begin
    cases = Tuple{String,Array{Bool,3}}[("open 10^3", ones(Bool, 10, 10, 10))]
    let img = ones(Bool, 12, 12, 12)
        img[:, :, 1:6] .= false
        push!(cases, ("half 12^3", img))
    end

    for (label, img) in cases, (dlabel, D) in diffusivity_cases(img)
        @testset "$(label) — $(dlabel)" begin
            nnodes = count(img)
            A, b_asm = build_steady_system(img; nnodes=nnodes, axis=:x, D=D)
            op, b_mf = build_steady_operator(img; nnodes=nnodes, axis=:x, D=D)
            @test b_mf == b_asm

            # CG converges on the residual, so the tolerance it is given has to
            # sit well below the agreement being asserted: the error it leaves
            # behind is that residual amplified by the operator's conditioning.
            prob_asm = Tortuosity.LinearProblem(A, copy(b_asm))
            prob_mf = Tortuosity.LinearProblem(op, copy(b_mf))
            sol_asm = solve(prob_asm, KrylovJL_CG(); reltol=1e-12)
            sol_mf = solve(prob_mf, KrylovJL_CG(); reltol=1e-12)
            @test Symbol(sol_asm.retcode) === :Success
            @test Symbol(sol_mf.retcode) === :Success
            @test isapprox(sol_mf.u, sol_asm.u; rtol=1e-8)

            tau_asm = tortuosity(reconstruct_field(sol_asm.u, img), img; axis=:x)
            tau_mf = tortuosity(reconstruct_field(sol_mf.u, img), img; axis=:x)
            @test isapprox(tau_mf, tau_asm; rtol=1e-8)

            # The apply's summation order is fixed and every output is owned by
            # one thread, so two identical solves agree to the last bit.
            prob_again = Tortuosity.LinearProblem(op, copy(b_mf))
            sol_again = solve(prob_again, KrylovJL_CG(); reltol=1e-12)
            tau_again = tortuosity(reconstruct_field(sol_again.u, img), img; axis=:x)
            @test sol_again.u == sol_mf.u
            @test tau_again == tau_mf
        end
    end
end

@testset "the matrixfree keyword routes construction without moving the default" begin
    img = trues(24, 14, 14)
    img[9:12, 4:7, 4:7] .= false
    img[3, 3, 3] = false

    # The public constructor requires `D` to vanish on the solids, which the
    # low-level fixtures do not bother with.
    for (dlabel, D0) in diffusivity_cases(img), axis in transport_axes
        D = isnothing(D0) ? nothing : D0 .* img
        @testset "$(dlabel) — axis=$(axis)" begin
            sim = SteadyDiffusionProblem(img; axis=axis, D=D)
            simf = SteadyDiffusionProblem(img; axis=axis, D=D, matrixfree=true)

            # The default has to stay exactly where it was.
            @test sim.prob.A isa SparseMatrixCSC
            @test simf.prob.A isa Tortuosity.MaskedLaplacian
            @test size(simf.prob.A) == size(sim.prob.A)
            @test eltype(simf.prob.b) === eltype(sim.prob.b)
            @test simf.prob.b == sim.prob.b
            @test simf.img == sim.img
            @test simf.axis === sim.axis

            sol = solve(sim.prob, KrylovJL_CG(); reltol=1e-12)
            solf = solve(simf.prob, KrylovJL_CG(); reltol=1e-12)
            tau = tortuosity(reconstruct_field(sol.u, img), img; axis=axis)
            tauf = tortuosity(reconstruct_field(solf.u, img), img; axis=axis)
            @test isapprox(tauf, tau; rtol=1e-8)
        end
    end

    @test occursin("matrix-free", sprint(show, SteadyDiffusionProblem(img; axis=:x, matrixfree=true)))
    @test occursin("assembled", sprint(show, SteadyDiffusionProblem(img; axis=:x)))
end

@testset "mul! rejects a mismatched vector instead of writing past the end" begin
    img = trues(6, 6, 6)
    img[3, 3, 3] = false
    nnodes = count(img)
    A, _ = build_steady_system(img; nnodes=nnodes, axis=:x)
    op, _ = build_steady_operator(img; nnodes=nnodes, axis=:x)

    # The kernel body is @inbounds, so a short vector is a memory-safety
    # question, not an accuracy one. The assembled path is the contract.
    for (leny, lenx) in ((nnodes - 32, nnodes), (nnodes, nnodes - 64), (nnodes + 8, nnodes))
        @test_throws DimensionMismatch mul!(zeros(leny), op, zeros(lenx))
        @test_throws DimensionMismatch mul!(zeros(leny), A, zeros(lenx))
        @test_throws DimensionMismatch mul!(zeros(leny), op, zeros(lenx), 1.0, 0.0)
    end
    @test mul!(zeros(nnodes), op, zeros(nnodes)) == zeros(nnodes)
end

@testset "the pore ordinal type refuses to overflow instead of wrapping" begin
    # An image this large is far past any device on hand, so the rule is tested
    # through the predicate rather than by building one.
    @test Tortuosity._operator_index_type(true, 1000) === Int32
    @test Tortuosity._operator_index_type(false, 1000) === Int32
    @test Tortuosity._operator_index_type(true, typemax(Int32) - 1) === Int32
    @test Tortuosity._operator_index_type(false, typemax(Int32) - 1) === Int32

    # On the host the ordinal simply widens; on device there is no 64-bit path,
    # and `cumsum!` into an Int32 buffer wraps to typemin rather than saturating,
    # so silently continuing would return a partly unwritten solution.
    @test Tortuosity._operator_index_type(false, typemax(Int32)) === Int
    @test Tortuosity._operator_index_type(false, Int64(typemax(Int32)) + 10) === Int
    @test_throws ArgumentError Tortuosity._operator_index_type(true, typemax(Int32))
    @test_throws ArgumentError Tortuosity._operator_index_type(true, Int64(typemax(Int32)) + 10)
    @test Int32(typemax(Int32)) + Int32(1) == typemin(Int32)
end

# The device half of the same rule, as it would be written by someone with the
# card for it. Never run: `@test_skip` records it Broken without evaluating it.
#
# The predicate above is called with a pore count; this is the thing that proves
# `build_steady_operator` actually consults it before allocating `idx`. Reaching
# it needs more than `typemax(Int32)` pore voxels — over 2.1 billion, about
# 1625³ at half porosity. The mask alone is 4.3 GB on the host and the operator's
# index array another 8.6 GB on the device, so the refusal has to be raised on a
# card in the 80 GB class to be observed at all; this machine has 48 GB.
#
# Nothing smaller substitutes. The failure it guards is silent by construction:
# `cumsum!` into an `Int32` buffer wraps to `typemin` rather than saturating, so
# `idx` goes negative, the kernel's `c0 > 0` test drops most voxels, and `y`
# comes back partly unwritten with no error raised anywhere.
function _device_operator_refuses_to_wrap()
    # `_gpu_adapt[]` rather than a backend package by name: this is the host-side
    # file, and the rule is the same on every backend. A dense `Bool` array, not
    # a `BitArray` — that is what `Imaginator.blobs` produces and what the device
    # kernels take, and it is the 4.3 GB the note above budgets for.
    img_dev = Tortuosity._gpu_adapt[](ones(Bool, 1625, 1625, 1625))
    nnodes = count(img_dev)
    nnodes > typemax(Int32) || error("fixture does not reach the bound")
    try
        build_steady_operator(img_dev; nnodes=nnodes, axis=:x, T=Float32)
        return false                     # wrapped silently instead of refusing
    catch err
        return err isa ArgumentError
    end
end

@testset "the device operator refuses an image past the 32-bit ordinal, end to end" begin
    # Needs an 80 GB-class card. Skipped unconditionally — see above.
    @test_skip _device_operator_refuses_to_wrap()
end

@testset "the operator releases only the arrays it owns" begin
    img = trues(8, 6, 6)
    nnodes = count(img)
    D = fill(2.0, size(img))

    # Built directly, `D` is the caller's and stays theirs.
    op, _ = build_steady_operator(img; nnodes=nnodes, axis=:x, D=D)
    @test op.owns_D === false
    @test Tortuosity._free!(op) === nothing

    owning, _ = build_steady_operator(img; nnodes=nnodes, axis=:x, D=copy(D), owns_D=true)
    @test owning.owns_D === true
    @test Tortuosity._free!(owning) === nothing

    # Through the constructor on CPU there is no copy to own.
    sim = SteadyDiffusionProblem(img; axis=:x, D=D .* img, matrixfree=true)
    @test sim.prob.A.owns_D === false
end

@testset "the package solve entry point" begin
    img = trues(24, 14, 14)
    img[9:12, 4:7, 4:7] .= false
    sim = SteadyDiffusionProblem(img; axis=:x)
    simf = SteadyDiffusionProblem(img; axis=:x, matrixfree=true)
    reference = tortuosity(
        reconstruct_field(solve(sim.prob, KrylovJL_CG(); reltol=1e-12).u, img), img; axis=:x
    )

    @testset "solves both paths to the same answer" begin
        for s in (sim, simf)
            sol = solve(s)
            @test Symbol(sol.retcode) === :Success
            tau = tortuosity(reconstruct_field(sol.u, img), img; axis=:x)
            @test isapprox(tau, reference; rtol=1e-8)
        end
    end

    @testset "leaves solve(sim.prob, alg) untouched" begin
        # The unopinionated form takes no preconditioner and the tolerance it is
        # handed, which is what every existing caller depends on.
        direct = solve(sim.prob, KrylovJL_CG(); reltol=1e-12)
        again = solve(sim.prob, KrylovJL_CG(); reltol=1e-12)
        @test direct.u == again.u
    end

    @testset "picks the tolerance from the element type" begin
        @test Tortuosity._default_reltol(Float64) == 1e-10
        @test Tortuosity._default_reltol(Float32) === 1.0f-6
    end

    @testset "resolves the preconditioner" begin
        # This image is far below the size where a coarse solve pays for itself.
        @test Tortuosity._resolve_precond(:auto, sim, false) === nothing
        @test Tortuosity._resolve_precond(:none, sim, false) === nothing
        @test count(img) < Tortuosity._PRECOND_MIN_NODES
        sentinel = Tortuosity.two_level_preconditioner(sim; block=4)
        @test Tortuosity._resolve_precond(sentinel, sim, false) === sentinel
    end

    @testset "drives the preconditioner on a problem large enough to want one" begin
        # Above _PRECOND_MIN_NODES the entry point reaches for the two-level
        # coarse space, which is the one path where the matrix-free operator has
        # to satisfy a consumer other than the Krylov solver.
        big = Imaginator.blobs(; shape=(64, 64, 64), porosity=0.65, blobiness=1, seed=42)
        @test count(big) > Tortuosity._PRECOND_MIN_NODES
        # Pinned to CPU: this file is the host-side suite, and the image is over
        # the size where construction would otherwise auto-detect a GPU.
        simb = SteadyDiffusionProblem(big; axis=:x, gpu=false, matrixfree=true,
                                      warn_nonpercolating=false)
        sima = SteadyDiffusionProblem(big; axis=:x, gpu=false, warn_nonpercolating=false)
        @test !isnothing(Tortuosity._resolve_precond(:auto, simb, false))

        solb = solve(simb; reltol=1e-10)
        sola = solve(sima; reltol=1e-10)
        @test Symbol(solb.retcode) === :Success
        @test Symbol(sola.retcode) === :Success
        # The two applies agree to rounding, not to the bit, so a residual
        # sitting near the tolerance can cross it one iteration apart. Equality
        # holds at every size measured, but asserting it would make this test a
        # tripwire for floating-point reassociation rather than for behaviour.
        @test abs(solb.iters - sola.iters) <= 1
        taub = tortuosity(reconstruct_field(solb.u, big), big; axis=:x)
        taua = tortuosity(reconstruct_field(sola.u, big), big; axis=:x)
        @test isapprox(taub, taua; rtol=1e-8)

        # The preconditioner has to be doing something, or the comparison above
        # would pass just as well with an inert one.
        plain = solve(simb; precond=:none, reltol=1e-10)
        @test plain.iters > solb.iters
    end

    @testset "forwards the tolerance and the algorithm" begin
        loose = solve(simf, KrylovJL_CG(); reltol=1e-4, precond=:none)
        tight = solve(simf, KrylovJL_CG(); reltol=1e-12, precond=:none)
        @test Symbol(loose.retcode) === :Success
        @test Symbol(tight.retcode) === :Success
        tau_tight = tortuosity(reconstruct_field(tight.u, img), img; axis=:x)
        @test isapprox(tau_tight, reference; rtol=1e-8)
        @test loose.u != tight.u
    end
end
