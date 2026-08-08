# Unit tests for the indexing, geometry, and array-plumbing helpers that every
# solver path is built on.
#
# These are the routines most likely to be rewritten for speed (the pore-index
# lookup tables, the boundary walk, the threaded set-intersection). Each test
# below pins a *property* the callers depend on — ordering, exact round-trip,
# element type — rather than the current implementation's internals, so an
# optimized rewrite either preserves the contract or fails here loudly.

using Test
using Random
using Tortuosity
using Tortuosity:
    atleast_3d,
    axis_dim,
    axis_faces,
    orthogonal_dims,
    build_pore_index,
    build_reverse_lookup,
    exclusive_scan!,
    find_boundary_nodes,
    find_chunk_bounds,
    find_true_indices,
    isin_slow,
    multihotvec,
    overlap_indices,
    overlap_indices_fast,
    overlap_indices_slow,
    reconstruct_field,
    reconstruct_slice,
    slice_indices

# A small mask with solid voxels on every face and an interior void, so slices
# along all three axes are partially occupied and no path is trivially uniform.
function swiss_cheese(dims=(6, 7, 5); seed=20240817)
    rng = MersenneTwister(seed)
    img = rand(rng, Bool, dims...)
    img[2, 3, 2] = false
    img[4, 4, 3] = true
    return BitArray(img)
end

# --- Axis symbol ↔ dimension mapping ---

@testset "axis helpers" begin
    @test axis_dim.((:x, :y, :z)) == (1, 2, 3)
    @test axis_faces(:x) == (:left, :right)
    @test axis_faces(:y) == (:front, :back)
    @test axis_faces(:z) == (:bottom, :top)

    @testset "orthogonal_dims excludes the transport dim" begin
        for (ax, d) in zip((:x, :y, :z), (1, 2, 3))
            od = orthogonal_dims(ax)
            @test length(od) == 2
            @test d ∉ od
            @test sort(collect((d, od...))) == [1, 2, 3]
            # The Int overload must agree with the Symbol one
            @test orthogonal_dims(d) == od
        end
    end

    @testset "invalid axis errors rather than silently defaulting" begin
        @test_throws ErrorException axis_dim(:w)
        @test_throws ErrorException axis_faces(:w)
        @test_throws ErrorException orthogonal_dims(:w)
    end
end

# --- Shape promotion ---

@testset "atleast_3d" begin
    @test size(atleast_3d(collect(1:4))) == (4, 1, 1)
    @test size(atleast_3d(rand(3, 5))) == (3, 5, 1)

    @testset "already-3D input is returned untouched (no copy)" begin
        # Callers rely on this: SteadyDiffusionProblem/TransientDiffusionProblem
        # call atleast_3d on every input, and copying a large image there would
        # double peak memory for no reason.
        a = rand(2, 3, 4)
        @test atleast_3d(a) === a
    end

    @testset "values survive the reshape in column-major order" begin
        m = [1 3; 2 4]
        @test vec(atleast_3d(m)) == vec(m)
    end
end

# --- Pore-index lookup tables ---

@testset "build_pore_index" begin
    img = swiss_cheese()
    pidx = build_pore_index(img)

    @test size(pidx) == size(img)
    @test eltype(pidx) === Int
    # Pore voxels are numbered 1..n in column-major order; solids are the 0 sentinel.
    @test pidx[img] == 1:count(img)
    @test all(iszero, pidx[.!img])

    @testset "agrees with find_true_indices / build_reverse_lookup" begin
        lin = find_true_indices(img)
        @test length(lin) == count(img)
        @test issorted(lin)
        # The three helpers are independent routes to the same numbering.
        for (ordinal, linear_idx) in enumerate(lin)
            @test pidx[linear_idx] == ordinal
        end
        lookup = build_reverse_lookup(img)
        @test length(lookup) == count(img)
        @test all(lookup[linear_idx] == pidx[linear_idx] for linear_idx in lin)
    end
end

@testset "slice_indices" begin
    img = swiss_cheese()
    pidx = build_pore_index(img)

    @testset "$(ax)-axis slices tile the pore numbering exactly once" for ax in (:x, :y, :z)
        d = axis_dim(ax)
        collected = Int[]
        for k in 1:size(img, d)
            inds = slice_indices(pidx, ax, k)
            # Every returned index is a real pore ordinal on that slice
            @test length(inds) == count(selectdim(img, d, k))
            @test all(1 .<= inds .<= count(img))
            # Ascending order is contractual: build_rhs and
            # StopAtPeriodicState scatter into device vectors with these.
            @test issorted(inds)
            append!(collected, inds)
        end
        # Stacking all slices along an axis must reproduce the whole pore set
        @test sort(collected) == collect(1:count(img))
    end
end

@testset "reconstruct_field / reconstruct_slice" begin
    img = swiss_cheese()
    n = count(img)
    u = collect(1.0:n)

    @testset "round-trips the pore vector and NaN-fills solids" begin
        c = reconstruct_field(u, img)
        @test size(c) == size(img)
        @test c[img] == u
        @test all(isnan, c[.!img])
    end

    @testset "preserves element type" begin
        for T in (Float32, Float64)
            c = reconstruct_field(T.(u), img)
            @test eltype(c) === T
        end
    end

    @testset "accepts a dense Bool mask as well as a BitArray" begin
        dense = Array(img)
        @test isequal(reconstruct_field(u, dense), reconstruct_field(u, img))
    end

    @testset "reconstruct_slice matches a slice of reconstruct_field" begin
        # Two independent reconstruction routes; the slice version exists purely
        # to avoid materialising the full grid, so it must agree exactly.
        pidx = build_pore_index(img)
        c_full = reconstruct_field(u, img)
        for ax in (:x, :y, :z)
            d = axis_dim(ax)
            for k in 1:size(img, d)
                @test isequal(
                    reconstruct_slice(u, pidx, ax, k),
                    Array(selectdim(c_full, d, k)),
                )
            end
        end
    end
end

# --- Boundary node detection ---

@testset "find_boundary_nodes" begin
    img = swiss_cheese()
    pidx = build_pore_index(img)

    @testset "matches the pore ordinals on each face" begin
        face_names = (:left, :right, :front, :back, :bottom, :top)
        slices = (
            (1, 1), (1, size(img, 1)),
            (2, 1), (2, size(img, 2)),
            (3, 1), (3, size(img, 3)),
        )
        for (face, (d, k)) in zip(face_names, slices)
            nodes = find_boundary_nodes(img, face)
            expected = filter(!iszero, vec(selectdim(pidx, d, k)))
            @test nodes == expected
            # Ascending order matters: apply_dirichlet_bc_fast! pairs these
            # positionally with a matching vals vector.
            @test issorted(nodes)
        end
    end

    @testset "a fully open cube puts a whole face on each boundary" begin
        open_cube = trues(4, 5, 6)
        @test length(find_boundary_nodes(open_cube, :left)) == 5 * 6
        @test length(find_boundary_nodes(open_cube, :front)) == 4 * 6
        @test length(find_boundary_nodes(open_cube, :bottom)) == 4 * 5
    end

    @testset "a single-voxel-thick axis maps both faces to the same nodes" begin
        thin = trues(1, 4, 4)
        @test find_boundary_nodes(thin, :left) == find_boundary_nodes(thin, :right)
    end

    @testset "an all-solid face yields no nodes" begin
        img2 = trues(4, 4, 4)
        img2[1, :, :] .= false
        @test isempty(find_boundary_nodes(img2, :left))
        @test length(find_boundary_nodes(img2, :right)) == 16
    end

    @test_throws ErrorException find_boundary_nodes(trues(2, 2, 2), :sideways)
end

# --- Set-intersection helpers (threaded; used inside Dirichlet BC application) ---

@testset "overlap_indices family" begin
    @testset "reference, serial, and threaded implementations agree" begin
        rng = MersenneTwister(7)
        cases = [
            (collect(1:10), [3, 4, 1]),
            (rand(rng, 1:20, 500), collect(5:15)),
            (rand(rng, 1:5, 300), [1, 1, 2]),        # duplicates in b
            (fill(3, 64), [3]),                       # every element matches
            (collect(1:50), Int[]),                   # nothing matches
            (rand(rng, 1:1000, 5000), rand(rng, 1:1000, 200)),
        ]
        for (a, b) in cases
            expected = overlap_indices_slow(a, b)
            @test overlap_indices(a, b) == expected
            @test overlap_indices_fast(a, b) == expected
            # Documented contract: ascending index order, not order-of-appearance.
            @test issorted(expected)
        end
    end

    @testset "Set and Array arguments are interchangeable" begin
        a = [4, 8, 15, 16, 23, 42]
        b = [15, 42, 99]
        @test overlap_indices(a, b) == overlap_indices(a, Set(b))
        @test overlap_indices_fast(a, b) == overlap_indices_fast(a, Set(b))
    end

    @testset "isin_slow flags membership element-wise" begin
        @test isin_slow([1, 2, 3, 4], [2, 4]) == [false, true, false, true]
        @test isin_slow(Int[], [1]) == Bool[]
    end
end

@testset "find_chunk_bounds" begin
    @test find_chunk_bounds(; nelems=10, ndivs=3) == [(1, 4), (5, 8), (9, 10)]

    @testset "an empty input yields no chunks rather than an invalid range" begin
        # `ceil(0/ndivs) == 0` makes the naive construction build `1:0:0`, which
        # throws. `apply_dirichlet_bc_fast!` reaches this whenever the matrix has
        # no stored entries — e.g. a pore space with no face-connected pairs.
        for ndivs in (1, 2, 8)
            @test isempty(find_chunk_bounds(; nelems=0, ndivs=ndivs))
        end
        @test overlap_indices_fast(Int[], [1, 2]) == Int[]
        @test overlap_indices_fast(Int[], [1, 2]) isa Vector{Int}
        @test overlap_indices_fast(Int[], [1, 2]) == overlap_indices_slow(Int[], [1, 2])
    end

    @testset "chunks tile 1:nelems exactly once, and never exceed ndivs" begin
        for nelems in (1, 2, 3, 7, 10, 64, 1000), ndivs in (1, 2, 3, 4, 8, 16)
            chunks = find_chunk_bounds(; nelems=nelems, ndivs=ndivs)
            @test length(chunks) <= ndivs
            @test first(first(chunks)) == 1
            @test last(last(chunks)) == nelems
            # Contiguous and non-overlapping
            @test all(chunks[i][2] + 1 == chunks[i + 1][1] for i in 1:(length(chunks) - 1))
            @test all(lo <= hi for (lo, hi) in chunks)
        end
    end
end

# --- Sparse-vector construction ---

@testset "multihotvec" begin
    @test multihotvec([1, 3, 4], 6) == [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    @test multihotvec([1, 3, 4], 6; vals=[0.1, 0.3, 0.2]) == [0.1, 0.0, 0.3, 0.2, 0.0, 0.0]

    @testset "element type follows vals" begin
        @test eltype(multihotvec([1, 2], 3; vals=Float32[1, 2])) === Float32
        @test eltype(multihotvec([1, 2], 3; vals=1.0f0)) === Float32
        @test eltype(multihotvec([1, 2], 3)) === Float64
    end

    @testset "template controls the output container" begin
        # On CPU every container is a `Vector`, so this cannot prove the output
        # was allocated *from the template* — that only shows up on GPU, where
        # ignoring `template` costs an implicit host→device copy. What it can
        # pin is that the element type comes from `vals` and not from the
        # template, i.e. that `similar(template, eltype(vals), n)` is the call
        # being made rather than `similar(template)`.
        template = zeros(Float64, 4)
        v = multihotvec([2], 4; vals=Float32[9], template=template)
        @test eltype(v) === Float32
        @test v == Float32[0, 9, 0, 0]
        @test length(v) == 4
    end

    @testset "rejects inconsistent inputs" begin
        @test_throws AssertionError multihotvec([1, 2], 4; vals=[1.0])
        @test_throws AssertionError multihotvec([1, 5], 4)
    end
end

# --- Prefix scan (backs the GPU connectivity builder's write offsets) ---

@testset "exclusive_scan!" begin
    @testset "matches the definition out[i] = sum(inp[1:i-1])" begin
        fuzz = rand(MersenneTwister(4242), Int32(0):Int32(9), 257)
        for inp in (Int32[1], Int32[1, 2, 3, 4, 5], Int32[0, 0, 7, 0], fuzz)
            out = similar(inp)
            exclusive_scan!(out, inp)
            @test out == [sum(inp[1:(i - 1)]) for i in eachindex(inp)]
            # Last element plus the last input equals the total — this identity is
            # exactly how _build_connectivity_list_ka derives total_conns.
            @test Int(out[end]) + Int(inp[end]) == Int(sum(inp))
        end
    end

    @testset "empty input is a no-op" begin
        out = Int32[]
        @test exclusive_scan!(out, Int32[]) === out
    end

    @testset "length mismatch throws" begin
        @test_throws DimensionMismatch exclusive_scan!(zeros(Int32, 3), Int32[1, 2])
    end
end
