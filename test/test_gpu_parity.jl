# Fuzz tests comparing the KA-based GPU path against the original CUDA baseline
# (preserved in bench/old_baseline.jl). Verifies that high-level operations
# produce mathematically equivalent results regardless of which atomic write
# order the kernels happen to use.
#
# Caller (runtests.jl) is responsible for ensuring CUDA is loaded and functional
# before including this file.
using Test
using Random
using LinearAlgebra
using SparseArrays
using CUDA
using Tortuosity
using Tortuosity: PortableSparseCSC, find_boundary_nodes, Imaginator,
    build_connectivity_list, build_adjacency_matrix, laplacian,
    apply_dirichlet_bc_fast!

include(joinpath(@__DIR__, "..", "bench", "old_baseline.jl"))
using .OldBaseline

# ---------------------------------------------------------------------------
# CSC + connectivity comparison helpers
# ---------------------------------------------------------------------------

# Field accessors covering both CuSparseMatrixCSC (old) and PortableSparseCSC (new)
_csc_colptr(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.colPtr
_csc_rowval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.rowVal
_csc_nzval(A::CUDA.CUSPARSE.CuSparseMatrixCSC) = A.nzVal
_csc_colptr(A::PortableSparseCSC) = A.colptr
_csc_rowval(A::PortableSparseCSC) = A.rowval
_csc_nzval(A::PortableSparseCSC) = A.nzval

"""Sort each column's `(rowval, nzval)` pair by row index, returning host arrays."""
function csc_canonical(A)
    n = size(A, 2)
    colptr = Array{Int}(_csc_colptr(A))
    rowval = Array{Int}(_csc_rowval(A))
    nzval = Array(_csc_nzval(A))
    new_rowval = similar(rowval)
    new_nzval = similar(nzval)
    for j in 1:n
        cs = colptr[j]
        ce = colptr[j + 1] - 1
        cs > ce && continue
        sub_rv = view(rowval, cs:ce)
        sub_nv = view(nzval, cs:ce)
        perm = sortperm(sub_rv)
        new_rowval[cs:ce] = sub_rv[perm]
        new_nzval[cs:ce] = sub_nv[perm]
    end
    return (colptr, new_rowval, new_nzval)
end

"""Compare two CSC matrices regardless of internal column ordering."""
function csc_equivalent(A_old, A_new; rtol=1e-5)
    size(A_old) == size(A_new) || return false
    cpo, rvo, nvo = csc_canonical(A_old)
    cpn, rvn, nvn = csc_canonical(A_new)
    cpo == cpn || return false
    rvo == rvn || return false
    return isapprox(nvo, nvn; rtol=rtol)
end

"""Convert connectivity list to a sorted vector of `(i,j)` tuples."""
function conns_canonical(conns)
    arr = Array(conns)
    return sort([(Int(arr[k, 1]), Int(arr[k, 2])) for k in 1:size(arr, 1)])
end

conns_equivalent(c_old, c_new) = conns_canonical(c_old) == conns_canonical(c_new)

# ---------------------------------------------------------------------------
# Test image fixtures: edge cases + fuzz
# ---------------------------------------------------------------------------

"""Generate a varied collection of test images: edge cases + ~20 random blobs."""
function fuzz_images()
    imgs = Tuple{String,Array{Bool,3}}[]

    # ---- Edge cases ----
    for n in (16, 24, 32)
        push!(imgs, ("open $(n)^3", ones(Bool, n, n, n)))
    end
    # Half-cube (one half is solid)
    for n in (16, 24)
        img = ones(Bool, n, n, n)
        img[:, :, 1:(n ÷ 2)] .= false
        push!(imgs, ("half-cube $(n)^3 (z<n/2 solid)", img))
    end
    # Two-voxel pair (most degenerate connected case)
    let img = falses(8, 8, 8)
        img[4, 4, 4] = true
        img[4, 4, 5] = true
        push!(imgs, ("two-voxel pair", Array(img)))
    end
    # Single straight channel
    let img = falses(16, 16, 16)
        img[8, 8, :] .= true
        push!(imgs, ("single z-channel 16^3", Array(img)))
    end

    # ---- Fuzz: blobs at various sizes / porosities / seeds ----
    for n in (16, 24, 32),
        ε in (0.4f0, 0.55f0, 0.7f0),
        seed in (1, 7, 42, 100)
        img = Bool.(Imaginator.blobs(; shape=(n, n, n), porosity=ε, blobiness=1, seed=seed))
        count(img) == 0 && continue
        push!(imgs, ("blob $(n)^3 ε=$(ε) seed=$(seed)", img))
    end

    return imgs
end

const TEST_IMAGES = fuzz_images()
@info "GPU parity: $(length(TEST_IMAGES)) test images"

# Helpful predicate: skip degenerate cases that can't form a usable problem
nonempty(img) = sum(img) >= 4

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "build_connectivity_list" begin
    for (label, img) in TEST_IMAGES
        nonempty(img) || continue
        cu_img = CuArray(img)
        old_conns = OldBaseline.create_connectivity_list_old(cu_img)
        new_conns = build_connectivity_list(cu_img)
        @test conns_equivalent(old_conns, new_conns)
    end
end

@testset "build_adjacency_matrix (uniform weights)" begin
    for (label, img) in TEST_IMAGES
        nonempty(img) || continue
        cu_img = CuArray(img)
        conns = OldBaseline.create_connectivity_list_old(cu_img)
        size(conns, 1) == 0 && continue
        nnodes = Int(sum(cu_img))
        weights = CUDA.fill(1.0f0, size(conns, 1))
        am_old = OldBaseline.create_adjacency_matrix_old(conns; n=nnodes, weights=weights)
        am_new = build_adjacency_matrix(conns; n=nnodes, weights=weights)
        @test csc_equivalent(am_old, am_new)
    end
end

@testset "build_adjacency_matrix (random weights)" begin
    for (label, img) in TEST_IMAGES
        nonempty(img) || continue
        cu_img = CuArray(img)
        conns = OldBaseline.create_connectivity_list_old(cu_img)
        size(conns, 1) == 0 && continue
        nnodes = Int(sum(cu_img))
        Random.seed!(hash(label))
        w_cpu = rand(Float32, size(conns, 1)) .+ 0.5f0
        weights = CuArray(w_cpu)
        am_old = OldBaseline.create_adjacency_matrix_old(conns; n=nnodes, weights=weights)
        am_new = build_adjacency_matrix(conns; n=nnodes, weights=weights)
        @test csc_equivalent(am_old, am_new)
    end
end

@testset "laplacian" begin
    for (label, img) in TEST_IMAGES
        nonempty(img) || continue
        cu_img = CuArray(img)
        conns = OldBaseline.create_connectivity_list_old(cu_img)
        size(conns, 1) == 0 && continue
        nnodes = Int(sum(cu_img))
        weights = CUDA.fill(1.0f0, size(conns, 1))
        am_old = OldBaseline.create_adjacency_matrix_old(conns; n=nnodes, weights=weights)
        am_new = build_adjacency_matrix(conns; n=nnodes, weights=weights)
        L_old = OldBaseline.laplacian_old(am_old)
        L_new = laplacian(am_new)
        @test csc_equivalent(L_old, L_new; rtol=1e-5)
    end
end

@testset "SpMV (mul!) on Laplacian" begin
    for (label, img) in TEST_IMAGES
        nonempty(img) || continue
        cu_img = CuArray(img)
        conns = OldBaseline.create_connectivity_list_old(cu_img)
        size(conns, 1) == 0 && continue
        nnodes = Int(sum(cu_img))
        weights = CUDA.fill(1.0f0, size(conns, 1))
        am_old = OldBaseline.create_adjacency_matrix_old(conns; n=nnodes, weights=weights)
        am_new = build_adjacency_matrix(conns; n=nnodes, weights=weights)
        L_old = OldBaseline.laplacian_old(am_old)
        L_new = laplacian(am_new)
        Random.seed!(hash(label))
        x = CuArray(rand(Float32, nnodes))
        y_old = similar(x)
        y_new = similar(x)
        mul!(y_old, L_old, x)
        mul!(y_new, L_new, x)
        @test isapprox(Array(y_old), Array(y_new); rtol=1e-4)
    end
end

@testset "apply_dirichlet_bc_fast!" begin
    for (label, img) in TEST_IMAGES
        nonempty(img) || continue
        # Need at least one pore voxel on each face along x for the BC to apply meaningfully
        any(img[1, :, :]) && any(img[end, :, :]) || continue

        cu_img = CuArray(img)
        nnodes = Int(sum(cu_img))
        conns = OldBaseline.create_connectivity_list_old(cu_img)
        size(conns, 1) == 0 && continue
        weights = CUDA.fill(1.0f0, size(conns, 1))

        # Build via OLD path
        am_old = OldBaseline.create_adjacency_matrix_old(conns; n=nnodes, weights=weights)
        L_old = OldBaseline.laplacian_old(am_old)
        b_old = CUDA.zeros(Float32, nnodes)

        # Build via NEW path
        am_new = build_adjacency_matrix(conns; n=nnodes, weights=weights)
        L_new = laplacian(am_new)
        b_new = CUDA.zeros(Float32, nnodes)

        # Boundary nodes (computed CPU-side from the same image)
        left_nodes = find_boundary_nodes(img, :left)
        right_nodes = find_boundary_nodes(img, :right)
        bc_nodes = vcat(left_nodes, right_nodes)
        bc_nodes_i32 = Int32.(bc_nodes)
        bc_vals_cpu = vcat(fill(1.0f0, length(left_nodes)), fill(0.0f0, length(right_nodes)))
        bc_vals_gpu = CuArray(bc_vals_cpu)

        OldBaseline.apply_dirichlet_bc_old!(L_old, b_old; nodes=bc_nodes_i32, vals=bc_vals_gpu)
        apply_dirichlet_bc_fast!(L_new, b_new; nodes=bc_nodes_i32, vals=bc_vals_gpu)

        @test csc_equivalent(L_old, L_new; rtol=1e-4)
        @test isapprox(Array(b_old), Array(b_new); rtol=1e-4)
    end
end

@testset "SteadyDiffusionProblem end-to-end (assembled A, b)" begin
    for (label, img) in TEST_IMAGES
        nonempty(img) || continue
        any(img[1, :, :]) && any(img[end, :, :]) || continue

        cu_img = CuArray(img)
        # Old pipeline returns (A_old, b_old) directly
        local A_old, b_old
        try
            A_old, b_old = OldBaseline.tortuosity_simulation_old(cu_img)
        catch e
            @warn "Old baseline pipeline failed for $label" exception=e
            continue
        end

        # New pipeline returns a SteadyDiffusionProblem
        ts_new = SteadyDiffusionProblem(cu_img; axis=:x, gpu=true)

        @test csc_equivalent(A_old, ts_new.prob.A; rtol=1e-4)
        @test isapprox(Array(b_old), Array(ts_new.prob.b); rtol=1e-4)
    end
end

# ---------------------------------------------------------------------------
# Row-index ordering
# ---------------------------------------------------------------------------
#
# `build_adjacency_matrix` scatters COO entries into CSC slots with atomic
# counters, so row indices within a column come out in whatever order the
# threads won — measured at ~90% of columns unsorted on a 24³ open image. That
# ordering survives `laplacian` and `dropzeros!` all the way to the matrix
# `TortuosityCUDAExt` hands to CUSPARSE.
#
# This is fine today: CUDA.jl runs CSC SpMV as a transposed CSR operation, which
# scatters and is therefore order-independent. But "fine" here is a property of
# the vendor library, not a guarantee we enforce, and enforcing it would mean a
# segmented sort over every nonzero on the setup path. So the property is tested
# instead: if a future CUDA.jl or CUSPARSE starts requiring sorted indices, this
# fails loudly rather than quietly returning a wrong tortuosity.

@testset "CUSPARSE SpMV tolerates unsorted row indices within a column" begin
    # CPU mask with gpu=true: the constructor moves it across itself, so this
    # doesn't trip the "img was passed on GPU" warning.
    A = SteadyDiffusionProblem(ones(Bool, 24, 24, 24); axis=:x, gpu=true).prob.A

    colptr, rowval, nzval = Array(A.colptr), Array(A.rowval), Array(A.nzval)
    unsorted = count(1:A.n) do j
        !issorted(@view rowval[colptr[j]:(colptr[j + 1] - 1)])
    end
    # Guard the premise: if assembly ever starts emitting sorted columns, the
    # rest of this testset stops proving anything and should be revisited.
    @test unsorted > 0

    # Sort the reference's columns before handing them to `SparseMatrixCSC`.
    # That type's invariant is sorted row indices and its constructor does not
    # check — leaving them unsorted would make the oracle depend on the very
    # order-independence this testset is trying to verify.
    rv_ref, nv_ref = copy(rowval), copy(nzval)
    for j in 1:A.n
        slot = colptr[j]:(colptr[j + 1] - 1)
        perm = sortperm(@view rv_ref[slot])
        rv_ref[slot] = rv_ref[slot][perm]
        nv_ref[slot] = nv_ref[slot][perm]
    end
    A_cpu = SparseMatrixCSC(A.m, A.n, copy(colptr), rv_ref, nv_ref)
    x = rand(Float32, A.n)
    y_ref = A_cpu * x

    @testset "as assembled" begin
        @test isapprox(Array(A * CuArray(x)), y_ref; rtol=1e-4)
    end

    @testset "with every column deliberately shuffled" begin
        Random.seed!(20260806)
        rv, nv = copy(rowval), copy(nzval)
        for j in 1:A.n
            slot = colptr[j]:(colptr[j + 1] - 1)
            perm = shuffle(collect(slot))
            rv[slot] = rv[perm]
            nv[slot] = nv[perm]
        end
        A_shuffled = PortableSparseCSC(A.m, A.n, CuArray(colptr), CuArray(rv), CuArray(nv))
        @test isapprox(Array(A_shuffled * CuArray(x)), y_ref; rtol=1e-4)
    end

    @testset "sorting the columns does not change the result" begin
        rv, nv = copy(rowval), copy(nzval)
        for j in 1:A.n
            slot = colptr[j]:(colptr[j + 1] - 1)
            perm = sortperm(@view rv[slot])
            rv[slot] = rv[slot][perm]
            nv[slot] = nv[slot][perm]
        end
        A_sorted = PortableSparseCSC(A.m, A.n, CuArray(colptr), CuArray(rv), CuArray(nv))
        @test isapprox(Array(A_sorted * CuArray(x)), y_ref; rtol=1e-4)
    end
end
