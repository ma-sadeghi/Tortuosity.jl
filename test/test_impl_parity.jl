# Cross-implementation parity fuzz tests.
#
# Several operations in Tortuosity.jl have multiple implementations — typically
# a readable "reference" version and an optimized "fast" version, or a
# CPU-specialized version alongside a backend-agnostic KA version. This file
# verifies that every pair agrees on a suite of deterministic + fuzzed inputs,
# so that future refactors to one implementation can't silently drift from the
# other.
#
# Pairs covered (all tested on CPU, so no GPU required):
#
# 1. apply_dirichlet_bc!               vs  apply_dirichlet_bc_fast!   (SparseMatrixCSC)
# 2. apply_dirichlet_bc_fast!(CPU)     vs  apply_dirichlet_bc_fast!(PortableSparseCSC)
# 3. _build_connectivity_list_cpu     vs  _build_connectivity_list_ka (CPU backend)
# 4. laplacian(AbstractMatrix)         vs  laplacian(PortableSparseCSC)
# 5. zero_rows!(SparseMatrixCSC)       vs  zero_rows!(PortableSparseCSC)

using Test
using Random
using SparseArrays
using LinearAlgebra
using Tortuosity
using Tortuosity: Imaginator,
    PortableSparseCSC,
    apply_dirichlet_bc!,
    apply_dirichlet_bc_fast!,
    _build_connectivity_list_cpu,
    _build_connectivity_list_ka,
    build_adjacency_matrix,
    laplacian,
    zero_rows!,
    find_boundary_nodes,
    axis_faces

function sparse_to_portable(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    return PortableSparseCSC(
        size(A, 1), size(A, 2),
        copy(SparseArrays.getcolptr(A)),
        copy(rowvals(A)),
        copy(nonzeros(A)),
    )
end

function to_dense(A::PortableSparseCSC)
    m, n = size(A)
    B = zeros(eltype(A.nzval), m, n)
    for j in 1:n
        for idx in A.colptr[j]:(A.colptr[j + 1] - 1)
            # Accumulate, matching how `mul!` sums duplicate (row, col) entries.
            # Assigning would render a duplicate-emitting kernel as correct.
            B[A.rowval[idx], j] += A.nzval[idx]
        end
    end
    return B
end

canonicalize_conns(c) = sort([(Int(c[k, 1]), Int(c[k, 2])) for k in 1:size(c, 1)])

function parity_fixtures()
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
    return imgs
end

const PARITY_IMAGES = parity_fixtures()

function build_laplacian_cpu(img::AbstractArray{Bool,3})
    conns = _build_connectivity_list_cpu(img)
    nnodes = count(img)
    (nnodes == 0 || size(conns, 1) == 0) && return nothing
    w = ones(Float64, size(conns, 1))
    am = build_adjacency_matrix(conns; n=nnodes, weights=w)
    return laplacian(am)
end

# Build the Laplacian the way the GPU pipeline does — natively as a
# PortableSparseCSC — rather than wrapping the CPU result. The two are not
# structurally identical: `spdiagm(d) - A` prunes a diagonal that evaluates to
# zero, while the KA kernel emits one in every column. Wrapping the CPU matrix
# would hand the portable BC routine an input its own assembly never produces,
# so the comparison would not reflect what actually runs on GPU.
function build_laplacian_portable(img::AbstractArray{Bool,3})
    conns = _build_connectivity_list_cpu(img)
    nnodes = count(img)
    (nnodes == 0 || size(conns, 1) == 0) && return nothing
    w = ones(Float64, size(conns, 1))
    am = sparse_to_portable(build_adjacency_matrix(conns; n=nnodes, weights=w))
    return laplacian(am)
end

function bc_pair(img)
    inlet_face, outlet_face = axis_faces(:x)
    inlet_nodes = find_boundary_nodes(img, inlet_face)
    outlet_nodes = find_boundary_nodes(img, outlet_face)
    bc_nodes = vcat(inlet_nodes, outlet_nodes)
    bc_vals = vcat(fill(1.0, length(inlet_nodes)), fill(0.0, length(outlet_nodes)))
    return bc_nodes, bc_vals
end

@testset "apply_dirichlet_bc! (ref) vs apply_dirichlet_bc_fast! (SparseMatrixCSC)" begin
    for (label, img) in PARITY_IMAGES
        any(img[1, :, :]) && any(img[end, :, :]) || continue
        L = build_laplacian_cpu(img)
        L === nothing && continue
        nnodes = size(L, 1)
        bc_nodes, bc_vals = bc_pair(img)

        A_ref = copy(L)
        b_ref = zeros(Float64, nnodes)
        apply_dirichlet_bc!(A_ref, b_ref; nodes=bc_nodes, vals=bc_vals)

        A_fast = copy(L)
        b_fast = zeros(Float64, nnodes)
        apply_dirichlet_bc_fast!(A_fast, b_fast; nodes=bc_nodes, vals=bc_vals)

        @test Array(A_ref) ≈ Array(A_fast)
        @test b_ref ≈ b_fast
    end
end

@testset "apply_dirichlet_bc_fast!(::SparseMatrixCSC) vs (::PortableSparseCSC)" begin
    for (label, img) in PARITY_IMAGES
        any(img[1, :, :]) && any(img[end, :, :]) || continue
        L = build_laplacian_cpu(img)
        L === nothing && continue
        nnodes = size(L, 1)
        bc_nodes, bc_vals = bc_pair(img)

        A_sparse = copy(L)
        b_sparse = zeros(Float64, nnodes)
        apply_dirichlet_bc_fast!(A_sparse, b_sparse; nodes=bc_nodes, vals=bc_vals)

        A_port = build_laplacian_portable(img)
        b_port = zeros(Float64, nnodes)
        apply_dirichlet_bc_fast!(A_port, b_port; nodes=bc_nodes, vals=bc_vals)

        @test to_dense(A_port) ≈ Array(A_sparse)
        @test b_port ≈ b_sparse
    end
end

@testset "_build_connectivity_list_cpu vs _ka (CPU backend)" begin
    for (label, img) in PARITY_IMAGES
        count(img) >= 2 || continue
        conns_cpu = _build_connectivity_list_cpu(img)
        conns_ka = _build_connectivity_list_ka(img)
        @test canonicalize_conns(conns_cpu) == canonicalize_conns(conns_ka)
    end
end

@testset "laplacian(::SparseMatrixCSC) vs laplacian(::PortableSparseCSC)" begin
    for (label, img) in PARITY_IMAGES
        count(img) >= 2 || continue
        conns = _build_connectivity_list_cpu(img)
        size(conns, 1) > 0 || continue
        nnodes = count(img)
        w = ones(Float64, size(conns, 1))
        am_sparse = build_adjacency_matrix(conns; n=nnodes, weights=w)
        L_sparse = laplacian(am_sparse)
        am_port = sparse_to_portable(am_sparse)
        L_port = laplacian(am_port)
        @test to_dense(L_port) ≈ Array(L_sparse)
    end
end

@testset "zero_rows!(::SparseMatrixCSC) vs zero_rows!(::PortableSparseCSC)" begin
    for (label, img) in PARITY_IMAGES
        count(img) >= 2 || continue
        L = build_laplacian_cpu(img)
        L === nothing && continue
        inlet, _ = axis_faces(:x)
        bc_nodes = find_boundary_nodes(img, inlet)
        isempty(bc_nodes) && continue

        L_sparse = copy(L)
        zero_rows!(L_sparse, bc_nodes)
        L_port = sparse_to_portable(copy(L))
        zero_rows!(L_port, bc_nodes)
        @test to_dense(L_port) ≈ Array(L_sparse)
    end
end

# 6. build_adjacency_matrix(::Array{Int,2})  vs  the backend-agnostic method
#
# The `Array{Int,2}` method writes CSC arrays straight from the pre-sorted
# connectivity list; the generic method runs the KA histogram/scan/scatter
# pipeline. Only the dense matrices are compared, because the atomic scatter
# leaves row indices unsorted within each column.
@testset "build_adjacency_matrix (direct CSC) vs (KA scatter)" begin
    for (label, img) in PARITY_IMAGES
        conns = _build_connectivity_list_cpu(img)
        nedges = size(conns, 1)
        nedges > 0 || continue
        nnodes = count(img)

        for weights in (ones(Float64, nedges), collect(range(0.25, 4.0; length=nedges)))
            am_cpu = build_adjacency_matrix(conns; n=nnodes, weights=weights)
            # Int32 indices dispatch away from the Array{Int,2} specialisation
            # and onto the generic (KA) method, running on the CPU backend.
            am_ka = build_adjacency_matrix(Matrix{Int32}(conns); n=nnodes, weights=weights)
            @test to_dense(am_ka) ≈ Array(am_cpu)
            @test nnz(am_ka) == nnz(am_cpu)
        end
    end
end

# 7. Whole-pipeline parity: the CPU-assembled steady system versus one built
#    entirely from the backend-agnostic path (KA connectivity → KA adjacency →
#    KA laplacian → PortableSparseCSC Dirichlet elimination). This is the same
#    chain the GPU takes, exercised on the CPU backend so it runs everywhere.
@testset "steady-system assembly parity: CPU chain vs backend-agnostic chain" begin
    for (label, img) in PARITY_IMAGES
        any(img[1, :, :]) && any(img[end, :, :]) || continue
        nnodes = count(img)
        nnodes >= 4 || continue
        bc_nodes, bc_vals = bc_pair(img)

        conns_cpu = _build_connectivity_list_cpu(img)
        size(conns_cpu, 1) > 0 || continue
        A_cpu = laplacian(build_adjacency_matrix(
            conns_cpu; n=nnodes, weights=ones(Float64, size(conns_cpu, 1)),
        ))
        b_cpu = zeros(Float64, nnodes)
        apply_dirichlet_bc_fast!(A_cpu, b_cpu; nodes=bc_nodes, vals=bc_vals)

        conns_ka = _build_connectivity_list_ka(img)
        A_ka = laplacian(build_adjacency_matrix(
            conns_ka; n=nnodes, weights=ones(Float64, size(conns_ka, 1)),
        ))
        b_ka = zeros(Float64, nnodes)
        apply_dirichlet_bc_fast!(A_ka, b_ka; nodes=bc_nodes, vals=bc_vals)

        @test to_dense(A_ka) ≈ Array(A_cpu)
        @test b_ka ≈ b_cpu
    end
end
