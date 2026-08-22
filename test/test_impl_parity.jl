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
# 6. build_adjacency_matrix(direct)    vs  build_adjacency_matrix(KA scatter)
# 7. CPU reference chain               vs  backend-agnostic reference chain
# 8. build_steady_system               vs  the reference chain, bit for bit

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
    build_connectivity_list,
    build_adjacency_matrix,
    build_steady_system,
    interpolate_edge_values,
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

# Most testsets below skip a fixture that has no pore voxels on both `:x` faces,
# or fewer than four nodes, because the Dirichlet cases they compare would be
# empty. A fixture list that stopped satisfying either would empty those loops
# silently and look exactly like a passing run — which is the reason case 8
# counts what it checked. Assert the precondition once here instead of eight
# times below.
@testset "every parity fixture reaches both Dirichlet faces" begin
    for (label, img) in PARITY_IMAGES
        @testset "$(label)" begin
            @test any(img[1, :, :])
            @test any(img[end, :, :])
            @test count(img) >= 4
            @test size(_build_connectivity_list_cpu(img), 1) > 0
        end
    end
end

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

# 7. Reference-chain parity: the CPU-specialized connectivity → adjacency →
#    laplacian → Dirichlet-elimination chain against the backend-agnostic one
#    (KA connectivity → KA adjacency → KA laplacian → PortableSparseCSC
#    elimination), run on the CPU backend so it works everywhere.
#
#    Neither chain is the production assembler any more — `SteadyDiffusionProblem`
#    calls `build_steady_system`, which is covered by case 8 below. What this
#    pins is that the two reference chains still agree with each other, so case 8
#    is comparing against a reference that has not drifted.
@testset "reference-chain parity: CPU chain vs backend-agnostic chain" begin
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

# 8. The fused assembler against the reference chain, bit for bit.
#
# `build_steady_system` replaced `build_connectivity_list → interpolate_edge_values
# → build_adjacency_matrix → laplacian → apply_dirichlet_bc_fast!` with two kernel
# passes over the mask. What made that safe to do was not "close enough" but
# *identical output*: same colptr, same rowval, same nzval, same b, to the last
# bit. Every other assertion covering assembly is `≈` or an rtol, so nothing else
# would catch a reordered summation or a weight that moved by an ulp — and the
# golden τ values would absorb both silently. This is the guard for the single
# largest change in the campaign, so it is written as `==`.
#
# Bit-identity is a real property here, not a coincidence: the reference chain
# sums a node's degree over its neighbours in ascending pore ordinal, which is
# the order the kernel walks its six face offsets in, and the harmonic mean
# `2ab/(a+b)` is exactly symmetric in its two arguments, so it does not matter
# which endpoint the chain calls `a`.

# Six-face degree of every pore voxel, used to prove the fixtures actually
# contain the zero-degree nodes the elimination convention is delicate about.
function isolated_pore_voxels(img)
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

# A duct that percolates, plus three deliberately isolated voxels: one in the
# interior (its column is dropped entirely), one on the inlet face and one on the
# outlet face (both pinned with a unit diagonal instead of their zero degree).
function isolated_voxel_fixture()
    img = zeros(Bool, 8, 6, 6)
    img[:, 3:4, 3:4] .= true
    img[4, 1, 1] = true
    img[1, 6, 6] = true
    img[8, 1, 6] = true
    return img
end

# Smooth, strictly positive and different in every voxel, so no two harmonic
# means coincide by accident.
variable_diffusivity(img) =
    [0.5 + 0.1i + 0.02j + 0.003k for i in 1:size(img, 1), j in 1:size(img, 2), k in 1:size(img, 3)]

function reference_steady_system(img; axis, D=nothing)
    nnodes = count(img)
    conns = build_connectivity_list(img)
    weights = isnothing(D) ? ones(Float64, size(conns, 1)) :
              interpolate_edge_values(D[img], conns)
    A = laplacian(build_adjacency_matrix(conns; n=nnodes, weights=weights))
    b = zeros(Float64, nnodes)

    inlet_face, outlet_face = axis_faces(axis)
    inlet = find_boundary_nodes(img, inlet_face)
    outlet = find_boundary_nodes(img, outlet_face)
    nodes = vcat(inlet, outlet)
    vals = vcat(ones(length(inlet)), zeros(length(outlet)))
    apply_dirichlet_bc_fast!(A, b; nodes=nodes, vals=vals)
    return A, b
end

@testset "build_steady_system == the reference chain" begin
    fixtures = vcat(PARITY_IMAGES, [("isolated voxels 8x6x6", isolated_voxel_fixture())])

    iso = isolated_pore_voxels(isolated_voxel_fixture())
    @test any(c -> c[1] == 1, iso)                  # zero-degree inlet node
    @test any(c -> c[1] == 8, iso)                  # zero-degree outlet node
    @test any(c -> 1 < c[1] < 8, iso)               # zero-degree interior node
    # The blob fixtures are untrimmed, which is the case a trimmed image would
    # never reach. If that ever stops being true this testset quietly narrows.
    @test any(!isempty(isolated_pore_voxels(img)) for (_, img) in PARITY_IMAGES)

    checked = 0
    for (label, img) in fixtures
        nnodes = count(img)
        nnodes >= 4 || continue
        cases = (("uniform D", nothing), ("variable D", variable_diffusivity(img)))
        @testset "$(label) — $(dlabel)" for (dlabel, D) in cases
            A_ref, b_ref = reference_steady_system(img; axis=:x, D=D)
            A_new, b_new = build_steady_system(img; nnodes=nnodes, axis=:x, D=D)

            @test SparseArrays.getcolptr(A_new) == SparseArrays.getcolptr(A_ref)
            @test rowvals(A_new) == rowvals(A_ref)
            @test nonzeros(A_new) == nonzeros(A_ref)
            @test b_new == b_ref
        end
        checked += 1
    end
    @test checked == length(fixtures)
end
