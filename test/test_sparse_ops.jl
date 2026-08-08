# Direct unit tests for PortableSparseCSC mutators (dropzeros!, set_diag!,
# get_diag, zero_rows_cols!, zero_rows!). These run on the CPU KA backend and
# need no GPU, complementing the GPU parity suite that only exercises the
# same operations indirectly through apply_dirichlet_bc_fast!.

using Test
using SparseArrays
using LinearAlgebra
using Tortuosity: PortableSparseCSC, set_diag!, get_diag,
    zero_rows_cols!, zero_rows!, dropzeros!

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
            # Accumulate rather than assign: two entries for the same (row, col)
            # is what a kernel emitting a duplicate produces, and that is
            # exactly how `mul!` would sum them. Assigning would hide it.
            B[A.rowval[idx], j] += A.nzval[idx]
        end
    end
    return B
end

@testset "dropzeros!" begin
    @testset "no-op when there are no explicit zeros" begin
        A = sprand(Float64, 12, 12, 0.4)
        SparseArrays.dropzeros!(A)
        P = sparse_to_portable(A)
        original_nnz = nnz(P)
        dropzeros!(P)
        @test nnz(P) == original_nnz
        @test to_dense(P) ≈ Array(A)
    end

    @testset "drops an explicit zero in the middle of a column" begin
        # Column 2 has entries [1 → 0.0, 2 → 2.0]; the zero should be dropped
        A = SparseMatrixCSC(3, 3, [1, 2, 4, 5], [1, 1, 2, 3], Float64[1, 0, 2, 3])
        P = sparse_to_portable(A)
        @test nnz(P) == 4
        dropzeros!(P)
        @test nnz(P) == 3
        @test to_dense(P) ≈ Array(SparseArrays.dropzeros(A))
    end

    @testset "drops all entries in a column" begin
        A = SparseMatrixCSC(3, 3, [1, 3, 3, 4], [1, 2, 3], Float64[0, 0, 5])
        P = sparse_to_portable(A)
        dropzeros!(P)
        @test nnz(P) == 1
        @test P.colptr == [1, 1, 1, 2]
        @test to_dense(P) ≈ Array(SparseArrays.dropzeros(A))
    end

    @testset "all-zero matrix compacts to empty" begin
        A = SparseMatrixCSC(3, 3, [1, 2, 3, 4], [1, 2, 3], zeros(Float64, 3))
        P = sparse_to_portable(A)
        dropzeros!(P)
        @test nnz(P) == 0
        @test P.colptr == [1, 1, 1, 1]
        @test to_dense(P) ≈ zeros(3, 3)
    end

    @testset "alternating zeros and nonzeros across columns" begin
        # colptr = [1, 3, 5, 7, 9]; four columns with 2 entries each
        A = SparseMatrixCSC(
            4, 4, [1, 3, 5, 7, 9],
            [1, 2, 1, 2, 3, 4, 3, 4],
            Float64[0, 1, 2, 0, 0, 3, 4, 0],
        )
        P = sparse_to_portable(A)
        dropzeros!(P)
        @test nnz(P) == 4
        @test to_dense(P) ≈ Array(SparseArrays.dropzeros(A))
    end

    @testset "larger random fuzz vs SparseArrays.dropzeros" begin
        for seed in (1, 17, 42)
            A_full = sprand(Float64, 40, 40, 0.25)
            # Zero out a random subset of entries to create explicit zeros
            nz = nonzeros(A_full)
            idx = 1:2:length(nz)
            nz[idx] .= 0
            P = sparse_to_portable(A_full)
            dropzeros!(P)
            @test to_dense(P) ≈ Array(SparseArrays.dropzeros(A_full))
        end
    end
end

@testset "set_diag!" begin
    @testset "updates existing diagonal entries" begin
        A = sparse(Float64[1 0 2; 0 3 0; 4 0 5])
        P = sparse_to_portable(A)
        set_diag!(P, Float64[10, 20, 30])
        @test to_dense(P) ≈ Float64[10 0 2; 0 20 0; 4 0 30]
    end

    @testset "structurally-absent diagonal entries are left alone" begin
        A = SparseMatrixCSC(3, 3, [1, 2, 2, 4], [1, 2, 3], Float64[1, 4, 5])
        P = sparse_to_portable(A)
        set_diag!(P, Float64[10, 20, 30])
        dense_P = to_dense(P)
        @test dense_P[1, 1] ≈ 10.0
        @test dense_P[2, 2] == 0.0
        @test dense_P[3, 3] ≈ 30.0
    end

    @testset "dimension mismatch throws" begin
        A = sparse(Diagonal(Float64[1, 2, 3]))
        P = sparse_to_portable(A)
        @test_throws DimensionMismatch set_diag!(P, Float64[1, 2])
    end
end

@testset "get_diag" begin
    @testset "extracts the diagonal" begin
        A = sparse(Float64[1 0 2; 0 3 0; 4 0 5])
        P = sparse_to_portable(A)
        @test Array(get_diag(P)) ≈ Float64[1, 3, 5]
    end

    @testset "structurally-absent diagonal reads as zero" begin
        A = SparseMatrixCSC(3, 3, [1, 2, 2, 4], [1, 2, 3], Float64[1, 4, 5])
        P = sparse_to_portable(A)
        @test Array(get_diag(P)) ≈ Float64[1, 0, 5]
    end

    @testset "empty matrix returns empty vector" begin
        P = PortableSparseCSC(0, 0, Int[1], Int[], Float64[])
        @test isempty(Array(get_diag(P)))
    end
end

@testset "zero_rows_cols!" begin
    @testset "zeros one row and one column" begin
        A = sparse(Float64[1 2 3; 4 5 6; 7 8 9])
        P = sparse_to_portable(A)
        zero_rows_cols!(P, [2])
        @test to_dense(P) ≈ Float64[1 0 3; 0 0 0; 7 0 9]
    end

    @testset "empty idxs is a no-op" begin
        A = sparse(Float64[1 2; 3 4])
        P = sparse_to_portable(A)
        zero_rows_cols!(P, Int[])
        @test to_dense(P) ≈ Array(A)
    end

    @testset "out-of-range idxs are silently filtered" begin
        A = sparse(Float64[1 2; 3 4])
        P = sparse_to_portable(A)
        zero_rows_cols!(P, [0, 5])
        @test to_dense(P) ≈ Array(A)
    end

    @testset "multiple rows/cols zeroed at once" begin
        A = sparse(Float64[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16])
        P = sparse_to_portable(A)
        zero_rows_cols!(P, [1, 3])
        expected = Float64[
            0 0 0 0
            0 6 0 8
            0 0 0 0
            0 14 0 16
        ]
        @test to_dense(P) ≈ expected
    end
end

@testset "zero_rows!" begin
    @testset "zeros specified rows and drops structural zeros" begin
        A = sparse(Float64[1 2; 3 4])
        P = sparse_to_portable(A)
        zero_rows!(P, [1])
        @test nnz(P) == 2
        @test to_dense(P) ≈ Float64[0 0; 3 4]
    end

    @testset "empty rows list is a no-op" begin
        A = sparse(Float64[1 2; 3 4])
        P = sparse_to_portable(A)
        zero_rows!(P, Int[])
        @test to_dense(P) ≈ Array(A)
    end

    @testset "multiple rows zeroed at once" begin
        A = sparse(Float64[1 2 3; 4 5 6; 7 8 9])
        P = sparse_to_portable(A)
        zero_rows!(P, [1, 3])
        @test to_dense(P) ≈ Float64[0 0 0; 4 5 6; 0 0 0]
    end
end

# --- SpMV ---
#
# `mul!` is the innermost loop of both solvers: LinearSolve's CG calls it once
# per Krylov iteration and the transient RHS calls it once per ODE stage. The
# tests below pin the two things an optimised kernel could plausibly get wrong —
# the result itself, and the fact that `y` is *overwritten* rather than
# accumulated into.

using Random
using Tortuosity: build_adjacency_matrix, build_connectivity_list, laplacian

@testset "mul! (SpMV)" begin
    @testset "matches dense multiplication on random matrices" begin
        for (m, n, p, seed) in ((20, 20, 0.2, 1), (37, 37, 0.1, 2), (13, 21, 0.3, 3), (21, 13, 0.3, 4))
            rng = MersenneTwister(seed)
            A = sprand(rng, Float64, m, n, p)
            P = sparse_to_portable(A)
            x = randn(rng, n)
            y = zeros(m)
            mul!(y, P, x)
            @test y ≈ Array(A) * x
        end
    end

    @testset "overwrites y instead of accumulating into it" begin
        # A kernel that dropped the leading fill! would still pass a single-shot
        # test but corrupt every Krylov iteration after the first.
        A = sprand(MersenneTwister(5), Float64, 16, 16, 0.25)
        P = sparse_to_portable(A)
        x = randn(MersenneTwister(6), 16)
        expected = Array(A) * x
        y = fill(1e6, 16)                      # deliberately dirty buffer
        mul!(y, P, x)
        @test y ≈ expected
        mul!(y, P, x)                          # a second call must not double it
        @test y ≈ expected
    end

    @testset "is linear in x" begin
        A = sprand(MersenneTwister(7), Float64, 24, 24, 0.2)
        P = sparse_to_portable(A)
        x1 = randn(MersenneTwister(8), 24)
        x2 = randn(MersenneTwister(9), 24)
        @test P * (x1 .+ 2 .* x2) ≈ (P * x1) .+ 2 .* (P * x2)
    end

    @testset "structurally empty matrices give a zero result" begin
        P = PortableSparseCSC(4, 4, [1, 1, 1, 1, 1], Int[], Float64[])
        y = fill(9.0, 4)
        mul!(y, P, ones(4))
        @test all(iszero, y)
        @test all(iszero, P * ones(4))
    end

    @testset "a zero-sized matrix is handled without indexing off the end" begin
        P = PortableSparseCSC(0, 0, Int[1], Int[], Float64[])
        @test isempty(mul!(Float64[], P, Float64[]))
        @test isempty(P * Float64[])
    end

    @testset "the * operator promotes element types" begin
        A = sparse(Float64[1 0; 0 2])
        P = sparse_to_portable(A)
        y = P * Float32[1, 1]
        @test eltype(y) === Float64
        @test y ≈ [1.0, 2.0]
    end
end

@testset "PortableSparseCSC accessors and display" begin
    A = sparse(Float64[1 0 2; 0 3 0; 4 0 5])
    P = sparse_to_portable(A)
    @test size(P) == (3, 3)
    @test nnz(P) == nnz(A)
    @test nonzeros(P) == nonzeros(A)
    @test rowvals(P) == rowvals(A)
    @test SparseArrays.getcolptr(P) == SparseArrays.getcolptr(A)

    io = IOBuffer()
    show(io, P)
    s = String(take!(io))
    @test occursin("PortableSparseCSC", s)
    @test occursin("nnz=$(nnz(A))", s)
    @test occursin("3×3", s)
end

# --- Laplacian assembly branches ---
#
# `_laplacian_entries_kernel!` chooses where to splice in the diagonal by
# scanning each column's row indices. The three reachable shapes are: the
# diagonal falls between two off-diagonal entries, it falls after all of them,
# or the column is empty. All three occur in real connectivity lists — the
# last pore voxel has only lower-numbered neighbours, and an isolated voxel has
# none — so each is exercised against the CPU reference here.

@testset "laplacian(::PortableSparseCSC) column shapes" begin
    cases = Dict(
        "diagonal splices between two entries" => sparse(Float64[0 1 0; 1 0 1; 0 1 0]),
        "diagonal appends after all entries" => sparse(Float64[0 1 0; 0 0 1; 0 0 0]),
        "empty leading column" => sparse(Float64[0 0 1; 0 0 1; 0 0 0]),
        "empty trailing column" => sparse(Float64[0 1 0; 1 0 0; 0 0 0]),
        "single entry per column" => sparse(Float64[0 0 3; 4 0 0; 0 5 0]),
    )
    for (label, A) in cases
        @testset "$label" begin
            @test to_dense(laplacian(sparse_to_portable(A))) ≈ Array(laplacian(A))
        end
    end
end

@testset "laplacian(::PortableSparseCSC) on a real image with an isolated voxel" begin
    # An isolated pore voxel produces a structurally empty column — the branch
    # where the kernel must emit a lone (zero) diagonal entry.
    img = falses(6, 6, 6)
    img[2:4, 2:4, 2:4] .= true
    img[6, 6, 6] = true                        # isolated
    conns = build_connectivity_list(img)
    n = count(img)
    am = build_adjacency_matrix(conns; n=n, weights=ones(size(conns, 1)))
    L_ref = laplacian(am)
    L_port = laplacian(sparse_to_portable(am))
    @test to_dense(L_port) ≈ Array(L_ref)
    # The isolated node is the last one in column-major order and must have a
    # zero row: no neighbours, hence no degree and no coupling.
    @test all(iszero, Array(L_ref)[n, :])
end

@testset "laplacian(::PortableSparseCSC) rejects a non-square matrix" begin
    A = sprand(MersenneTwister(11), Float64, 4, 6, 0.4)
    @test_throws AssertionError laplacian(sparse_to_portable(A))
end

# --- Self-loops (a nonzero A[j,j]) ---
#
# `build_connectivity_list` cannot produce a self-loop — a voxel is never its
# own face-neighbour — but `laplacian` takes an arbitrary adjacency matrix, and
# a column that already holds a diagonal gains no new entry. Sizing the output
# as `nnz(A) + n` therefore over-allocates by one slot per such column and
# leaves it uninitialised; the garbage row index then reaches
# `_spmv_kernel!`, which indexes `@inbounds` and writes through an atomic.
# These tests keep the output structure exact for any input.

# Assert the CSC arrays are internally consistent and index only real rows.
#
# The load-bearing check is `colptr[end] == length(nzval) + 1`: it catches an
# over-allocated array whose tail was never written, regardless of what the
# uninitialised bytes happen to contain. The `1 <= rowval <= m` bound only fires
# when that garbage lands outside the valid row range — which it did for the bug
# this file was written against (the slot held a pointer-sized value), but that
# is luck, not a guarantee. Returns whether the structure is sound so callers
# can skip operations that would read out of bounds.
function check_portable_structure(P)
    m, n = size(P)
    @test length(P.colptr) == n + 1
    @test P.colptr[1] == 1
    @test issorted(P.colptr)
    @test P.colptr[end] == length(P.nzval) + 1
    @test length(P.rowval) == length(P.nzval)
    @test all(1 .<= P.rowval .<= m)
    return length(P.colptr) == n + 1 &&
           P.colptr[end] == length(P.nzval) + 1 &&
           length(P.rowval) == length(P.nzval) &&
           all(1 .<= P.rowval .<= m)
end

@testset "laplacian(::PortableSparseCSC) with self-loops" begin
    cases = Dict(
        "self-loop alone in its column" => sparse(Float64[2 0; 0 3]),
        "self-loop first in its column" => sparse(Float64[2 1; 1 0]),
        "self-loop last in its column" => sparse(Float64[0 1; 1 3]),
        "self-loop between off-diagonals" => sparse(Float64[0 1 0; 1 5 1; 0 1 0]),
        "self-loops in every column" => sparse(Float64[1 2 0; 2 1 3; 0 3 1]),
        "mixed: some columns have one, some do not" => sparse(Float64[1 1 0; 1 0 1; 0 1 2]),
    )
    for (label, A) in cases
        @testset "$label" begin
            L = laplacian(sparse_to_portable(A))
            # Structure must be exact — no slot left unwritten.
            check_portable_structure(L)
            # A diagonal is added only where one is *structurally* absent. A
            # stored-but-zero A[j,j] still counts as present, so this must ask
            # the sparsity pattern, not the values.
            missing_diag = count(j -> j ∉ rowvals(A)[nzrange(A, j)], 1:size(A, 2))
            @test nnz(L) == nnz(A) + missing_diag
            # …and the values must match the reference implementation, which
            # gets self-loops right for free via generic sparse arithmetic.
            @test to_dense(L) ≈ Array(laplacian(A))
        end
    end
end

@testset "laplacian(::PortableSparseCSC) survives a fuzz of arbitrary matrices" begin
    # Random sparsity puts diagonals in unpredictable places, including columns
    # where the self-loop is neither first nor last.
    for seed in (1, 13, 29, 61), (n, p) in ((8, 0.3), (25, 0.15), (40, 0.08))
        A = sprand(MersenneTwister(seed), Float64, n, n, p)
        L = laplacian(sparse_to_portable(A))
        # Only touch the entries once the structure is known to be sound: an
        # over-allocated tail makes `to_dense` throw and `mul!` write out of
        # bounds through an @inbounds atomic, which aborts the whole process
        # instead of reporting a failure.
        if check_portable_structure(L)
            @test to_dense(L) ≈ Array(laplacian(A))
            # The action is what LinearSolve and ROCK4 actually consume.
            x = randn(MersenneTwister(seed + 1000), n)
            @test L * x ≈ Array(laplacian(A)) * x
        end
    end
end

@testset "laplacian(::PortableSparseCSC) does not require sorted row indices" begin
    # The GPU `build_adjacency_matrix` scatters with atomics, so row indices
    # within a column arrive in arbitrary order. Values must be right anyway.
    rng = MersenneTwister(77)
    A = sprand(rng, Float64, 20, 20, 0.2)
    P = sparse_to_portable(A)
    for j in 1:size(P, 2)
        rng_slice = P.colptr[j]:(P.colptr[j + 1] - 1)
        perm = shuffle(rng, collect(rng_slice))
        P.rowval[rng_slice] = P.rowval[perm]
        P.nzval[rng_slice] = P.nzval[perm]
    end
    L = laplacian(P)
    check_portable_structure(L)
    @test to_dense(L) ≈ Array(laplacian(A))
end

@testset "laplacian(::PortableSparseCSC) vs the CPU reference: values equal, pattern differs" begin
    # The KA path emits a structural diagonal in every column. The CPU reference
    # is `spdiagm(degrees) - A`, and sparse subtraction prunes entries whose
    # *computed* value is zero — so the two disagree on `nnz` whenever
    # degrees[j] == A[j,j], most visibly for an isolated node where both are 0.
    #
    # This predates the self-loop fix and is benign in the package: values are
    # what every caller consumes, and `apply_dirichlet_bc_fast!` runs
    # `dropzeros!` afterwards. It is pinned here because every other parity
    # assertion compares dense forms only, so a genuine future divergence in
    # either direction would be invisible.
    img = falses(6, 6, 6)
    img[2:4, 2:4, 2:4] .= true
    img[6, 6, 6] = true                          # isolated ⇒ degree 0
    n = count(img)
    conns = build_connectivity_list(img)
    am = build_adjacency_matrix(conns; n=n, weights=ones(size(conns, 1)))

    L_ref = laplacian(am)
    L_port = laplacian(sparse_to_portable(am))

    @test to_dense(L_port) ≈ Array(L_ref)        # values: identical
    @test nnz(L_port) > nnz(L_ref)               # pattern: one extra per zero diagonal
    extra = nnz(L_port) - nnz(L_ref)
    @test extra == count(j -> iszero(Array(L_ref)[j, j]), 1:n)

    # The surplus is explicit zeros and nothing else, so the two patterns
    # coincide once those are dropped — which is what the solver path does.
    dropzeros!(L_port)
    @test nnz(L_port) == nnz(L_ref)
    @test Array(L_port.colptr) == SparseArrays.getcolptr(L_ref)
    @test to_dense(L_port) ≈ Array(L_ref)
end

@testset "laplacian(::PortableSparseCSC) sizes the output from colptr, not from nzval" begin
    # `PortableSparseCSC` never validates that `length(nzval)` matches the
    # colptr span. Sizing L from `nnz(am) + extras` would over-allocate here and
    # leave the tail uninitialised — the same class of bug as the self-loop case.
    oversized = PortableSparseCSC(2, 2, [1, 2, 3], [2, 1, 99, 99], [1.0, 1.0, NaN, NaN])
    L = laplacian(oversized)
    @test check_portable_structure(L)
    @test nnz(L) == 4                            # 2 stored entries + 2 diagonals
    @test to_dense(L) ≈ Array(laplacian(sparse(Float64[0 1; 1 0])))
end

@testset "laplacian(::PortableSparseCSC) handles a zero-sized matrix" begin
    L = laplacian(PortableSparseCSC(0, 0, Int[1], Int[], Float64[]))
    @test size(L) == (0, 0)
    @test nnz(L) == 0
    @test L.colptr == [1]
end
