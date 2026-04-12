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
            B[A.rowval[idx], j] = A.nzval[idx]
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
