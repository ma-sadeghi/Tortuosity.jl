"""
    apply_dirichlet_bc!(A::SparseMatrixCSC, b; nodes, vals)

Reference implementation of Dirichlet BC application. Uses single-threaded
`overlap_indices`. Kept as a readable baseline for verifying the optimized
[`apply_dirichlet_bc_fast!`](@ref).

Zeroes out rows and columns of `A` for boundary nodes, sets the diagonal
to its original value, and adjusts `b` so that `x[nodes] .= vals` upon solve.
"""
function apply_dirichlet_bc!(A::SparseMatrixCSC, b; nodes, vals)
    diag_inds = SparseArrays.diagind(A)[nodes]
    diag_vals = SparseArrays.diag(A)[nodes]
    # Add contribution from BCs to the RHS
    x_bc = multihotvec(nodes, length(b); vals=vals)
    b .-= A * x_bc
    # Zero out rows and columns corresponding to BCs
    I, J, _ = findnz(A)
    row_inds = overlap_indices(I, nodes)
    col_inds = overlap_indices(J, nodes)
    A.nzval[union(row_inds, col_inds)] .= 0.0
    # Ensure Dirichlet BCs are satisfied
    A[diag_inds] .= diag_vals
    b[nodes] .= vals .* diag_vals
    dropzeros!(A)
end

"""
    apply_dirichlet_bc_fast!(A, b; nodes, vals)

Apply Dirichlet boundary conditions to the linear system `A x = b` in place.
Zeroes out rows and columns of `A` for boundary `nodes`, preserves the original
diagonal, and adjusts `b` so that `x[nodes] .= vals` upon solve. Uses
multi-threaded `overlap_indices_fast` on CPU and KA kernels on GPU.

# Keyword Arguments
- `nodes`: vector of node indices where Dirichlet conditions are applied.
- `vals`: corresponding boundary values.
"""
function apply_dirichlet_bc_fast!(A::SparseMatrixCSC, b; nodes, vals)
    # NOTE: This is the standard way to apply Dirichlet BCs:
    #  - Add contribution from BCs to the non-BC nodes in the RHS
    #  - Zero out rows and columns corresponding to BC nodes to keep A symmetric
    #  - Modify diagonal and RHS corresponding to BC nodes to satisfy Dirichlet BCs

    # Fetch the diagonal before it's zeroed out
    diag_inds = SparseArrays.diagind(A)[nodes]
    diag_vals = SparseArrays.diag(A)[nodes]
    # Add contribution from BCs to the RHS
    x_bc = multihotvec(nodes, length(b); vals=vals)
    b .-= A * x_bc

    # Zero out rows and columns corresponding to BCs
    I, J, _ = findnz(A)
    row_inds = overlap_indices_fast(I, nodes)
    col_inds = overlap_indices_fast(J, nodes)
    A.nzval[union(row_inds, col_inds)] .= 0.0

    # Apply BCs x[i] = vals[i] via diag[i] * x[i] = diag[i] * vals[i]
    A[diag_inds] .= diag_vals
    b[nodes] .= vals .* diag_vals
    dropzeros!(A)
end

function apply_dirichlet_bc_fast!(A::PortableSparseCSC, b; nodes, vals)
    # NOTE: This is the standard way to apply Dirichlet BCs:
    #  - Add contribution from BCs to the non-BC nodes in the RHS
    #  - Zero out rows and columns corresponding to BC nodes to keep A symmetric
    #  - Modify diagonal and RHS corresponding to BC nodes to satisfy Dirichlet BCs
    diag_vals = get_diag(A)  # Fetch the diagonal before it's zeroed out
    # Transfer vals to GPU if needed, using b as template
    gpu_vals = vals
    if _on_gpu(b) && !(vals isa typeof(b))
        gpu_vals = similar(b, eltype(vals), length(vals))
        copyto!(gpu_vals, vals)
    end
    x_bc = multihotvec(nodes, length(b); vals=gpu_vals, template=b)
    b .-= A * x_bc
    zero_rows_cols!(A, nodes)
    # Apply BCs x[i] = vals[i] via diag[i] * x[i] = diag[i] * vals[i]
    set_diag!(A, diag_vals)
    gpu_nodes = similar(A.rowval, eltype(nodes), length(nodes))
    copyto!(gpu_nodes, nodes)
    b[gpu_nodes] .= gpu_vals .* diag_vals[gpu_nodes]
    dropzeros!(A)
end
