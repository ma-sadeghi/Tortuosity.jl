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

function apply_dirichlet_bc🚀!(A::SparseMatrixCSC, b; nodes, vals)
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
    row_inds = overlap_indices🚀(I, nodes)
    col_inds = overlap_indices🚀(J, nodes)
    A.nzval[union(row_inds, col_inds)] .= 0.0

    # Apply BCs x[i] = vals[i] via diag[i] * x[i] = diag[i] * vals[i]
    A[diag_inds] .= diag_vals
    b[nodes] .= vals .* diag_vals
    dropzeros!(A)
end

function apply_dirichlet_bc🚀!(A::CUDA.CUSPARSE.CuSparseMatrixCSC, b; nodes, vals)
    # NOTE: This is the standard way to apply Dirichlet BCs:
    #  - Add contribution from BCs to the non-BC nodes in the RHS
    #  - Zero out rows and columns corresponding to BC nodes to keep A symmetric
    #  - Modify diagonal and RHS corresponding to BC nodes to satisfy Dirichlet BCs
    diag_vals = get_diag(A)  # Fetch the diagonal before it's zeroed out
    x_bc = Tortuosity.multihotvec(nodes, length(b); vals=cu(vals), gpu=true)
    b .-= A * x_bc
    zero_rows_cols!(A, nodes)
    # Apply BCs x[i] = vals[i] via diag[i] * x[i] = diag[i] * vals[i]
    set_diag!(A, diag_vals)
    b[nodes] .= vals .* diag_vals[nodes]
    dropzeros!(A)
end
