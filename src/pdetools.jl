using SparseArrays

includet("numpytools.jl")


function apply_dirichlet_bc!(A::SparseMatrixCSC, b; nodes, vals)
    diag_inds = SparseArrays.diagind(A)[nodes]
    diag_vals = SparseArrays.diag(A)[nodes]
    # Add contribution from BCs to the RHS
    x_bc = multihotvec(nodes, length(b), vals=vals)
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
    diag_inds = SparseArrays.diagind(A)[nodes]
    diag_vals = SparseArrays.diag(A)[nodes]
    # Add contribution from BCs to the RHS
    x_bc = multihotvec(nodes, length(b), vals=vals)
    b .-= A * x_bc
    # Zero out rows and columns corresponding to BCs
    I, J, _ = findnz(A)
    row_inds = overlap_indices🚀(I, nodes)
    col_inds = overlap_indices🚀(J, nodes)
    A.nzval[union(row_inds, col_inds)] .= 0.0
    # Ensure Dirichlet BCs are satisfied
    A[diag_inds] .= diag_vals
    b[nodes] .= vals .* diag_vals
    dropzeros!(A)
end
