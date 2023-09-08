using SparseArrays


function apply_dirichlet_bc!(A::SparseMatrixCSC, b; bc_nodes, bc_values)
    # Add contribution from BCs to the RHS
    x_bc = zeros(length(b))
    x_bc[bc_nodes] .= bc_values
    b .-= A * x_bc
    # Zero out rows and columns corresponding to BCs
    I, J, V = findnz(A)
    row_indices = findall(in.(I, Ref(bc_nodes)))
    col_indices = findall(in.(J, Ref(bc_nodes)))
    V[col_indices] .= 0.0
    V[row_indices] .= 0.0
    diag_values = SparseArrays.diag(A)[bc_nodes]
    A.nzval .= V
    # Ensure Dirichlet BCs are satisfied
    diag_indices = SparseArrays.diagind(A)[bc_nodes]
    A[diag_indices] .= diag_values
    b[bc_nodes] .= bc_values .* diag_values
    dropzeros!(A)
end
