# Post-hoc Dirichlet elimination. Nothing in the package calls it any more:
# `build_steady_system` eliminates the boundary values while it assembles, and
# the transient path zeroes rows instead of eliminating. What is left here is
# the reference statement of the convention — the readable, obviously-correct
# form that `test_assembly.jl` and `test_impl_parity.jl` check the fused
# assembler against. Keep it working; do not put it back on the production path.
#
# A Dirichlet value is imposed as `diag[i] * x[i] = diag[i] * val[i]`, which
# preserves the original diagonal and keeps `A` symmetric. That encoding
# degenerates when the node has no neighbours: its degree, and therefore its
# diagonal, is zero, so the row reads `0 = 0`, `dropzeros!` deletes it, and the
# prescribed value is never applied.
#
# The consequence is not a crash but a plausible wrong answer. An isolated pore
# voxel on the inlet face keeps `c = 0` while sitting on a `c = 1` face, which
# drags the inlet-slice mean below the imposed drop, inflates `D_eff`, and
# reports a tortuosity below 1 — impossible, from a solve that reports success.
# Reachable on any untrimmed image: 33 of the 36 blob fixtures in
# `test/test_gpu_parity.jl` contain such a voxel.
#
# Scaling those rows by 1 instead enforces `x[i] = val[i]` exactly and keeps `A`
# symmetric, since such a row and column are empty apart from the diagonal. It
# is the identity everywhere else: any node with at least one neighbour has a
# positive degree, and `SteadyDiffusionProblem` already requires `D > 0` across
# the pore space.
_unit_where_zero(d) = ifelse.(iszero.(d), one(eltype(d)), d)

"""
    apply_dirichlet_bc!(A::SparseMatrixCSC, b; nodes, vals)

Reference implementation of Dirichlet BC application. Uses single-threaded
`overlap_indices`. Kept as a readable baseline for verifying
[`apply_dirichlet_bc_fast!`](@ref), which is itself parity material rather than
production code — see the note at the top of this file.

Zeroes out rows and columns of `A` for boundary nodes, sets the diagonal
to its original value, and adjusts `b` so that `x[nodes] .= vals` upon solve.
"""
function apply_dirichlet_bc!(A::SparseMatrixCSC, b; nodes, vals)
    diag_inds = SparseArrays.diagind(A)[nodes]
    diag_vals = _unit_where_zero(SparseArrays.diag(A)[nodes])
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

The parity reference for [`build_steady_system`](@ref), which produces the same
`(A, b)` without ever building the pre-elimination Laplacian. No production
caller remains — see the note at the top of this file.

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
    diag_vals = _unit_where_zero(SparseArrays.diag(A)[nodes])
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

# Requires a structurally-present diagonal in every boundary column: the update
# goes through `set_diag!`, which rewrites existing slots and cannot insert.
# `laplacian(::PortableSparseCSC)` guarantees this — it emits a diagonal in
# every column, including zero ones. Wrapping a `SparseMatrixCSC` Laplacian
# would not, because `spdiagm(d) - A` prunes a zero-valued diagonal.
function apply_dirichlet_bc_fast!(A::PortableSparseCSC, b; nodes, vals)
    # NOTE: This is the standard way to apply Dirichlet BCs:
    #  - Add contribution from BCs to the non-BC nodes in the RHS
    #  - Zero out rows and columns corresponding to BC nodes to keep A symmetric
    #  - Modify diagonal and RHS corresponding to BC nodes to satisfy Dirichlet BCs
    diag_vals = get_diag(A)  # Fetch the diagonal before it's zeroed out
    # Transfer/convert `vals` so it lives on the same device and has the same
    # eltype as `b`. Previously this checked `vals isa typeof(b)`, which was
    # brittle to CuArray/MtlArray buffer parameters and didn't cover the
    # eltype case.
    gpu_vals = if _on_gpu(b) == _on_gpu(vals) && eltype(vals) === eltype(b)
        vals
    else
        v = similar(b, eltype(b), length(vals))
        copyto!(v, vals)
        v
    end
    x_bc = multihotvec(nodes, length(b); vals=gpu_vals, template=b)
    b .-= A * x_bc
    zero_rows_cols!(A, nodes)
    # Patch the boundary rows only. `set_diag!` writes the whole diagonal, and a
    # zero-degree *interior* node is deliberately left alone: it carries no flux
    # and its row is already consistent at 0 = 0.
    gpu_nodes = similar(A.rowval, eltype(nodes), length(nodes))
    copyto!(gpu_nodes, nodes)
    diag_vals[gpu_nodes] .= _unit_where_zero(diag_vals[gpu_nodes])
    # Apply BCs x[i] = vals[i] via diag[i] * x[i] = diag[i] * vals[i]
    set_diag!(A, diag_vals)
    b[gpu_nodes] .= gpu_vals .* diag_vals[gpu_nodes]
    dropzeros!(A)
end
