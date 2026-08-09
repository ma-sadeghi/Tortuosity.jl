# Fused assembly of the steady diffusion system: two passes over the pore image
# emit the Dirichlet-eliminated Laplacian and its right-hand side straight into
# CSC arrays, with no connectivity list, no adjacency matrix, no edge-weight
# vector and no post-hoc elimination.

# Face neighbours in ascending linear-index order — offsets -nx·ny, -nx, -1 then
# +1, +nx, +nx·ny. Pore ordinals are monotone in linear grid index, so writing a
# column in this order with the diagonal between the two halves gives strictly
# ascending row indices for free; no sort is needed anywhere.
const _LOWER_NEIGHBOURS = ((0, 0, -1), (0, -1, 0), (-1, 0, 0))
const _UPPER_NEIGHBOURS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
const _NEIGHBOURS = (_LOWER_NEIGHBOURS..., _UPPER_NEIGHBOURS...)

# The coordinate along the transport axis, which is the only thing that decides
# whether a voxel is a boundary node.
@inline _face_coord(i, j, k, d) = d == 1 ? i : (d == 2 ? j : k)

# `SteadyDiffusionProblem` imposes c = 1 on the low face and c = 0 on the high
# face, so membership of the Dirichlet set is a coordinate test rather than a
# node list — the boundary nodes never have to be enumerated on the host.
@inline _is_bc(c, nbc) = (c == 1) | (c == nbc)

# `nothing` is the uniform-diffusivity case: every edge weight is `D0` and no
# diffusivity array is read at all.
@inline _node_diffusivity(::Nothing, D0, i, j, k) = D0
@inline _node_diffusivity(D, D0, i, j, k) = @inbounds D[i, j, k]

@inline _edge_weight(::Nothing, D0, da, db) = D0
# Harmonic mean of the two half-cell conductances — see `interpolate_edge_values`
# for the derivation. Symmetric in its arguments down to the last bit.
@inline _edge_weight(D, D0, da, db) = 2 * da * db / (da + db)

"""
    _steady_count_kernel!(counts, b, idx, D, nx, ny, nz, bcdim, nbc, D0)

KA kernel, pass 1: one thread per grid voxel writes the number of stored entries
its column will hold, and that column's right-hand-side value.

A boundary column keeps its diagonal and nothing else. A free column keeps its
diagonal plus one entry per free neighbour, and is empty when the node has no
neighbours at all — its diagonal would be a stored zero, which is what
[`dropzeros!`](@ref) exists to remove.
"""
@kernel function _steady_count_kernel!(
    counts, b, @Const(idx), D, nx, ny, nz, bcdim, nbc, D0,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        c0 = idx[i, j, k]
        if c0 > 0
            Tv = typeof(D0)
            Tb = eltype(b)
            Ti = eltype(counts)
            fc = _face_coord(i, j, k, bcdim)
            self_bc = _is_bc(fc, nbc)
            da = _node_diffusivity(D, D0, i, j, k)

            deg = zero(Tv)
            nfree = 0
            rhs = zero(Tb)
            for (di, dj, dk) in _NEIGHBOURS
                ii, jj, kk = i + di, j + dj, k + dk
                (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
                q = idx[ii, jj, kk]
                q > 0 || continue
                w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, ii, jj, kk))
                deg += w
                self_bc && continue
                qc = _face_coord(ii, jj, kk, bcdim)
                if _is_bc(qc, nbc)
                    # Only the inlet face carries a nonzero value, so an outlet
                    # neighbour contributes nothing to the folded-in load.
                    qc == 1 && (rhs += Tb(w))
                else
                    nfree += 1
                end
            end

            if self_bc
                # A zero-degree boundary node has nothing to scale, so it is
                # pinned with a unit diagonal instead — see `_unit_where_zero`.
                d = iszero(deg) ? one(Tv) : deg
                counts[c0] = one(Ti)
                b[c0] = fc == 1 ? Tb(d) : zero(Tb)
            else
                counts[c0] = iszero(deg) ? zero(Ti) : Ti(nfree + 1)
                b[c0] = rhs
            end
        end
    end
end

"""
    _steady_fill_kernel!(rowval, nzval, colptr, idx, D, nx, ny, nz, bcdim, nbc, D0)

KA kernel, pass 2: one thread per grid voxel owns its column's whole contiguous
slot range, so there are no atomics and the output is bit-reproducible.
"""
@kernel function _steady_fill_kernel!(
    rowval, nzval, @Const(colptr), @Const(idx), D, nx, ny, nz, bcdim, nbc, D0,
)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        c0 = idx[i, j, k]
        if c0 > 0
            pos = colptr[c0]
            # An empty column is a free node with no neighbours: nothing to write.
            if colptr[c0 + 1] > pos
                Tv = typeof(D0)
                Ti = eltype(rowval)
                fc = _face_coord(i, j, k, bcdim)
                self_bc = _is_bc(fc, nbc)
                da = _node_diffusivity(D, D0, i, j, k)
                deg = zero(Tv)

                for (di, dj, dk) in _LOWER_NEIGHBOURS
                    ii, jj, kk = i + di, j + dj, k + dk
                    (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
                    q = idx[ii, jj, kk]
                    q > 0 || continue
                    w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, ii, jj, kk))
                    deg += w
                    (self_bc || _is_bc(_face_coord(ii, jj, kk, bcdim), nbc)) && continue
                    rowval[pos] = q
                    nzval[pos] = -w
                    pos += one(Ti)
                end

                # The diagonal sits between the two halves, which is exactly its
                # sorted position; its value is only known once every neighbour
                # has been visited, so the slot is reserved and written last.
                diag_slot = pos
                self_bc || (pos += one(Ti))

                for (di, dj, dk) in _UPPER_NEIGHBOURS
                    ii, jj, kk = i + di, j + dj, k + dk
                    (1 <= ii <= nx && 1 <= jj <= ny && 1 <= kk <= nz) || continue
                    q = idx[ii, jj, kk]
                    q > 0 || continue
                    w = _edge_weight(D, D0, da, _node_diffusivity(D, D0, ii, jj, kk))
                    deg += w
                    (self_bc || _is_bc(_face_coord(ii, jj, kk, bcdim), nbc)) && continue
                    rowval[pos] = q
                    nzval[pos] = -w
                    pos += one(Ti)
                end

                rowval[diag_slot] = c0
                nzval[diag_slot] = (self_bc && iszero(deg)) ? one(Tv) : deg
            end
        end
    end
end

"""
    build_steady_system(img; nnodes, axis, D=nothing, T=Float64)

Assemble the steady diffusion system `(A, b)` directly from the pore mask `img`,
with Dirichlet values already eliminated: `c = 1` on the low face along `axis`
and `c = 0` on the high one.

`A` is `SparseMatrixCSC` for a host mask and [`PortableSparseCSC`](@ref) for a
device one; either way its row indices are ascending within every column and its
sparsity pattern carries no explicit zeros.

Equivalent to `laplacian(build_adjacency_matrix(build_connectivity_list(img)))`
followed by [`apply_dirichlet_bc_fast!`](@ref), but it never materialises the
connectivity list, the adjacency matrix, the edge-weight vector or the
pre-elimination Laplacian, and it needs no compaction pass afterwards. Those
functions are kept for the transient path and as the reference the parity tests
compare against.

# Keyword Arguments
- `nnodes`: number of pore voxels, i.e. `count(img)`.
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `D`: diffusivity array matching `img`, or `nothing` for uniform `D = 1`.
- `T`: element type of `b`. `A` follows `D`'s element type when one is given.
"""
function build_steady_system(img; nnodes, axis, D=nothing, T=Float64)
    nx, ny, nz = size(img)
    bcdim = axis_dim(axis)
    nbc = size(img, bcdim)
    on_gpu = _on_gpu(img)
    # Int32 halves the index traffic on the device; the host path keeps Int so
    # its CSC stays a plain SparseMatrixCSC{Tv,Int}.
    Ti = on_gpu ? Int32 : Int
    Tv = isnothing(D) ? T : eltype(D)
    D0 = one(Tv)

    # Pore numbering: an inclusive scan over the mask hands each pore voxel its
    # ordinal, and masking the solids back to zero lets `idx` double as the
    # pore/solid test — so neither kernel below ever reads `img`.
    idx = similar(img, Ti)
    cumsum!(vec(idx), vec(img))
    idx .*= img
    backend = get_backend(idx)
    # 256 threads laid out along the contiguous dimension, so a warp reads one
    # run of `idx` and its two in-plane neighbour rows coalesced.
    wg = (64, 4, 1)

    counts = similar(idx, Ti, nnodes)
    b = similar(idx, T, nnodes)
    _steady_count_kernel!(backend, wg)(
        counts, b, idx, D, nx, ny, nz, bcdim, nbc, D0; ndrange=(nx, ny, nz),
    )
    KernelAbstractions.synchronize(backend)

    scan = accumulate(+, counts)
    _free!(counts)
    colptr = similar(idx, Ti, nnodes + 1)
    _build_colptr_kernel!(backend)(colptr, scan, nnodes; ndrange=max(nnodes, 1))
    KernelAbstractions.synchronize(backend)
    _free!(scan)

    # One single-slot host read, the idiom `_build_connectivity_list_ka` uses.
    nnz_A = Int(Array(@view colptr[end:end])[1]) - 1
    rowval = similar(idx, Ti, nnz_A)
    nzval = similar(b, Tv, nnz_A)
    _steady_fill_kernel!(backend, wg)(
        rowval, nzval, colptr, idx, D, nx, ny, nz, bcdim, nbc, D0; ndrange=(nx, ny, nz),
    )
    KernelAbstractions.synchronize(backend)
    _free!(idx)

    # Symmetric to the last bit: an edge contributes `-w` to both `A[p, q]` and
    # `A[q, p]` from the same `_edge_weight` call, which is itself symmetric in
    # its two arguments, and Dirichlet elimination empties a boundary node's row
    # and its column alike. `test_assembly.jl` pins this.
    A = on_gpu ? PortableSparseCSC(nnodes, nnodes, colptr, rowval, nzval; symmetric=true) :
        SparseMatrixCSC(nnodes, nnodes, colptr, rowval, nzval)
    return A, b
end
