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
    InletFlux

Compact data needed to reduce the physical inlet flux from a pore-vector
solution. `sources`, `targets`, and `weights` describe edges across the inlet
plane. `inlet` and `outlet` contain every pore ordinal on the two boundary
faces. `direct` is the unit-drop flux when the transport axis is only two
voxels long.
"""
struct InletFlux{Vs,Vt,Vw,Vin,Vout,T}
    sources::Vs
    targets::Vt
    weights::Vw
    inlet::Vin
    outlet::Vout
    direct::T
end

function _free!(flux::InletFlux)
    isnothing(flux.sources) || _free!(flux.sources)
    isnothing(flux.targets) || _free!(flux.targets)
    isnothing(flux.weights) || _free!(flux.weights)
    isnothing(flux.inlet) || _free!(flux.inlet)
    isnothing(flux.outlet) || _free!(flux.outlet)
    return nothing
end

@inline function _inlet_pair(p, nx, ny, bcdim)
    if bcdim == 1
        j = (p - 1) % ny + 1
        k = (p - 1) ÷ ny + 1
        return 1, j, k, 2, j, k
    elseif bcdim == 2
        i = (p - 1) % nx + 1
        k = (p - 1) ÷ nx + 1
        return i, 1, k, i, 2, k
    end
    i = (p - 1) % nx + 1
    j = (p - 1) ÷ nx + 1
    return i, j, 1, i, j, 2
end

@kernel function _inlet_flags_kernel!(flags, @Const(idx), nx, ny, bcdim)
    p = @index(Global)
    i1, j1, k1, i2, j2, k2 = _inlet_pair(p, nx, ny, bcdim)
    @inbounds flags[p] = (idx[i1, j1, k1] > 0) & (idx[i2, j2, k2] > 0)
end

@inline _store_source!(::Nothing, pos, source) = nothing
@inline function _store_source!(sources, pos, source)
    @inbounds sources[pos] = source
    return nothing
end

@kernel function _inlet_fill_kernel!(
    sources, targets, weights, @Const(scan), @Const(idx), D, nx, ny, bcdim, D0,
)
    p = @index(Global)
    i1, j1, k1, i2, j2, k2 = _inlet_pair(p, nx, ny, bcdim)
    @inbounds begin
        source = idx[i1, j1, k1]
        target = idx[i2, j2, k2]
        if (source > 0) & (target > 0)
            pos = scan[p]
            da = _node_diffusivity(D, D0, i1, j1, k1)
            db = _node_diffusivity(D, D0, i2, j2, k2)
            _store_source!(sources, pos, source)
            targets[pos] = target
            weights[pos] = _edge_weight(D, D0, da, db)
        end
    end
end

@inline function _face_point(p, nx, ny, bcdim, coord)
    if bcdim == 1
        j = (p - 1) % ny + 1
        k = (p - 1) ÷ ny + 1
        return coord, j, k
    elseif bcdim == 2
        i = (p - 1) % nx + 1
        k = (p - 1) ÷ nx + 1
        return i, coord, k
    end
    i = (p - 1) % nx + 1
    j = (p - 1) ÷ nx + 1
    return i, j, coord
end

@kernel function _face_flags_kernel!(flags, @Const(idx), nx, ny, bcdim, coord)
    p = @index(Global)
    i, j, k = _face_point(p, nx, ny, bcdim, coord)
    @inbounds flags[p] = idx[i, j, k] > 0
end

@kernel function _face_fill_kernel!(nodes, @Const(scan), @Const(idx), nx, ny, bcdim, coord)
    p = @index(Global)
    i, j, k = _face_point(p, nx, ny, bcdim, coord)
    @inbounds begin
        node = idx[i, j, k]
        node > 0 && (nodes[scan[p]] = node)
    end
end

function _build_face_nodes(idx, bcdim, coord)
    nx, ny, _ = size(idx)
    face_area = length(idx) ÷ size(idx, bcdim)
    backend = get_backend(idx)
    Ti = eltype(idx)

    flags = similar(idx, Ti, face_area)
    _face_flags_kernel!(backend)(flags, idx, nx, ny, bcdim, coord; ndrange=face_area)
    KernelAbstractions.synchronize(backend)
    scan = accumulate(+, flags)
    n = Int(Array(@view scan[end:end])[1])
    _free!(flags)

    nodes = similar(idx, Ti, n)
    if n > 0
        _face_fill_kernel!(backend)(
            nodes, scan, idx, nx, ny, bcdim, coord; ndrange=face_area,
        )
        KernelAbstractions.synchronize(backend)
    end
    _free!(scan)
    return nodes
end

"""
Build the compact inlet-edge map while the full pore-index grid is available.

Only boundary faces are retained, so permanent storage is O(N²) rather than
O(N³). The scans compact pore ordinals and valid pore-to-pore edges without
atomics and keep their order deterministic on every backend.
"""
function _build_inlet_flux(idx, D, bcdim, D0; checkpoint_readout=false)
    nx, ny, _ = size(idx)
    face_area = length(idx) ÷ size(idx, bcdim)
    backend = get_backend(idx)
    Ti = eltype(idx)

    flags = similar(idx, Ti, face_area)
    _inlet_flags_kernel!(backend)(flags, idx, nx, ny, bcdim; ndrange=face_area)
    KernelAbstractions.synchronize(backend)
    scan = accumulate(+, flags)
    n = Int(Array(@view scan[end:end])[1])
    _free!(flags)

    sources = checkpoint_readout ? similar(idx, Ti, n) : nothing
    targets = similar(idx, Ti, n)
    weights = similar(idx, typeof(D0), n)
    if n > 0
        _inlet_fill_kernel!(backend)(
            sources, targets, weights, scan, idx, D, nx, ny, bcdim, D0;
            ndrange=face_area,
        )
        KernelAbstractions.synchronize(backend)
    end
    _free!(scan)

    inlet = checkpoint_readout ? _build_face_nodes(idx, bcdim, 1) : nothing
    outlet = checkpoint_readout ?
        _build_face_nodes(idx, bcdim, size(idx, bcdim)) : nothing
    two_layer = size(idx, bcdim) == 2
    direct = two_layer && n > 0 ? sum(weights) : zero(D0)
    if two_layer && !checkpoint_readout
        _free!(targets)
        _free!(weights)
        targets = nothing
        weights = nothing
    end
    return InletFlux(sources, targets, weights, inlet, outlet, direct)
end

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
    _assembled_index_type(nnodes)

The integer type the CSC index arrays are stored in.

`Int32` halves the index traffic, which is what an unpreconditioned CG spends
nearly all of its time on: 16 B per stored entry becomes 12 B. That holds only
while every offset the assembly computes fits in 32 bits. A column holds at most
seven entries — its diagonal and six face neighbours — so `7 * nnodes` bounds
`nnz` from above, and the bound sits at 306,783,378 pore voxels. That is a pore
count rather than an edge length, so porosity decides which image reaches it:
roughly 1150³ at ε = 0.2, 800³ at ε = 0.6, 690³ at ε = 0.95. Measured
`nnz / nnodes` on blob images falls between 6.2 and 6.9 across ε = 0.3 to 0.95
and 128³ to 400³, so the bound sits within a tenth of the truth and counting the
entries exactly would buy almost nothing.

Past the bound every offset widens to `Int`, on host and device alike. The
alternative is not a slower path but a wrong one: `accumulate(+, counts)` and
`colptr` wrap to negative rather than saturating, `nnz_A` comes out wrong, and
`_steady_fill_kernel!` writes through `@inbounds` at the true offsets — off the
end of arrays sized from the wrapped count. On the host that corrupts memory; on
the device it faults with `ERROR_ILLEGAL_ADDRESS`, asynchronously and
uncatchably.

Widening doubles index storage exactly where device memory binds, and index
traffic is what the SpMV spends its time on: measured at 400³ on an RTX PRO
5000, one `Float32` SpMV takes 3.45 ms with `Int32` indices against 4.03 ms with
`Int64`, for 0.93 GiB of index storage against 1.85 GiB. At the bound the whole
matrix goes from 15.9 GB to 24.4 GB, against 8.6 GB for the matrix-free
operator, which also measured faster at equal iteration counts — 6.31 s against
8.33 s at 600³. So `matrixfree=true` is the recommendation at these sizes, and
only a recommendation: nothing routes there on its own.

Only the offsets widen. A pore ordinal is bounded by `nnodes` rather than by
`7 * nnodes` and carries its own type; see [`_ordinal_index_type`](@ref).
"""
_assembled_index_type(nnodes::Integer) = 7 * nnodes + 1 <= typemax(Int32) ? Int32 : Int

"""
    _ordinal_index_type(nnodes)

The integer type a pore ordinal is stored in: the element type of `idx`, and of
the row indices read out of it.

An ordinal is bounded by `nnodes` where an offset is bounded by `7 * nnodes`, so
the two walls sit a factor of seven apart and `idx` has no reason to widen when
the offsets do. `idx` spans the whole grid rather than the pore space, so holding
it at `Int32` past the offset wall is worth 4 B per voxel — 2.3 GB at 850³,
taken off the peak at the moment the matrix is live.

[`_operator_index_type`](@ref) applies this same bound to the matrix-free
operator, where the ordinal is the only index in play.
"""
_ordinal_index_type(nnodes::Integer) = nnodes + 1 <= typemax(Int32) ? Int32 : Int

"""
    _resolve_index_type(Ti, nnodes)

Apply the caller's `Ti` request against what [`_assembled_index_type`](@ref)
would have chosen, or error when the request cannot be honoured.

`nothing` takes the automatic choice. `Int64` is always honoured — asking for
more range than the image needs costs memory and nothing else. `Int32` is
honoured only while the bound holds: granting it past the bound would reinstate
the wrap-around corruption the bound exists to prevent, which is not something a
keyword argument gets to switch off.
"""
function _resolve_index_type(Ti, nnodes::Integer)
    isnothing(Ti) && return _assembled_index_type(nnodes)
    Ti === Int64 && return Int64
    Ti === Int32 || throw(ArgumentError(
        "`Ti` must be `Int32`, `Int64` or `nothing` (choose automatically); got $(Ti)"
    ))
    _assembled_index_type(nnodes) === Int32 || throw(ArgumentError(
        "`Ti=Int32` was requested for an image with $(nnodes) pore voxels, whose \
         assembled matrix needs more index range than 32 bits can address \
         ($(typemax(Int32))); pass `Ti=Int64`, or leave `Ti` unset to widen \
         automatically"
    ))
    return Int32
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
- `D`: diffusivity array matching `img`, a scalar for uniform diffusivity at that
  value, or `nothing` for uniform `D = 1`. A scalar takes the same path as
  `nothing` — no diffusivity array is read, `D0` carries the weight.
- `T`: element type of `b`. `A` follows `D`'s element type when one is given.
- `Ti`: index type of `A`, `Int32` or `Int64`. `nothing` (default) picks the
  narrowest that fits — see [`_assembled_index_type`](@ref). Force `Int64` to
  exercise the wide path below the size that needs it, or on a card with room to
  spare; force `Int32` to have the request checked against the bound rather than
  assumed.
"""
function build_steady_system(
    img; nnodes, axis, D=nothing, T=Float64, Ti=nothing, return_flux::Bool=false,
    checkpoint_readout::Bool=false,
)
    nx, ny, nz = size(img)
    bcdim = axis_dim(axis)
    nbc = size(img, bcdim)
    on_gpu = _on_gpu(img)
    Ti = _resolve_index_type(Ti, nnodes)
    # A scalar `D` is the uniform case, which the kernels already express as
    # `D === nothing` with `D0` carrying the weight — so it rides that path
    # rather than being expanded into a grid-sized array of one repeated value.
    D_scalar = D isa Number
    # `float` on the scalar's own type, so `D = 2` means the same thing as
    # `D = 2.0`. Taking `eltype(2)` literally would build the matrix over the
    # integers against a floating-point `b`, which Krylov warns about and the
    # preconditioner cannot construct at all.
    Tv = isnothing(D) ? T : (D_scalar ? float(typeof(D)) : eltype(D))
    D0 = D_scalar ? Tv(D) : one(Tv)
    # Handing the kernels `nothing` rather than the scalar is required, not
    # tidiness. `_node_diffusivity`'s array method reads `@inbounds D[i, j, k]`,
    # and `getindex` on a `Number` bounds-checks that every index is 1 — so a
    # scalar left in place appears to work only because `@inbounds` elides that
    # check, and throws `BoundsError` under `--check-bounds=yes`. It also takes
    # the harmonic mean of the value with itself at every face, which is a no-op
    # mathematically but not in floating point.
    D = D_scalar ? nothing : D

    # Pore numbering: an inclusive scan over the mask hands each pore voxel its
    # ordinal, and masking the solids back to zero lets `idx` double as the
    # pore/solid test — so neither kernel below ever reads `img`. Its element
    # type is the ordinal's, not the offsets': the two bounds differ by a factor
    # of seven, and this array spans the grid rather than the pore space.
    idx = similar(img, _ordinal_index_type(nnodes))
    _pore_index!(idx, img)
    backend = get_backend(idx)
    inlet_flux = return_flux ?
        _build_inlet_flux(idx, D, bcdim, D0; checkpoint_readout) : nothing
    # The backend-selected shape keeps the first dimension contiguous while
    # balancing occupancy against locality.
    wg = _steady_workgroup(idx)

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
    return return_flux ? (A, b, inlet_flux) : (A, b)
end
