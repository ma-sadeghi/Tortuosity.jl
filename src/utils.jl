"""
    args_to_dict(args)

Parse command-line arguments of the form `--key=value` into a `Dict{String,String}`.
"""
function args_to_dict(args)
    s = join(args, " ")
    # Parse command-line arguments in the form of "--key=value"
    regex = r"--(\w+)=([^\s]+)"
    matches = eachmatch(regex, s)
    pairs = Dict(m.captures[1] => m.captures[2] for m in matches)
    return pairs
end

"""
    format_args_dict(args_dict) -> (fpath, path_export, gpu_id)

Extract and parse standard CLI arguments from a dict returned by [`args_to_dict`](@ref).
Returns `(fpath::String, path_export::String, gpu_id::Int)`.
"""
function format_args_dict(args_dict)
    fpath = args_dict["fpath"]
    path_export = args_dict["path_export"]
    gpu_id = parse(Int, args_dict["gpu_id"])
    return fpath, path_export, gpu_id
end

"""
    export_to_hdf5(fname; kwargs...)

Write keyword arguments as datasets to an HDF5 file. Each keyword becomes a
dataset named after the keyword.

Requires `HDF5.jl`, an optional dependency that does not install with Tortuosity:
run `Pkg.add("HDF5")`, then `using HDF5`, before calling this.
"""
function export_to_hdf5(fname; kwargs...)
    _h5open(fname, "w") do fid
        for (name, value) in pairs(kwargs)
            fid[String(name)] = value
        end
    end
end

"""
    reconstruct_field(u, img::AbstractArray{Bool})

Expand a pore-only solution vector `u` into a full-sized array matching `img`.
Pore voxels receive values from `u`; solid voxels are filled with `NaN`.
The element type of the output matches `eltype(u)`.
"""
function reconstruct_field(u, img::AbstractArray{Bool})
    @assert length(u) == count(img) "Length of u must match the number of true voxels in img"
    # Logical-indexing a CPU Array with a GPU Bool mask triggers scalar
    # iteration, so pull img to CPU when it isn't already there.
    # `_on_gpu` is defined on the bare device array types only, so a device mask
    # behind a wrapper — a strided `SubArray`, a `PermutedDimsArray` — reports
    # false and would reach the logical index below still on the device. Test for
    # the host types that can be indexed directly instead, and copy anything else.
    img_cpu = img isa Union{Array,BitArray} ? img : Array(img)
    T = eltype(u)
    c = fill(T(NaN), size(img_cpu))
    c[img_cpu] = Array(u)
    return c
end

"""
    build_pore_index(img::BitArray)

Build a 3D `Array{Int}` the same shape as `img` mapping each voxel to its index
in the flat pore-only vector (ordered column-major over pore voxels). Pore
voxels store their 1-based pore-vector position; solid voxels store `0` as a
"not-a-pore" sentinel.

Solid voxels use `0` rather than `NaN` because the return type is `Array{Int}`
(for O(1) indexing into the pore vector) and `Int` has no NaN. Since pore
indices are 1-based, `0` is unambiguously invalid.

Used internally by [`slice_indices`](@ref) and [`reconstruct_slice`](@ref) to
avoid walking the full image on every slice operation.
"""
function build_pore_index(img::BitArray)
    g = zeros(Int, size(img))
    g[img] = 1:count(img)
    return g
end

const _PORE_INDEX_MIN_THREADED = 1_000_000

function _pore_index!(idx, img)
    cumsum!(vec(idx), vec(img))
    idx .*= img
    return idx
end

function _pore_index!(idx::Array{Ti}, img::Union{Array{Bool},BitArray}) where {Ti<:Integer}
    n = length(img)
    if Threads.nthreads() == 1 || n < _PORE_INDEX_MIN_THREADED
        return invoke(_pore_index!, Tuple{Any,Any}, idx, img)
    end

    nchunks = min(n, 4 * Threads.nthreads())
    chunk_size = cld(n, nchunks)
    offsets = zeros(Ti, nchunks)
    Threads.@threads :dynamic for chunk in 1:nchunks
        ilo = (chunk - 1) * chunk_size + 1
        ihi = min(ilo + chunk_size - 1, n)
        total = zero(Ti)
        @inbounds for i in ilo:ihi
            total += img[i]
            idx[i] = total
        end
        offsets[chunk] = total
    end

    running_offset = zero(Ti)
    @inbounds for chunk in eachindex(offsets)
        total = offsets[chunk]
        offsets[chunk] = running_offset
        running_offset += total
    end
    Threads.@threads :dynamic for chunk in 1:nchunks
        ilo = (chunk - 1) * chunk_size + 1
        ihi = min(ilo + chunk_size - 1, n)
        chunk_offset = offsets[chunk]
        @inbounds @simd for i in ilo:ihi
            idx[i] = img[i] ? idx[i] + chunk_offset : zero(Ti)
        end
    end
    return idx
end

"""
    find_true_indices(a::AbstractArray{Bool})

Return the linear indices of all `true` elements in `a` as a `Vector{Int}`.
"""
function find_true_indices(a::AbstractArray{Bool})
    j = 0
    indices = Vector{Int}(undef, count(a))
    @inbounds for i in eachindex(a)
        @inbounds if a[i]
            j += 1
            indices[j] = i
        end
    end
    return indices
end

"""
    build_reverse_lookup(img::AbstractArray{Bool})

Build a `Dict` mapping each `true`-element's linear index in `img` to its
sequential pore-voxel number (1, 2, …, `count(img)`).
"""
function build_reverse_lookup(img::AbstractArray{Bool})
    return Dict(zip(find_true_indices(img), 1:count(img)))
end
