"""
    atleast_3d(x)

Reshape `x` to at least 3 dimensions by appending singleton dimensions.
0D → `(1,1,1)`, 1D → `(n,1,1)`, 2D → `(m,n,1)`, ≥3D → unchanged.
"""
function atleast_3d(x)
    if ndims(x) == 0
        reshape([x], 1, 1, 1)
    elseif ndims(x) == 1
        reshape(x, length(x), 1, 1)
    elseif ndims(x) == 2
        reshape(x, size(x)..., 1)
    else
        x  # Already 3D or higher
    end
end

"""
    isin_slow(a::AbstractArray, b::AbstractArray)

Reference implementation of element-wise membership test. Returns a boolean
vector indicating which elements of `a` are present in `b`. Simple and correct
but O(n×m). Kept as a readable baseline for verifying optimized versions.
Similar to `numpy.isin`.

# Arguments
- `a`: array whose elements are tested for membership.
- `b`: set of values to test against.

# Returns
- `Vector{Bool}`: `true` at index `i` if `a[i] ∈ b`.
"""
function isin_slow(a::AbstractArray, b::AbstractArray)
    b = Set(b)
    return [x in b for x in a]
end

"""
    overlap_indices_slow(a::AbstractArray, b::AbstractArray)
    overlap_indices_slow(a::AbstractArray, b::Set)

Reference implementation of `overlap_indices` using `isin_slow`. Simple
and correct but slower than `overlap_indices`. Kept as a readable
baseline for verifying optimized versions.
"""
function overlap_indices_slow(a::AbstractArray, b::AbstractArray)
    return findall(isin_slow(a, b))
end

"""
    overlap_indices(a::AbstractArray, b::AbstractArray)
    overlap_indices(a::AbstractArray, b::Set)

Returns the indices of elements in `a` that are also in `b`, sorted
in ascending order, not the order in which they appear in `a`.

# Arguments
- `a::AbstractArray`: The array to search for overlapping elements.
- `b::AbstractArray`: The array to search for in `a`.

# Returns
- `Vector{Int}`: The indices of elements in `a` that are also in `b`.

# Example
```jldoctest; setup = :(using Tortuosity: overlap_indices)
julia> overlap_indices([1, 2, 3, 4, 5], [3, 4, 1])
3-element Vector{Int64}:
 1
 3
 4
```
"""
function overlap_indices(a::AbstractArray, b::Set)
    indices = Int[]
    @inbounds for i in eachindex(a)
        if a[i] in b
            push!(indices, i)
        end
    end
    return indices
end

function overlap_indices(a, b::AbstractArray)
    return overlap_indices(a, Set(b))
end

"""
    overlap_indices_fast(a::AbstractArray, b::AbstractArray)
    overlap_indices_fast(a::AbstractArray, b::Set)

Parallelized version of `overlap_indices` using `Threads.@threads`.
"""
function overlap_indices_fast(a::AbstractArray, b::Set)
    num_threads = Threads.nthreads()
    # Chunk `a` into `num_threads` number of chunks (might be less)
    bounds = find_chunk_bounds(; nelems=length(a), ndivs=num_threads)
    # Preallocate a list to store each thread's indices
    thread_indices = Vector{Vector{Int}}(undef, length(bounds))

    Threads.@threads for tid in eachindex(bounds)
        (start_idx, end_idx) = bounds[tid]
        local_indices = Int[]

        @inbounds for i in start_idx:end_idx
            if a[i] in b
                push!(local_indices, i)
            end
        end

        thread_indices[tid] = local_indices
    end

    return vcat(thread_indices...)
end

function overlap_indices_fast(a::AbstractArray, b::AbstractArray)
    return overlap_indices_fast(a, Set(b))
end

"""
    find_chunk_bounds(; nelems::Int, ndivs::Int)

Returns an array of tuples where each tuple represents the start and end
indices of a chunk when dividing an array of length `nelems` into `ndivs`
number of chunks.

# Arguments
- `nelems::Int`: The number of elements in the array to divide.
- `ndivs::Int`: The number of divisions to make.

# Returns
- `Vector{Tuple{Int, Int}}`: An array of tuples representing the start and end indices of each chunk.

# Example
```jldoctest; setup = :(using Tortuosity: find_chunk_bounds)
julia> find_chunk_bounds(; nelems=10, ndivs=3)
3-element Vector{Tuple{Int64, Int64}}:
 (1, 4)
 (5, 8)
 (9, 10)
```
"""
function find_chunk_bounds(; nelems::Int, ndivs::Int)
    chunk_size = ceil(Int, nelems / ndivs)
    chunks = [(i, min(i + chunk_size - 1, nelems)) for i in 1:chunk_size:nelems]
    return chunks
end

"""
    multihotvec(indices::AbstractArray, n::Int; vals=1.0, template=nothing)

Build a length-`n` vector whose entries at positions `indices` are `vals`,
and which is zero everywhere else. If `vals` is a scalar, every flagged
position receives that scalar; if `vals` is an array, `length(indices)` must
equal `length(vals)` and each index gets the corresponding element.

`template` exists to support GPU backends: when supplied, the output vector
is allocated on the same device (and with element type matching `vals`),
so downstream GPU kernels can broadcast against it without an implicit
host→device copy. When `template === nothing` the output is a plain
`Vector{eltype(vals)}`.

# Arguments
- `indices::AbstractArray`: positions to set.
- `n::Int`: length of the output vector (must be `≥ maximum(indices)`).

# Keyword Arguments
- `vals`: scalar or array of values to place at `indices`. Default: `1.0`.
- `template`: optional array whose storage type (CPU/GPU) and shape API is
  used to allocate the output via `similar(template, eltype, n)`. Default:
  `nothing` (allocate on CPU).

# Example
```jldoctest; setup = :(using Tortuosity: multihotvec)
julia> multihotvec([1, 3, 4], 6)
6-element Vector{Float64}:
 1.0
 0.0
 1.0
 1.0
 0.0
 0.0

julia> multihotvec([1, 3, 4], 6; vals=[0.1, 0.3, 0.2])
6-element Vector{Float64}:
 0.1
 0.0
 0.3
 0.2
 0.0
 0.0
```
"""
function multihotvec(indices::AbstractArray, n::Int; vals=1.0, template=nothing)
    if vals isa AbstractArray
        @assert length(indices) == length(vals) "indices and vals must have the same length"
    end
    @assert n >= maximum(indices) "n must be >= max(indices)"
    T = vals isa AbstractArray ? eltype(vals) : typeof(vals)
    vec = isnothing(template) ? zeros(T, n) : fill!(similar(template, T, n), zero(T))
    vec[indices] .= vals
    return vec
end
