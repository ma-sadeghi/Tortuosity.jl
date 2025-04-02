"""
    isin👎(a::AbstractArray, b::AbstractArray)

Returns a boolean array where each element of `a` is checked
for membership in `b`. Similar to `numpy.isin`.

# Arguments
- `a::AbstractArray`: The array to check for membership.
- `b::AbstractArray`: The array to check for membership in `a`.

# Returns
- `Vector{Bool}`: A boolean array for `a` elements checked for membership in `b`.

# Example
```jldoctest
julia> a = [1, 2, 3, 4, 5]
julia> b = [3, 4, 1]
julia> isin👎(a, b)
5-element Vector{Bool}:
 1
 0
 1
 1
 0
```
"""
function isin👎(a::AbstractArray, b::AbstractArray)
    b = Set(b)
    return [x in b for x in a]
end

"""
    overlap_indices👎(a::AbstractArray, b::AbstractArray)
    overlap_indices👎(a::AbstractArray, b::Set)

Similar to `overlap_indices`, but uses legacy internal functions.
"""
function overlap_indices👎(a::AbstractArray, b::AbstractArray)
    return findall(isin👎(a, b))
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
```jldoctest
julia> a = [1, 2, 3, 4, 5]
julia> b = [3, 4, 1]
julia> overlap_indices👎(a, b)
3-element Vector{Int}:
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
    overlap_indices🚀(a::AbstractArray, b::AbstractArray)
    overlap_indices🚀(a::AbstractArray, b::Set)

Similar to `overlap_indices`, but parallelized using `Threads.@threads`.
"""
function overlap_indices🚀(a::AbstractArray, b::Set)
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

function overlap_indices🚀(a::AbstractArray, b::AbstractArray)
    return overlap_indices🚀(a, Set(b))
end

"""
    find_chunk_bounds(; nelems::Int, ndivs::Int)

Returns an array of tuples where each tuple represents the start and end
indices of a chunk when dividing an array of length `n` into `divisions`
number of chunks.

# Arguments
- `nelems::Int`: The number of elements in the array to divide.
- `ndivs::Int`: The number of divisions to make.

# Returns
- `Vector{Tuple{Int, Int}}`: An array of tuples representing the start and end indices of each chunk.

# Example
```jldoctest
julia> find_chunk_bounds(nelems=10, ndivs=3)
3-element Vector{Tuple{Int, Int}}:
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
    multihotvec(indices::AbstractArray, n::Int; vals::Number=1.0)
    multihotvec(indices::AbstractArray, n::Int; vals::AbstractArray)

Returns a vector of length `n` with `vals` at the indices specified
in `indices`.

# Arguments
- `indices::AbstractArray`: The indices to set to `vals`.
- `n::Int`: The length of the vector.
- `vals::Number=1.0`: The value to set at the indices.
- `vals::AbstractArray`: The values to set at the indices.

# Returns
- `Vector{Number}`: A vector of length `n` with `vals` at the specified indices.

# Example
```jldoctest
julia> multihotvec([1, 3, 4], 6)
6-element Vector{Float64}:
 1.0
 0.0
 1.0
 1.0
 0.0
 0.0
julia> multihotvec([1, 3, 4], 6, vals=[0.1, 0.3, 0.2])
6-element Vector{Float64}:
 0.1
 0.0
 0.3
 0.2
 0.0
 0.0
```
"""
function multihotvec(indices, n; vals=1.0, gpu=false)
    vals isa Array ? (gpu ? (@assert vals isa CuArray "vals must be a CuArray") : nothing) : nothing
    vec = gpu ? CUDA.zeros(n) : zeros(n)
    vec[indices] .= vals
    return vec
end
