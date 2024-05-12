function isin_L(a, b)
    b = Set(b)
    return [x in b for x in a]
end

function overlap_indices_L(a, b)
    return findall(isin_L(a, b))
end

function overlap_indices(a, b::Set)
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
    generate_chunks(n, divisions)

Returns an array of tuples where each tuple represents
the start and end indices of a chunk when dividing an
array of length `n` into `divisions` number of chunks.
"""
function find_chunk_bounds(; nelems, ndivs)
    chunk_size = ceil(Int, nelems / ndivs)
    chunks = [(i, min(i + chunk_size - 1, nelems)) for i in 1:chunk_size:nelems]
    return chunks
end

function overlap_indices🚀(a, b::Set)
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

function overlap_indices🚀(a, b::AbstractArray)
    return overlap_indices🚀(a, Set(b))
end

function multihotvec(indices, n; vals=1.0)
    vec = zeros(n)
    vec[indices] .= vals
    return vec
end
