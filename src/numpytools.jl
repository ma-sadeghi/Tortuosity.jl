function isin_L(a, b)
    b = Set(b)
    [x in b for x in a]
end


function overlap_indices_L(a, b)
    findall(isin_L(a, b))
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
function get_chunks_bounds(n, divisions)
    chunk_size = ceil(Int, n / divisions)
    chunks = [(i, min(i + chunk_size - 1, n)) for i in 1:chunk_size:n]
    return chunks
end


function overlap_indices🚀(a, b::Set)
    n = length(a)
    num_threads = Threads.nthreads()

    # Get chunks based on number of threads
    bounds = get_chunks_bounds(n, num_threads)
    # Preallocate a list to store each thread's indices
    thread_indices = Vector{Vector{Int}}(undef, num_threads)

    Threads.@threads for tid in 1:num_threads
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
