using HDF5


function args_to_dict(args)
    s = join(args, " ")
    # Parse command-line arguments in the form of "--key=value"
    regex = r"--(\w+)=([^\s]+)"
    matches = eachmatch(regex, s)
    pairs = Dict(m.captures[1] => m.captures[2] for m in matches)
    return pairs
end


function format_args_dict(args_dict)
    fpath = args_dict["fpath"]
    path_export = args_dict["path_export"]
    gpu_id = parse(Int, args_dict["gpu_id"])
    return fpath, path_export, gpu_id
end


function export_to_hdf5(fname; kwargs...)
    h5open(fname, "w") do fid
        for (name, value) in pairs(kwargs)
            fid[String(name)] = value
        end
    end
end


function vec_to_field(u, im)
    c = fill(NaN, size(im))
    c[im] = Array(u)
    return c
end


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


function reverse_lookup(im::AbstractArray{Bool})
    Dict(zip(find_true_indices(im), 1:count(im)))
    # sparsevec(find_true_indices(im), 1:count(im))
end


"""
    slice_at_dim(dim, index)

Returns a tuple of `:`s and `index` at the specified dimension `dim`.
"""
function slice_at_dim(dim, index)
    return ntuple(i -> i == dim ? index : :, 3)
end
