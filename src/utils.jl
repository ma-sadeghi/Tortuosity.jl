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
