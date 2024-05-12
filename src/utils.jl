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
    get_taufactor_conc(tau_solver; fill_value=NaN) -> Array{Float64, 3}

Extract the concentration field from a TauFactor solver object. The concentration field is
normalized to the range [0, 1] and NaN-filled where the conducting phase is absent.

# Arguments
- `tau_solver::taufactor.Solver`: TauFactor solver object

# Keywords
- `fill_value::Real=NaN`: Value to fill non-conducting phase voxels

# Returns
- `c::Array{Float64, 3}`: Concentration field
"""
function get_taufactor_conc(tau_solver; fill_value=NaN, normalize=true)
    # Get needed data from the solver object
    c = tau_solver.conc
    bcs = pyconvert(Array{Float64}, [tau_solver.bot_bc, tau_solver.top_bc])
    c_low, c_high = min(bcs...), max(bcs...)
    img = pyconvert(Array, tau_solver.cpu_img.squeeze())
    # NOTE: TauFactor always solves along the x-axis
    img_padded = pad(img, :replicate, (1, 0, 0))
    c = isa(c, Py) ? pyconvert(Array, tausolver.conc.squeeze().numpy()) : c
    # Hardcode BC values; taufactor doesn't update them
    c[1, :, :] .= 0.5
    c[end, :, :] .= -0.5
    # Remove padded voxels (in-plane), and ensure non-conducting phase is NaN-filled
    c = c[:, 2:(end - 1), 2:(end - 1)]
    c[.!img] .= fill_value
    c = normalize ? (c .- c_low) ./ (c_high - c_low) : c
    return c
end
