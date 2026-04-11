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
"""
function export_to_hdf5(fname; kwargs...)
    h5open(fname, "w") do fid
        for (name, value) in pairs(kwargs)
            fid[String(name)] = value
        end
    end
end

"""
    vec_to_grid(u, img::AbstractArray{Bool})

Expand a pore-only solution vector `u` into a full-sized array matching `img`.
Pore voxels receive values from `u`; solid voxels are filled with `NaN`.
The element type of the output matches `eltype(u)`.
"""
function vec_to_grid(u, img::AbstractArray{Bool})
    @assert length(u) == count(img) "Length of u must match the number of true voxels in img"
    # Logical-indexing a CPU Array with a GPU Bool mask triggers scalar
    # iteration, so pull img to CPU when it isn't already there.
    img_cpu = img isa Array ? img : Array(img)
    T = eltype(u)
    c = fill(T(NaN), size(img_cpu))
    c[img_cpu] = Array(u)
    return c
end

"""
    grid_to_vec(img::BitArray)

Build a lookup array mapping each pore voxel in `img` to its 1D index in the
pore-only vector. Solid voxels are mapped to `0`. This is the inverse of
[`vec_to_grid`](@ref).
"""
function grid_to_vec(img::BitArray)
    g = zeros(Int, size(img))
    g[img] = 1:count(img)
    return g
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
    reverse_lookup(im::AbstractArray{Bool})

Build a `Dict` mapping each `true`-element's linear index in `im` to its
sequential pore-voxel number (1, 2, …, `count(im)`).
"""
function reverse_lookup(im::AbstractArray{Bool})
    return Dict(zip(find_true_indices(im), 1:count(im)))
    # sparsevec(find_true_indices(im), 1:count(im))
end

"""
    get_taufactor_conc(tau_solver; fill_value=NaN, normalize=true)

Extract the concentration field from a TauFactor (Python) solver object.
Pads are removed and non-conducting voxels are filled with `fill_value`.

# Arguments
- `tau_solver`: a `taufactor.Solver` object (via PythonCall).

# Keyword Arguments
- `fill_value`: value for non-conducting voxels. Default: `NaN`.
- `normalize`: if `true`, rescale concentrations to `[0, 1]` using the
  solver's boundary conditions. Default: `true`.

# Returns
- `Array{Float64, 3}`: the 3D concentration field.
"""
function get_taufactor_conc(tau_solver; fill_value=NaN, normalize=true)
    # Get needed data from the solver object
    c = tau_solver.conc
    bcs = pyconvert(Array{Float64}, [tau_solver.bot_bc, tau_solver.top_bc])
    c_low, c_high = min(bcs...), max(bcs...)
    img = pyconvert(Array, tau_solver.cpu_img.squeeze())
    # NOTE: TauFactor always solves along the x-axis
    img_padded = pad(img, :replicate, (1, 0, 0))
    c = isa(c, Py) ? pyconvert(Array, tau_solver.conc.squeeze().numpy()) : c
    # Hardcode BC values; taufactor doesn't update them
    c[1, :, :] .= 0.5
    c[end, :, :] .= -0.5
    # Remove padded voxels (in-plane), and ensure non-conducting phase is NaN-filled
    c = c[:, 2:(end - 1), 2:(end - 1)]
    c[.!img] .= fill_value
    c = normalize ? (c .- c_low) ./ (c_high - c_low) : c
    return c
end
