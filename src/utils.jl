using HDF5
using Plots


function read_linear_sys(path; sparse_fmt)
    @assert sparse_fmt in ["csc", "coo"]
    fid = h5open(path, "r")

    # Load common attributes
    shape = Tuple(read(fid["shape"]))
    template = Bool.(read(fid["template"]))
    rhs = read(fid["rhs"])
    nzval = read(fid["nzval"])

    # Load matrix-specific attributes
    if sparse_fmt == "csc"
        # NOTE: Convert to 1-based indexing
        colptr = read(fid["colptr"]) .+ 1
        rowval = read(fid["rowval"]) .+ 1
        spmat_args = (colptr, rowval, nzval, shape)
    elseif sparse_fmt == "coo"
        row = read(fid["row"]) .+ 1
        col = read(fid["col"]) .+ 1
        spmat_args = (row, col, nzval, shape)
    end

    return spmat_args, rhs, template
end


function compute_effective_diffusivity(scalar_field, axis)
    Δc = 1.0
    axis_idx = Dict(:x => 1, :y => 2, :z => 3)[axis]
    L = size(scalar_field)[axis_idx]
    A = prod(size(scalar_field)) / L

    # Extract slices based on the specified axis
    function slice_at_dim(dim, index)
        return ntuple(i -> i == dim ? index : :, 3)
    end

    first_slice = scalar_field[slice_at_dim(axis_idx, 1)...]
    second_slice = scalar_field[slice_at_dim(axis_idx, 2)...]
    rate = nansum(first_slice - second_slice)
    Deff = rate * (L-1) / A / Δc
    return Deff
end


function compute_formation_factor(c, axis)
    Deff = compute_effective_diffusivity(c, axis)
    return 1 / Deff
end


function compute_tortuosity_factor(c, axis)
    # !: Assumes that c is NaN-filled outside the pore space
    ε = sum(isfinite.(c)) / prod(size(c))
    Deff = compute_effective_diffusivity(c, axis)
    return ε / Deff
end


function parse_args(s::String)
    # Parse command-line arguments in the form of "--key=value"
    regex = r"--(\w+)=([^\s]+)"
    matches = eachmatch(regex, s)
    pairs = Dict(m.captures[1] => m.captures[2] for m in matches)
    return pairs
end
