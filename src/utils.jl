using HDF5
using Plots


function set_plotsjl_defaults()
    default()
    default(size=(600, 400), thickness_scaling=1.25)
    scalefontsizes(1.25)
end


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


function calc_effective_diffusivity(c, template)
    Δc = 1.0
    L = size(c)[1]
    A = prod(size(c)) / L
    rate = nansum(c[1, :, :] - c[2, :, :])
    Deff = rate * (L-1) / A / Δc
    return Deff
end


function calc_formation_factor(c, template)
    Deff = calc_effective_diffusivity(c, template)
    return 1 / Deff    
end


function calc_tortuosity(c, template)
    ε = sum(template) / length(template)
    Deff = calc_effective_diffusivity(c, template)
    return ε / Deff
end


function parse_args(s::String)
    # Parse command-line arguments in the form of "--key=value"
    regex = r"--(\w+)=([^\s]+)"    
    matches = eachmatch(regex, s)
    pairs = Dict(m.captures[1] => m.captures[2] for m in matches)
    return pairs
end


nothing
