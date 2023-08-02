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


nothing
