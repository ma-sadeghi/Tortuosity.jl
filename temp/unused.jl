using HDF5


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


# NOTE: This function is 3x faster and 2x less allocating than the one in Tortuosity.jl.
# (In fact, it's non-allocating, other than the output allocation, which is unavoidable.)
# There's a catch, though: the output conns is noncontiguous and needs to be relabeled.
function create_adjacency_list(template; kind=:triu)
    nx, ny, nz = size(template)

    # Preallocate the adjacency list with the maximum possible number of edges
    num_edges_max = count(template) * 3 * (kind == :sym ? 2 : 1)
    adjacency_list = Matrix{Int}(undef, num_edges_max, 2)
    s2i = LinearIndices(template)

    neighbors = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    counter = 0

    # Iterate through the 3D grid
    for x = 1:nx, y = 1:ny, z = 1:nz
        # If the current node is not valid (is a hole), skip to the next iteration
        if !template[x, y, z]
            continue
        end
        # i = n2c[s2i[x, y, z]]
        i = s2i[x, y, z]
        # Check all possible neighbors
        for (dx, dy, dz) in neighbors
            xn, yn, zn = x + dx, y + dy, z + dz
            # Check if the neighbor coordinates are valid and not a hole
            if xn ≥ 1 && xn ≤ nx && yn ≥ 1 && yn ≤ ny && zn ≥ 1 && zn ≤ nz && template[xn, yn, zn]
                # j = n2c[s2i[xn, yn, zn]]
                j = s2i[xn, yn, zn]
                if kind == :sym || (i < j && kind == :triu)
                    counter += 1
                    adjacency_list[counter, :] .= i, j
                end
            end
        end
    end

    # Trim the adjacency_list to remove unused preallocated space
    return @view adjacency_list[1:counter, :]
end
