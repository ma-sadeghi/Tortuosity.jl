# Hooks for Tortuosity's optional dependencies. Each stub names the package the
# caller has to load; the matching extension under `ext/` supplies a more
# specific method that does the real work as soon as that package is in the
# session. Keeping the stubs here means the public entry points stay defined
# whether or not the optional package is present, and fail with an actionable
# message rather than an `UndefVarError` when it is not.

function _weakdep_error(pkg::AbstractString, what::AbstractString)
    msg = "$what requires $pkg, which Tortuosity does not load itself. Run `using $pkg` first (`Pkg.add(\"$pkg\")` if it is not installed)."
    return error(msg)
end

# HDF5 — see `export_to_hdf5` in utils.jl
_h5open(args...; kwargs...) = _weakdep_error("HDF5", "export_to_hdf5")
