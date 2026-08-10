# Hooks for Tortuosity's optional dependencies. Each stub names the package the
# caller has to load; the matching extension under `ext/` supplies a more
# specific method that does the real work as soon as that package is in the
# session. Keeping the stubs here means the public entry points stay defined
# whether or not the optional package is present, and fail with an actionable
# message rather than an `UndefVarError` when it is not.

#
# The stubs take positional arguments only, deliberately. Every extension method
# has fixed positional arity and no keywords, so a stub that also swallowed
# keywords would become the best match the moment a call site grew one — and
# would then tell a user with the package already loaded to go load it. Without
# the keyword catch-all that same slip is an honest `MethodError` naming the
# argument that does not fit.

function _weakdep_error(pkg::AbstractString, what::AbstractString)
    msg = "$pkg is not loaded. Tortuosity keeps it optional, so run `using $pkg` before calling $what (`Pkg.add(\"$pkg\")` if it is not installed)."
    return error(msg)
end

# HDF5 — see `export_to_hdf5` in utils.jl
_h5open(args...) = _weakdep_error("HDF5", "export_to_hdf5")

# LsqFit — see `fit_effective_diffusivity` and `fit_voxel_diffusivity` in
# transient_fitting.jl
const _FITTERS = "fit_effective_diffusivity or fit_voxel_diffusivity"
_curve_fit(args...) = _weakdep_error("LsqFit", _FITTERS)
_stderror(args...) = _weakdep_error("LsqFit", _FITTERS)

# ImageFiltering — see `Imaginator.apply_gaussian_blur` in imgen.jl
_gaussian_blur(args...) = _weakdep_error(
    "ImageFiltering", "Imaginator.blobs or Imaginator.apply_gaussian_blur"
)
