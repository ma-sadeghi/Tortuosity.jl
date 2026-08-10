# Optional dependency: routes Tortuosity's Gaussian-blur hook to ImageFiltering.jl.
module TortuosityImageFilteringExt

using ImageFiltering
using Tortuosity
using PrecompileTools: @setup_workload, @compile_workload

function Tortuosity._gaussian_blur(img, sigma)
    sigma = tuple(fill(sigma, ndims(img))...)
    kernel = Kernel.gaussian(sigma)
    return imfilter(img, kernel, "symmetric")
end

# Blob generation used to sit in the CPU workload in src/Tortuosity.jl; it moved
# here with ImageFiltering, so users who do load ImageFiltering still get a
# compiled first call. `Tortuosity._workload_enabled()` rather than the
# extension's own preference: `set_preferences!` refuses an extension UUID, so
# only the parent package's `precompile_workload` preference can switch this
# off. On by default.
@setup_workload begin
    if Tortuosity._workload_enabled()
        @compile_workload begin
            Tortuosity.Imaginator.blobs(
                shape=(12, 12, 12), porosity=0.65, blobiness=1.0, seed=1
            )
        end
    end
end

end
