# Changelog

All notable changes to Tortuosity.jl are recorded here. Versions follow [Julia's SemVer rules](https://pkgdocs.julialang.org/v1/compatibility/), under which a change to the leftmost non-zero version component is breaking.

## Unreleased

### Breaking

Three dependencies moved into package extensions, so they are no longer installed or loaded by `using Tortuosity`. `Pkg.add` them yourself, then load them before calling the entry points that need them. This is a breaking interface change for callers of those entry points.

| Add and load this | Before calling |
|-------------------|----------------|
| `ImageFiltering` | `Imaginator.blobs`, `Imaginator.apply_gaussian_blur` |
| `LsqFit`         | `fit_effective_diffusivity`, `fit_voxel_diffusivity` |
| `HDF5`           | `Tortuosity.export_to_hdf5` |

Calling one of them without its package raises an error naming the package to load, rather than an `UndefVarError`. Everything else — problem construction, solving, the observables, `Imaginator.trim_nonpercolating_paths` — is unaffected.

### Performance

- `using Tortuosity` loads 151 packages instead of 212, and takes about 3.5 s instead of about 4.6 s (Julia 1.12.6, Windows 11, warm cache).
- The precompile workloads that cover the moved entry points moved with them into their extensions, so first-call latency is unchanged for anyone who loads those packages.

### Fixed

- `set_preferences!(Tortuosity, "precompile_workload" => false)` now also disables the GPU extensions' precompile workloads. `PrecompileTools` resolved that preference against the extension module, whose UUID `set_preferences!` refuses, so the GPU workload kept running. This is a development-time switch only; the workloads remain enabled by default.
