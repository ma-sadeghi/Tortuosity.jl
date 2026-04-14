module Tortuosity

using HDF5
using KernelAbstractions
using LinearAlgebra
using LinearSolve
using NaNStatistics
using SparseArrays
using OrdinaryDiffEq
using OrdinaryDiffEq: ROCK4
using LsqFit

# GPU backend registration (populated by package extensions)
const _preferred_gpu_backend = Ref{Any}(nothing)
const _gpu_adapt = Ref{Any}(identity)

"""True for GPU arrays; extensions override for CuArray, MtlArray, etc."""
_on_gpu(::AbstractArray) = false

include("utils.jl")
include("geometry.jl")
include("imgen.jl")
include("sparse_type.jl")
include("kernels/graph.jl")
include("kernels/sparse.jl")
include("topotools.jl")
include("numpytools.jl")
include("pdetools.jl")
include("simulations.jl")
include("transient.jl")
include("transient_measurements.jl")
include("transient_fitting.jl")
include("dnstools.jl")
include("caverns.jl")

# Submodules
export Imaginator
export KrylovJL_CG
export ROCK4

# Structs
export SteadyDiffusionProblem
export TransientDiffusionProblem

# Steady-state analysis
export solve
export solve!
export tortuosity
export effective_diffusivity
export formation_factor
export reconstruct_field

# Transient solver + stop conditions
export StopAtSteadyState
export StopAtFluxBalance
export StopAtSaturation
export StopAtPeriodicState

# Transient measurements
export flux
export slice_concentration
export mass_uptake

# Transient fitting
export fit_effective_diffusivity
export fit_voxel_diffusivity

# Analytical solutions
export slab_concentration
export slab_mass_uptake
export slab_flux
export slab_cumulative_flux

end  # module Tortuosity
