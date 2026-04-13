module Tortuosity

using HDF5
using KernelAbstractions
using LinearAlgebra
using LinearSolve
using NaNStatistics
using SparseArrays
using OrdinaryDiffEq
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

# Structs
export SteadyDiffusionProblem
export TransientDiffusionProblem
export TransientState

# Steady-state analysis
export solve
export tortuosity
export effective_diffusivity
export formation_factor
export reconstruct_field

# Transient solver
export solve!
export init_state
export stop_at_time
export stop_at_avg_concentration
export stop_at_flux_balance
export stop_at_periodic

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
