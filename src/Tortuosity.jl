module Tortuosity

using CUDA
using CUDA.CUSPARSE
using HDF5
using LinearAlgebra
using LinearSolve
using NaNStatistics
using SparseArrays
using OrdinaryDiffEq
using LsqFit

include("utils.jl")
include("geometry.jl")
include("imgen.jl")
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
export TortuositySimulation
export TransientProblem
export TransientState

# Steady-state analysis
export solve
export tortuosity
export effective_diffusivity
export formation_factor
export vec_to_grid

# Transient solver
export solve!
export init_state
export stop_at_time
export stop_at_avg_concentration
export stop_at_delta_flux
export stop_at_periodic

# Transient measurements
export compute_flux
export get_slice_conc
export compute_mass_uptake

# Transient fitting
export fit_effective_diffusivity
export fit_voxel_diffusivity

# Analytical solutions
export slab_concentration
export slab_mass_uptake
export slab_flux
export slab_cumulative_flux

end  # module Tortuosity
