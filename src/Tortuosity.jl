module Tortuosity

using CUDA
using HDF5
using LinearSolve
using NaNStatistics
using SparseArrays

include("utils.jl")
include("imgen.jl")
include("dnstools.jl")
include("topotools.jl")
include("pdetools.jl")
include("simulations.jl")

# Submodules
export Imaginator

# Structs
export TortuositySimulation

# Functions
export vec_to_field
export effective_diffusivity, formation_factor, tortuosity
export phase_fraction

end  # module Tortuosity
