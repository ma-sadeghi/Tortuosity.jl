module Tortuosity

include("dnstools.jl")
include("imgen.jl")
include("simulations.jl")
include("utils.jl")

# Submodules
export Imaginator

# Structs
export TortuositySimulation

# Functions
export vec_to_field
export effective_diffusivity, formation_factor, tortuosity
export phase_fraction
export create_connectivity_list🚀, create_adjacency_matrix, laplacian
export find_boundary_nodes
export apply_dirichlet_bc🚀!

end  # module Tortuosity
