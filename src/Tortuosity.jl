module Tortuosity

using HDF5
using KernelAbstractions
using LinearAlgebra
using LinearSolve
using NaNStatistics
using SparseArrays
using OrdinaryDiffEqStabilizedRK
using OrdinaryDiffEqStabilizedRK: ROCK4, ODEProblem
using LsqFit
using PrecompileTools: @compile_workload

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

# Precompile a representative end-to-end workload so the first user-visible
# `solve` doesn't pay inference cost. Touches the steady linear path
# (KrylovJL_CG via LinearSolve), the transient ROCK4 path (with SavingCallback),
# the porous-media observables, and the LsqFit-based effective-diffusivity fit.
# Intentionally CPU-only and tiny (12³ image): the goal is type coverage, not
# correctness — accuracy is verified in the test suite. See issue #30.
@compile_workload begin
    # Precompile the imgen path with the same kwargs users pass in tutorials.
    Imaginator.blobs(shape=(12, 12, 12), porosity=0.65, blobiness=1.0, seed=1)

    # `ones(Bool, ...)` returns `Array{Bool,3}`, matching `Imaginator.blobs`'s
    # output type — `trues` would return a `BitArray{3}` and the steady-state
    # specializations would miss the user path entirely.
    img = ones(Bool, 12, 12, 12)

    sim = SteadyDiffusionProblem(img; axis=:x)
    sol = solve(sim.prob, KrylovJL_CG())
    c = reconstruct_field(sol.u, img)
    tortuosity(c, img; axis=:x)
    effective_diffusivity(c, img; axis=:x)
    formation_factor(c, img; axis=:x)

    prob = TransientDiffusionProblem(img; axis=:z, bc_inlet=1, bc_outlet=0)
    tsol = solve(prob, ROCK4(); saveat=0.1, tspan=(0.0, 0.2))
    flux(tsol.u, prob.D, prob.voxel_size, prob.img, prob.axis; ind=1, pore_index=prob.pore_index)
    mass_uptake(tsol.u, prob)
    slice_concentration(tsol.u, prob.img, prob.axis, 1; pore_index=prob.pore_index, pore_only=true)

    fit_effective_diffusivity(tsol, prob, :mass)
end

end  # module Tortuosity
