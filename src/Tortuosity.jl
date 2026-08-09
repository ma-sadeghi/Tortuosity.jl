module Tortuosity

using KernelAbstractions
using LinearAlgebra
using LinearSolve
using NaNStatistics
using SparseArrays
using OrdinaryDiffEqStabilizedRK
using OrdinaryDiffEqStabilizedRK: ROCK4, ODEProblem
using PrecompileTools: @compile_workload

# GPU backend registration (populated by package extensions)
const _preferred_gpu_backend = Ref{Any}(nothing)
const _gpu_adapt = Ref{Any}(identity)

"""True for GPU arrays; extensions override for CuArray, MtlArray, etc."""
_on_gpu(::AbstractArray) = false

"""
    _free!(x)

Release the storage behind `x` now instead of waiting for the garbage
collector, and return `x`'s slot to the allocator. `x` must not be read
afterwards.

Assembly runs a chain of stages that each allocate device arrays the size of
the connectivity list; without this the earlier stages' arrays stay reachable
while the next one allocates, so the pool has to hold several of them at once.
No-op for host arrays and for anything that is not an array (a scalar `D`, say);
GPU backends override it in their extension.
"""
_free!(x) = nothing

include("weakdeps.jl")
include("utils.jl")
include("geometry.jl")
include("imgen.jl")
include("sparse_type.jl")
include("kernels/graph.jl")
include("kernels/sparse.jl")
include("topotools.jl")
include("assembly.jl")
include("numpytools.jl")
include("pdetools.jl")
include("simulations.jl")
include("preconditioner.jl")
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
export two_level_preconditioner
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
# and the porous-media observables. Paths that need an optional dependency are
# precompiled by that dependency's extension instead, so nobody pays for a
# package they did not load: the effective-diffusivity fit lives in
# TortuosityLsqFitExt and blob generation in TortuosityImageFilteringExt.
# Intentionally CPU-only and tiny (12³ image): the goal is type coverage, not
# correctness — accuracy is verified in the test suite. See issue #30.
@compile_workload begin
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
end

end  # module Tortuosity
