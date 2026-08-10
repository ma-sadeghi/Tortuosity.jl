# Steady-state diffusion simulation setup and utilities.

"""
    interpolate_edge_values(node_vals, conns)

Compute edge weights from node diffusivities using the harmonic mean:
`2 * D_a * D_b / (D_a + D_b)`. This is the standard finite-volume interface
conductance for unit-spacing grids.

# Physics

Each voxel stores a cell-centered diffusivity `D`. In a finite-volume
scheme the edge between two adjacent cells acts as two half-cell
conductances in series — one for the path from cell `a`'s center to the
face (length Δx/2), and one from the face to cell `b`'s center. With
Δx = 1, each half has conductance `2·D`, so resistances add:

    R_edge = 1/(2·D_a) + 1/(2·D_b)
    G_edge = 1/R_edge = 2·D_a·D_b / (D_a + D_b)

i.e. the harmonic mean of the two node diffusivities. That simplified form
is the one we broadcast — it is ~3× faster on GPU than the literal
`1/(1/(2·a) + 1/(2·b))` (three reciprocals vs one divide).

The two are *not* bit-identical: in `Float32` they round differently for
roughly half of all input pairs, by at most 3 ULP over a survey of 2·10⁵
random pairs. That is orders of magnitude below the `O(dx²)` discretisation
error, so it does not affect the solve — but do not rely on exact equality
when comparing against a reference implementation that uses the other form.

# Arguments
- `node_vals`: diffusivity value for each node (1D vector, length = number of nodes).
- `conns`: `nedges x 2` connectivity matrix where each row is a `(source, target)` pair.
"""
function interpolate_edge_values(node_vals, conns)
    P1 = @view conns[:, 1]
    P2 = @view conns[:, 2]
    a = @view node_vals[P1]
    b = @view node_vals[P2]
    # Harmonic mean of half-cell conductances in series — see docstring.
    return (2 .* a .* b) ./ (a .+ b)
end

"""
    _warn_nonpercolating(img, axis, check)

Warn when part of the pore space does not span the domain from inlet to outlet
along `axis`.

Such voxels are not an error and nothing is changed: they simply carry no
steady flux while still counting toward porosity, so the reported tortuosity
`τ = ε / D_eff` includes stagnant volume. Worth surfacing because the image
usually looks fine and the number that comes out looks plausible.

`check` follows the same three-state convention as the `gpu` keyword:
`nothing` decides automatically, `true`/`false` force it. The automatic choice
is by size — the check labels connected components over the full grid, which
allocates an `Int` array the shape of `img`. That is a few hundred MB and well
under a second up to ~50M voxels (measured: 0.31 s against 5.6 s of problem
construction at 256³, i.e. ~6% overhead), but at 800³ it would be several GB,
so it is skipped there unless explicitly requested.
"""
function _warn_nonpercolating(img, axis::Symbol, check::Union{Nothing,Bool})
    do_check = isnothing(check) ? length(img) <= 50_000_000 : check
    do_check || return nothing

    n_pore = count(img)
    n_dead = n_pore - count(Imaginator.trim_nonpercolating_paths(img; axis=axis))
    n_dead == 0 && return nothing

    pct = round(100 * n_dead / n_pore; digits=2)
    @warn "$(n_dead) of $(n_pore) pore voxels ($(pct)%) are not connected from \
           inlet to outlet along axis :$(axis). They carry no steady flux but \
           still count toward porosity, so the reported tortuosity includes \
           that stagnant volume. Pass the image through \
           `Imaginator.trim_nonpercolating_paths(img; axis=:$(axis))` first to \
           exclude them, or set `warn_nonpercolating=false` to silence this."
    return nothing
end

"""
    SteadyDiffusionProblem{A}

Holds the data for a steady-state diffusion problem on a binary pore image.

# Fields
- `img::A`: boolean pore mask (`true` = pore).
- `axis::Symbol`: transport direction (`:x`, `:y`, or `:z`).
- `prob::LinearProblem`: the assembled linear system ready for `solve(sim.prob, alg)`.
"""
struct SteadyDiffusionProblem{A<:AbstractArray{Bool}}
    img::A
    axis::Symbol
    prob::LinearProblem
end

function Base.show(io::IO, ts::SteadyDiffusionProblem)
    gpu = _on_gpu(ts.prob.b)
    form = ts.prob.A isa MaskedLaplacian ? "matrix-free" : "assembled"
    msg = "SteadyDiffusionProblem(shape=$(size(ts.img)), axis=$(ts.axis), gpu=$(gpu), $(form))"
    return print(io, msg)
end

"""
    SteadyDiffusionProblem(img; axis, D=nothing, gpu=nothing, verbose=false)

Construct a `SteadyDiffusionProblem` for steady-state diffusion on a binary pore
image. Builds the graph Laplacian, applies Dirichlet boundary conditions
(`c = 1` at inlet, `c = 0` at outlet), and returns a ready-to-solve `LinearProblem`.

# Arguments
- `img`: boolean array where `true` = pore, `false` = solid. Promoted to 3D if needed.

# Keyword Arguments
- `axis`: transport direction (`:x`, `:y`, or `:z`).
- `D`: diffusivity. `nothing` for uniform (default), or an array matching `img` shape
  for spatially variable diffusivity.
- `gpu`: `true` to force GPU, `false` for CPU, `nothing` (default) to auto-detect
  (uses GPU when a backend package is loaded *and* the image has ≥100k pore
  voxels). See [GPU backends](@ref) for how to activate CUDA, Metal, or AMDGPU.
- `warn_nonpercolating`: warn when part of the pore space does not span the
  domain along `axis`. Nothing is changed either way — such voxels carry no
  steady flux but still count toward porosity, so `τ` includes stagnant volume.
  `nothing` (default) runs the check for images up to ~50M voxels and skips it
  above that; `true`/`false` force it. See [`_warn_nonpercolating`](@ref).
- `matrixfree`: build the operator as a matrix-free stencil
  ([`MaskedLaplacian`](@ref)) instead of an assembled sparse matrix. Same pore
  numbering, same right-hand side and the same `τ`, at a fraction of the memory
  — which is what makes images past the assembled path's ~850³ ceiling solvable
  — and a faster apply. Default: `false`, the assembled path.
- `verbose`: print progress messages. Default: `false`.
"""
function SteadyDiffusionProblem(
    img; axis, D=nothing, gpu=nothing, warn_nonpercolating=nothing,
    matrixfree::Bool=false, verbose=false,
)
    verbose && @info "Preprocessing image..."
    img = atleast_3d(img)
    @assert img isa AbstractArray{Bool} "Image must be a boolean array"
    # The struct holds `img` on CPU so postprocessing helpers (tortuosity,
    # effective_diffusivity, ...) work against a GPU-built sim. If the caller
    # handed us a GPU array, copy it back and warn once.
    if _on_gpu(img)
        @warn "`img` was passed on GPU; copying to CPU so the struct holds a CPU mask. \
               Pass `gpu=true` if you want the solver kernels to run on GPU." maxlog = 1
        img = Array(img)
    end
    D = isnothing(D) ? nothing : atleast_3d(D)

    # Deal with variable diffusivity
    if D isa AbstractArray
        @assert size(D) == size(img) "Diffusivity matrix D must match image size"
        @assert count(D .> 0) == count(img) "Diffusivity matrix D must have the same \
            number of non-zero elements as the image"
    end

    nnodes = sum(img)
    @assert nnodes > 0 "Image must contain at least one pore voxel (got all-solid)"
    # With one voxel along the transport axis the inlet and outlet faces are the
    # same voxels, so both Dirichlet values land on the same nodes and the
    # solution is silently meaningless. Easy to hit by accident: `atleast_3d`
    # promotes a 2D image to `(m, n, 1)`, so asking for `axis=:z` on 2D data
    # arrives here. `TransientDiffusionProblem` already rejects this.
    @assert size(img, axis_dim(axis)) > 1 "Image must have at least 2 voxels along the chosen axis"
    _warn_nonpercolating(img, axis, warn_nonpercolating)
    # Auto-detect GPU: use if backend is available and image is large enough.
    # When the image is big enough to benefit from GPU but no backend has been
    # loaded, nudge the user once — the alternative is a silent CPU fallback
    # where the caller thinks they're getting GPU performance but aren't.
    if isnothing(gpu)
        has_backend = !isnothing(_preferred_gpu_backend[])
        if !has_backend && nnodes >= 100_000
            @warn "Image has $(nnodes) pore voxels but no GPU backend is loaded; \
                   running on CPU. To enable GPU kernels, load a backend package \
                   (`using CUDA`, `using Metal`, or `using AMDGPU`) before \
                   constructing the simulation. Pass `gpu=false` explicitly to \
                   silence this message." maxlog = 1
        end
        gpu = has_backend && nnodes >= 100_000
    elseif gpu && isnothing(_preferred_gpu_backend[])
        error("`gpu=true` was requested but no GPU backend is registered. \
               Load a GPU package first (e.g. `using CUDA`, `using Metal`, or `using AMDGPU`).")
    end

    # Move to GPU if needed. Keep `img` on CPU for the struct (postprocessing
    # helpers like tortuosity() expect a CPU mask); `img_dev` is the copy
    # handed to the kernels.
    verbose && gpu && @info "Using GPU..."
    T = gpu ? Float32 : Float64
    img_dev = gpu ? _gpu_adapt[](img) : img
    D_dev = isnothing(D) ? nothing : (gpu ? _gpu_adapt[](D) : D)

    # Assemble the Dirichlet-eliminated Laplacian in one shot. A fixed
    # concentration drop of 1.0 between inlet and outlet is the boundary
    # condition, and the two faces are implied by `axis`.
    verbose && @info "Assembling the linear system..."
    build = matrixfree ? build_steady_operator : build_steady_system
    A, b = build(img_dev; nnodes=nnodes, axis=axis, D=D_dev, T=T)
    if gpu
        # The device copies are dead the moment the system is built; releasing
        # them here rather than at the next GC frees ~2.4 GiB at 800³ before the
        # solver starts allocating its Krylov vectors. Only the copies we made
        # are ours to release — `_gpu_adapt` hands a `D` that is already a device
        # array of the right eltype straight back.
        _free!(img_dev)
        # The matrix-free operator recomputes its weights from `D` on every
        # apply, so it holds that array for its whole life; only the assembled
        # path is finished with it here.
        (matrixfree || D_dev === D) || _free!(D_dev)
    end

    return SteadyDiffusionProblem(img, axis, LinearProblem(A, b))
end

"""
    solve(sim::SteadyDiffusionProblem, alg=KrylovJL_CG(); precond=:auto, reltol=nothing, ...)

Solve a steady diffusion problem, choosing the preconditioner and the tolerance
for you.

This is the package-owned entry point. `solve(sim.prob, alg)` remains the
unopinionated one and is untouched by anything here: it takes LinearSolve's
defaults, no preconditioner, and whatever tolerance you pass. Use this form when
you want the settings the package considers right for the problem in front of
it, and that one when you want to drive LinearSolve yourself.

What it decides:

- **Preconditioner.** `precond=:auto` (the default) builds a
  [`two_level_preconditioner`](@ref) once the problem is large enough to pay for
  the coarse solve, and runs unpreconditioned below that. The coarse space cuts
  iteration counts by an order of magnitude at bench sizes and — unlike a cheap
  diagonal scaling — keeps them nearly flat as the image grows. Pass
  `precond=:none` to disable it, or a preconditioner object to supply your own.
- **Tolerance.** `reltol=nothing` picks `1e-10` for a `Float64` (CPU) system and
  `1e-6` for a `Float32` (GPU) one. `Float32` CG stalls before `1e-7`, so a
  tolerance carried over from a CPU run is a request the solver cannot meet.

# Arguments
- `sim`: the problem, from [`SteadyDiffusionProblem`](@ref).
- `alg`: any LinearSolve algorithm. Default: `KrylovJL_CG()`.

# Keyword Arguments
- `precond`: `:auto`, `:none`, or a preconditioner to use as `Pl`.
- `reltol`, `abstol`, `maxiters`: passed to `solve`; `nothing` means "decide".
- `verbose`: print what was chosen. Default: `false`.
- Any other keyword is forwarded to LinearSolve unchanged.

# Returns
The LinearSolve solution. `sol.u` is the pore-ordered concentration vector, in
the same numbering either path produces, so it feeds
[`reconstruct_field`](@ref) directly.
"""
function LinearSolve.solve(
    sim::SteadyDiffusionProblem, alg=KrylovJL_CG();
    precond=:auto, reltol=nothing, abstol=nothing, maxiters=nothing,
    verbose=false, kwargs...,
)
    T = eltype(sim.prob.b)
    reltol = isnothing(reltol) ? _default_reltol(T) : reltol
    Pl = _resolve_precond(precond, sim, verbose)
    opts = Any[:reltol => reltol]
    isnothing(abstol) || push!(opts, :abstol => abstol)
    isnothing(maxiters) || push!(opts, :maxiters => maxiters)
    isnothing(Pl) || push!(opts, :Pl => Pl)
    verbose && @info "Solving" alg reltol precond = isnothing(Pl) ? :none : :two_level
    return solve(sim.prob, alg; opts..., kwargs...)
end

# `Float32` CG cannot drive the relative residual much below `1e-6`; asking it to
# is how a GPU run ends in `maxiters` rather than `Success`.
_default_reltol(::Type{Float32}) = 1.0f-6
_default_reltol(::Type{T}) where {T} = T(1e-10)

# Below this the coarse solve costs more than the iterations it removes, and it
# is also the size at which a system stops fitting comfortably in cache.
const _PRECOND_MIN_NODES = 100_000

function _resolve_precond(precond, sim, verbose)
    precond === :none && return nothing
    precond === :auto || return precond
    size(sim.prob.A, 1) < _PRECOND_MIN_NODES && return nothing
    Pl = two_level_preconditioner(sim)
    # `two_level_preconditioner` returns `nothing` when there is no usable coarse
    # space — an empty system, every block dropped, or a coarse factorization
    # that failed. Running unpreconditioned is the honest fallback.
    verbose && isnothing(Pl) && @info "No usable coarse space; solving unpreconditioned"
    return Pl
end
