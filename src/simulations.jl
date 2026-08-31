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
    SteadyDiffusionProblem{A,P,R,F}

Holds the data for a steady-state diffusion problem on a binary pore image.

# Fields
- `img::A`: boolean pore mask (`true` = pore).
- `axis::Symbol`: transport direction (`:x`, `:y`, or `:z`).
- `prob::LinearProblem`: the assembled linear system ready for `solve(sim.prob, alg)`.
- `D0::R`: reference pore diffusivity used by the direct transport-property methods.
- `flux::F`: compact inlet-edge data used by the direct transport-property methods.
"""
struct SteadyDiffusionProblem{A<:AbstractArray{Bool},P<:LinearProblem,R,F}
    img::A
    axis::Symbol
    prob::P
    D0::R
    flux::F
end

function SteadyDiffusionProblem(img, axis, prob::LinearProblem)
    return SteadyDiffusionProblem(img, axis, prob, nothing, nothing)
end

function SteadyDiffusionProblem{A}(img, axis, prob::LinearProblem) where {A<:AbstractArray{Bool}}
    converted = convert(A, img)
    return SteadyDiffusionProblem{A,typeof(prob),Nothing,Nothing}(
        converted, axis, prob, nothing, nothing,
    )
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
- `D`: diffusivity. `nothing` for uniform `D = 1` (default), a scalar for uniform
  diffusivity at that value, or an array matching `img` shape for a spatially
  variable one. A uniform `D` cancels out of the steady problem — `∇·(D∇c) = 0`
  reduces to `∇²c = 0` — so it leaves the concentration field where `D = 1` puts
  it and shows up only in [`effective_diffusivity`](@ref), which is what carries
  the physical units. Pass the same value to that and to [`tortuosity`](@ref),
  whose reference diffusivity divides it back out.
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
  and with a faster apply: measured at 600³ on GPU, both paths converging in 106
  iterations, 6.31 s and 7.92 GB against the assembled path's 8.33 s and
  19.09 GB. Worth reaching for from a few hundred voxels per side up. Default:
  `false`, the assembled path — nothing switches between them on your behalf.
- `Ti`: index type of the assembled matrix, `Int32` or `Int64`. `nothing`
  (default) picks the narrowest that fits. Rejected together with
  `matrixfree=true`, whose operator indexes on its own terms.
- `verbose`: print progress messages. Default: `false`.
"""
function SteadyDiffusionProblem(
    img; axis, D=nothing, gpu=nothing, warn_nonpercolating=nothing,
    matrixfree::Bool=false, Ti=nothing, verbose=false,
)
    verbose && @info "Preprocessing image..."
    img = atleast_3d(img)
    @assert img isa AbstractArray{Bool} "Image must be a boolean array"
    # Silently ignoring `Ti` on the matrix-free path would leave a caller who
    # asked for a wide index believing they got one.
    (matrixfree && !isnothing(Ti)) && throw(ArgumentError(
        "`Ti` sets the index type of the assembled matrix and has no meaning for the \
         matrix-free operator, which chooses its own; pass one or the other"
    ))
    # The struct holds `img` on CPU so postprocessing helpers (tortuosity,
    # effective_diffusivity, ...) work against a GPU-built sim. If the caller
    # handed us a GPU array, copy it back and warn once.
    if _on_gpu(img)
        @warn "`img` was passed on GPU; copying to CPU so the struct holds a CPU mask. \
               Pass `gpu=true` if you want the solver kernels to run on GPU." maxlog = 1
        img = Array(img)
    end
    # A scalar `D` is uniform diffusivity and stays a scalar: `atleast_3d` would
    # turn it into a 1x1x1 array and the shape check below would then reject it
    # with a message about matching the image.
    D = (isnothing(D) || D isa Number) ? D : atleast_3d(D)

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
    # A scalar `D` narrows the same way `_gpu_adapt` narrows an array, so the
    # device path computes its edge weights in `Float32` either way.
    D_dev = if isnothing(D)
        nothing
    elseif D isa Number
        gpu ? T(D) : D
    elseif gpu
        adapted = _gpu_adapt[](D)
        isnothing(_device_backend(adapted)) ? _gpu_adapt[](Array(D)) : adapted
    elseif !isnothing(_device_backend(D))
        Array(D)
    else
        D
    end

    # Assemble the Dirichlet-eliminated Laplacian in one shot. A fixed
    # concentration drop of 1.0 between inlet and outlet is the boundary
    # condition, and the two faces are implied by `axis`.
    verbose && @info "Assembling the linear system..."
    # The matrix-free operator recomputes its weights from `D` on every apply, so
    # it holds that array for its whole life. When the device copy is one we
    # made, ownership goes with it — otherwise nothing would ever release it.
    A, b, inlet_flux = if matrixfree
        build_steady_operator(img_dev; nnodes=nnodes, axis=axis, D=D_dev, T=T,
                              owns_D=(D_dev isa AbstractArray && D_dev !== D),
                              return_flux=true)
    else
        build_steady_system(
            img_dev; nnodes=nnodes, axis=axis, D=D_dev, T=T, Ti=Ti, return_flux=true,
        )
    end
    D0 = if isnothing(D_dev)
        one(T)
    elseif D_dev isa Number
        D_dev
    else
        D_mask = isnothing(_device_backend(D_dev)) ? img : img_dev
        _reference_diffusivity(D_dev, D_mask)
    end
    if gpu
        # The device copies are dead the moment the system is built; releasing
        # them here rather than at the next GC frees ~2.4 GiB at 800³ before the
        # solver starts allocating its Krylov vectors. Only the copies we made
        # are ours to release, and `_gpu_adapt` allocates one even when `D` is
        # already a device array of the right element type, so `D_dev === D`
        # holds on the CPU path alone.
        _free!(img_dev)
        (matrixfree || D_dev === D) || _free!(D_dev)
    end

    return SteadyDiffusionProblem(img, axis, LinearProblem(A, b), D0, inlet_flux)
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
- **Refinement.** A `Float32` system is refined against a `Float64` residual
  before it is returned, because the residual `Float32` CG stops on is a
  recurrence that drifts away from `b - A*x`: without this the solver reports
  success on a low-porosity image whose tortuosity is wrong by 2e-3. `sol.resid[]`
  is then the true relative residual and `sol.iters` counts every iteration
  spent, correction rounds included. `sol.retcode` likewise describes the vector
  returned, so a base solve that hit its iteration cap and was then refined below
  `reltol` reports `Success`. `sol.stats` stays a record of the base solve alone —
  its `niter` and `residuals` are that solve's, because a correction's residuals
  belong to a different system and cannot be appended. Pass `refine=false` for the
  unrepaired behaviour.

# Arguments
- `sim`: the problem, from [`SteadyDiffusionProblem`](@ref).
- `alg`: any LinearSolve algorithm. Default: `KrylovJL_CG()`.

# Keyword Arguments
- `precond`: `:auto`, `:none`, or a preconditioner to use as `Pl`.
- `reltol`, `abstol`, `maxiters`: passed to `solve`; `nothing` means "decide".
- `verbose`: print what was chosen. Default: `false`.
- `refine`: refine the solution against a `Float64` residual. `nothing` (the
  default) turns it on for a `Float32` system and off for a `Float64` one.
- Any other keyword is forwarded to LinearSolve unchanged.

# Returns
The LinearSolve solution. `sol.u` is the pore-ordered concentration vector, in
the same numbering either path produces, so it feeds
[`reconstruct_field`](@ref) directly.
"""
function LinearSolve.solve(
    sim::SteadyDiffusionProblem, alg=KrylovJL_CG();
    precond=:auto, reltol=nothing, abstol=nothing, maxiters=nothing,
    verbose=false, refine=nothing, kwargs...,
)
    T = eltype(sim.prob.b)
    reltol = isnothing(reltol) ? _default_reltol(T) : reltol
    Pl = _resolve_precond(precond, sim, verbose)
    # LinearSolve defaults `abstol` to `sqrt(eps(T))`, which on `Float32` is
    # 3.5e-4 — loose enough to stop the solve before the `reltol` chosen just
    # above is anywhere near met. Defaulting it to zero makes `reltol` the only
    # stopping rule, which is what this function documents.
    opts = Any[:reltol => reltol, :abstol => isnothing(abstol) ? zero(T) : abstol]
    isnothing(maxiters) || push!(opts, :maxiters => maxiters)
    isnothing(Pl) || push!(opts, :Pl => Pl)
    verbose && @info "Solving" alg reltol precond = isnothing(Pl) ? :none : :two_level
    sol = solve(sim.prob, alg; opts..., kwargs...)
    (isnothing(refine) ? _refines_by_default(T) : refine) || return sol
    return _refine(sol, sim, alg)
end

# `Float32` CG cannot drive the relative residual much below `1e-6`; asking it to
# is how a GPU run ends in `maxiters` rather than `Success`.
_default_reltol(::Type{Float32}) = 1.0f-6
_default_reltol(::Type{T}) where {T} = T(1e-10)

# Refinement is for the low-precision path only. It cannot be decided from the
# residual: `Float64` CG overshoots its own `reltol` by 2.4x and `Float32` by
# 2.6x, so the two are indistinguishable by any ratio test. What separates them
# is the consequence — at `Float64` that overshoot leaves 1e-8 in tortuosity, at
# `Float32` it leaves 2e-3.
_refines_by_default(::Type{Float32}) = true
_refines_by_default(::Type{T}) where {T} = false

# Widen or narrow a vector without assuming which array type it is, so the same
# code serves the host and every device backend.
_as(::Type{T}, v) where {T} = (w = similar(v, T); w .= v; w)

"""
    _refine(sol, sim, alg)

Repair a `Float32` solve by refining it against a `Float64` residual.

`Float32` CG stops on a recursively-updated residual that drifts away from
`b - A*x`. It therefore reports success while the answer is still wrong — by
2e-3 in tortuosity on a low-porosity image, where the benchmark asks for 1e-3.
Recomputing the true residual and solving the correction equation `A*d = r`
recovers it, at 5e-7.

The residual is `Float64` while the operator stays `Float32`: the matvec
accumulates in `eltype(y)`, so a `Float64` output and iterate give a genuine
`Float64` product without a second copy of the operator. Both must be `Float64`
— feeding a `Float32` iterate rounds each product before the accumulator sees
it and recovers nothing. A `Float32` residual is not enough either: it reaches
3e-5 and then degrades, because by then the correction it feeds on is rounding
noise amplified by the conditioning.

Rounds stop when the true residual meets `reltol`, or when a round fails to
shrink it — past that point there is no signal left to correct.
"""
function _refine(sol, sim, alg; rounds=8, shrink=0.5, correction_reltol=1.0f-1)
    A, b = sim.prob.A, sim.prob.b
    # Reuse the cache the main solve already built: `cache.b = r; solve!(cache)`
    # runs the correction on the Krylov vectors and the preconditioner that are
    # already resident. Building a fresh `LinearProblem` per round instead cost a
    # second workspace — 42 B per pore node against the 20 B the algorithm needs.
    #
    # `cache.u` is the same array as `sol.u`, so the first correction overwrites
    # it. `sol.u` is therefore read once, here, before any of that happens.
    cache = sol.cache
    b_before, reltol_before = cache.b, cache.reltol
    # Every residual below is measured relative to `‖b‖`, so a zero right-hand side
    # would make each of them `NaN` — and `NaN` compares false against the shrink
    # test, so the rounds would all run and report a `NaN` residual for an answer
    # that is exactly right. A zero right-hand side has `x = 0` as its exact
    # solution and nothing to refine.
    nb = Float64(norm(b))
    iszero(nb) && return sol
    # Refinement needs 20 bytes per pore node on top of the solve: two `Float64`
    # vectors and one `Float32`. Measured, the base solve itself costs 32 B per pore
    # node plus 4 B per voxel of the full grid, so a 950M-node image already holds
    # 34.4 GB of a 50.7 GB card and the refinement buffers do not fit. Failing to
    # allocate them must not take the solve down with it — and must not quietly
    # hand back the unrefined answer either, since that is the defect refinement
    # exists to remove.
    local x64, r64, correction_rhs
    try
        x64 = _as(Float64, sol.u)
        r64 = similar(x64)
        correction_rhs = similar(b)
    catch err
        @warn """
              Not enough device memory to refine this solve, so it is returned unrefined. \
              On an ill-conditioned image — low porosity, high tortuosity — the tortuosity \
              may be wrong by ~1e-3 even though the solver reports success. Refinement needs \
              about 20 bytes per pore node beyond the solve itself. Solve on the CPU, or in \
              smaller pieces, if you need the accuracy here.""" nodes = length(b) exception = err
        return sol
    end
    # `sol.stats` is the workspace object Krylov mutates in place, so each
    # correction below overwrites it. Returning it as-is would hand back a record
    # of the last correction round wearing the whole solve's name — `niter` would
    # read 3 where `iters` read 500. Copy it now, while it still describes the
    # base solve, and return that copy instead. The residual history inside it is
    # the base solve's, and the corrections cannot be appended to it: they are
    # residuals of `A*d = r`, a different system, so concatenating them would be
    # the same kind of lie in a different field.
    base_stats = sol.stats === nothing ? nothing : deepcopy(sol.stats)
    corrections_ok = true
    prev = Inf
    iters = sol.iters
    resid = nothing
    for k in 1:rounds
        mul!(r64, A, x64)
        # `b` is `Float32` and widens exactly here, so it never needs a `Float64`
        # copy of its own.
        r64 .= b .- r64
        resid = norm(r64) / nb
        # Deliberately not stopping at `reltol`. The error in tortuosity is the
        # residual times the conditioning, and that factor reaches ~760 on a
        # low-porosity image: stopping at `reltol=1e-6` leaves 4e-4, which clears
        # the 1e-3 benchmark target by only 2x. Refining until the residual stops
        # improving costs one or two more rounds and leaves ~1e-6.
        if k > 1 && resid > shrink * prev
            # Stopping here leaves the last correction applied, and a correction
            # that failed to shrink the residual may have grown it. `sol.u` still
            # holds that correction — nothing overwrites it until the narrowing
            # below — so undoing it costs no buffer and keeps the promise the
            # function is built on: refinement never returns a worse answer than
            # the solve it repairs.
            resid > prev && (x64 .-= sol.u)
            break
        end
        prev = resid
        correction_rhs .= r64
        cache.b = correction_rhs
        # `oftype` because the cache's tolerance field is typed to the problem's,
        # and a `Float32` literal into a `Float64` field is a `TypeError`. That
        # only shows up when a caller asks for refinement on a `Float64` system,
        # which the default never does.
        cache.reltol = oftype(cache.reltol, correction_reltol)
        correction = LinearSolve.solve!(cache)
        corrections_ok &= correction.retcode == LinearSolve.SciMLBase.ReturnCode.Success
        x64 .+= correction.u
        iters += correction.iters
    end
    # Report the residual of the vector actually returned, which is narrowed back
    # to the working precision and so is slightly worse than the `Float64` iterate
    # refinement carried internally. Reporting the iterate's residual instead
    # would be the same defect this function exists to remove: a number that does
    # not describe the answer it is attached to.
    # Narrow into `sol.u` rather than allocating a fourth buffer. By this point it
    # holds nothing but the last correction, and reusing it does two things: it
    # keeps refinement at 20 bytes per pore node instead of 24, and it leaves every
    # allocation refinement makes inside the guard above. A fourth allocation after
    # the guard is a fourth way to throw where the guard promised a warning.
    u = sol.u
    u .= x64
    x64 .= u
    mul!(r64, A, x64)
    r64 .= b .- r64
    resid = norm(r64) / nb
    # Hand the cache back as it was found. The rounds above pointed its right-hand
    # side at a scratch vector and loosened its tolerance, and a caller who reuses
    # the cache should not inherit either.
    cache.b, cache.reltol = b_before, reltol_before
    # `retcode` describes the vector returned, for the same reason `resid` does.
    # A base solve that stopped at its iteration cap and was then refined below
    # the requested tolerance did reach that tolerance, and reporting `MaxIters`
    # for it is the same class of defect this function exists to remove: a field
    # that describes something other than the answer it is attached to. A
    # correction that failed outright is the one event that can leave the answer
    # worse than the fields claim, so it is the one that forces a failure.
    retcode = if !corrections_ok
        LinearSolve.SciMLBase.ReturnCode.Failure
    elseif resid <= reltol_before
        LinearSolve.SciMLBase.ReturnCode.Success
    else
        sol.retcode
    end
    if base_stats !== nothing
        base_stats.solved = retcode == LinearSolve.SciMLBase.ReturnCode.Success
        base_stats.status = "base solve, then refined against a Float64 residual: " *
                            "`niter` and `residuals` describe the base solve, " *
                            "`sol.iters` counts the correction rounds as well"
    end
    # `resid` is wrapped in a `Ref` because that is what an unrefined solve
    # returns, and a field that changes type with the working precision would make
    # `sol.resid` something a caller has to branch on. The number inside is not the
    # same quantity LinearSolve puts there — theirs is an absolute preconditioned
    # residual, this is the true one relative to `‖b‖` — but that difference is the
    # point of refining, and it is documented above.
    return LinearSolve.SciMLBase.build_linear_solution(
        alg, u, Ref(resid), sol.cache; retcode=retcode, iters=iters, stats=base_stats,
    )
end

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
