# CPU-specialized conjugate gradient with deterministic threaded vector kernels.

const HOST_CG_MIN_NODES = 10_000
const _HOST_CG_CHUNKS_PER_THREAD = 4

_host_cg_continue(_) = false

"""
    HostCG(; callback)

Conjugate gradient specialized for host vectors. The iteration is mathematically
the same as `KrylovJL_CG`, but combines the two solution/residual updates and
fuses the preconditioner prolongation with `rᵀz` to reduce memory traffic.

The callback receives the live [`HostCGWorkspace`](@ref) after each iteration
and must return `true` to stop.
"""
struct HostCG{F}
    callback::F
end

HostCG(; callback=_host_cg_continue) = HostCG(callback)

struct HostCGWorkspace{V,S}
    x::V
    r::V
    z::V
    p::V
    Ap::V
    partial::S
end

mutable struct HostCGStats{T}
    niter::Int
    solved::Bool
    status::String
    residuals::Vector{T}
end

mutable struct HostCGCache{A,B,P,Alg,W,T}
    A::A
    b::B
    P::P
    alg::Alg
    reltol::T
    abstol::T
    maxiters::Int
    workspace::W
    u::B
end

@inline function _host_chunks(n)
    return min(n, max(1, _HOST_CG_CHUNKS_PER_THREAD * Threads.nthreads()))
end

@inline function _host_chunk_bounds(n, nchunks, chunk)
    chunk_size = cld(n, nchunks)
    ilo = (chunk - 1) * chunk_size + 1
    return ilo, min(ilo + chunk_size - 1, n)
end

function _host_dot(x, y, partial)
    n = length(x)
    nchunks = length(partial)
    Threads.@threads :dynamic for chunk in 1:nchunks
        ilo, ihi = _host_chunk_bounds(n, nchunks, chunk)
        acc = zero(eltype(partial))
        @inbounds @simd for i in ilo:ihi
            acc += x[i] * y[i]
        end
        partial[chunk] = acc
    end
    return sum(partial)
end

function _host_axpby!(y, alpha, x, beta)
    n = length(y)
    nchunks = _host_chunks(n)
    Threads.@threads :dynamic for chunk in 1:nchunks
        ilo, ihi = _host_chunk_bounds(n, nchunks, chunk)
        @inbounds @simd for i in ilo:ihi
            y[i] = alpha * x[i] + beta * y[i]
        end
    end
    return y
end

function _host_update_xr!(x, r, p, Ap, alpha)
    n = length(x)
    nchunks = _host_chunks(n)
    Threads.@threads :dynamic for chunk in 1:nchunks
        ilo, ihi = _host_chunk_bounds(n, nchunks, chunk)
        @inbounds @simd for i in ilo:ihi
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        end
    end
    return nothing
end

function _host_prolong_dot!(y, agg, xc, x, inv_lambda, partial)
    n = length(agg)
    nchunks = length(partial)
    Threads.@threads :dynamic for chunk in 1:nchunks
        ilo, ihi = _host_chunk_bounds(n, nchunks, chunk)
        acc = zero(eltype(partial))
        @inbounds @simd for i in ilo:ihi
            a = agg[i]
            yi = (a > 0 ? xc[a] : zero(eltype(y))) + inv_lambda * x[i]
            y[i] = yi
            acc += x[i] * yi
        end
        partial[chunk] = acc
    end
    return sum(partial)
end

function _host_precondition_dot!(y, P, x, partial)
    if P isa TwoLevelPreconditioner && P.agg isa Aggregation &&
       P.agg.fwd isa Vector && P.rc isa Vector && P.xc isa Vector
        _restrict!(P.rc, P.agg, x)
        copyto!(P.coarse_rhs, P.rc)
        _vcycle!(P.coarse_sol, P.levels, 1, P.coarse_rhs, P.fact)
        copyto!(P.xc, P.coarse_sol)
        return _host_prolong_dot!(y, P.agg.fwd, P.xc, x, P.inv_lambda, partial)
    end

    ldiv!(y, P, x)
    return _host_dot(x, y, partial)
end

function _host_apply_preconditioner!(y, P, x, partial)
    if isnothing(P)
        y === x || copyto!(y, x)
        return _host_dot(x, y, partial)
    end
    return _host_precondition_dot!(y, P, x, partial)
end

function _host_cg_cache(A, b, P, alg, reltol, abstol, maxiters)
    n = length(b)
    x = similar(b)
    r = similar(b)
    z = isnothing(P) ? r : similar(b)
    workspace = HostCGWorkspace(
        x, r, z, similar(b), similar(b),
        zeros(eltype(b), _host_chunks(n)),
    )
    return HostCGCache(
        A, b, P, alg, convert(eltype(b), reltol), convert(eltype(b), abstol),
        maxiters, workspace, x,
    )
end

function _host_cg_solve!(cache::HostCGCache)
    A, b, P = cache.A, cache.b, cache.P
    workspace = cache.workspace
    x, r, z, p, Ap, partial =
        workspace.x, workspace.r, workspace.z, workspace.p, workspace.Ap, workspace.partial
    fill!(x, zero(eltype(x)))
    copyto!(r, b)

    gamma = _host_apply_preconditioner!(z, P, r, partial)
    gamma >= 0 || error("The linear operator or preconditioner is not positive definite")
    copyto!(p, z)
    initial_residual = sqrt(gamma)
    tolerance = cache.abstol + cache.reltol * initial_residual
    residual = initial_residual
    iterations = 0
    requested_exit = false
    curvature_failure = false

    while residual > tolerance && iterations < cache.maxiters
        mul!(Ap, A, p)
        pAp = _host_dot(p, Ap, partial)
        if pAp <= zero(pAp)
            curvature_failure = true
            break
        end

        alpha = gamma / pAp
        _host_update_xr!(x, r, p, Ap, alpha)
        gamma_next = _host_apply_preconditioner!(z, P, r, partial)
        gamma_next >= 0 ||
            error("The linear operator or preconditioner is not positive definite")
        residual = sqrt(gamma_next)
        solved = residual <= tolerance
        if !solved
            beta = gamma_next / gamma
            _host_axpby!(p, one(eltype(p)), z, beta)
            gamma = gamma_next
        end
        iterations += 1
        requested_exit = cache.alg.callback(workspace)::Bool
        (requested_exit || solved) && break
    end

    solved = !curvature_failure && residual <= tolerance
    retcode = if requested_exit
        LinearSolve.SciMLBase.ReturnCode.Terminated
    elseif curvature_failure
        LinearSolve.SciMLBase.ReturnCode.Failure
    elseif solved
        LinearSolve.SciMLBase.ReturnCode.Success
    elseif iterations >= cache.maxiters
        LinearSolve.SciMLBase.ReturnCode.MaxIters
    else
        LinearSolve.SciMLBase.ReturnCode.Failure
    end
    status = requested_exit ? "user-requested exit" :
             solved ? "solution good enough given abstol and reltol" :
             curvature_failure ? "nonpositive curvature detected" :
             "maximum number of iterations exceeded"
    stats = HostCGStats(iterations, solved, status, eltype(b)[initial_residual, residual])
    return LinearSolve.SciMLBase.build_linear_solution(
        cache.alg, x, Ref(residual), cache;
        retcode, iters=iterations, stats,
    )
end

LinearSolve.solve!(cache::HostCGCache) = _host_cg_solve!(cache)

function LinearSolve.solve(
    sim::SteadyDiffusionProblem, alg::HostCG;
    precond=:auto, reltol=nothing, abstol=nothing, maxiters=nothing,
    verbose=false, refine=nothing,
)
    isnothing(refine) || refine == false ||
        throw(ArgumentError("HostCG solves Float64 host systems and does not use refinement"))
    b = sim.prob.b
    b isa Vector || throw(ArgumentError("HostCG requires a host Vector right-hand side"))
    T = eltype(b)
    T === Float64 || throw(ArgumentError("HostCG requires a Float64 host system"))
    reltol = isnothing(reltol) ? _default_reltol(T) : reltol
    abstol = isnothing(abstol) ? zero(T) : abstol
    maxiters = isnothing(maxiters) ? length(b) :
               iszero(maxiters) ? 2 * length(b) : maxiters
    P = _resolve_precond(precond, sim, verbose)
    verbose && @info "Solving" alg reltol precond = isnothing(P) ? :none : :two_level
    cache = _host_cg_cache(sim.prob.A, b, P, alg, reltol, abstol, maxiters)
    return solve!(cache)
end

function LinearSolve.solve(
    sim::SteadyDiffusionProblem;
    precond=:auto, reltol=nothing, abstol=nothing, maxiters=nothing,
    verbose=false, refine=nothing, kwargs...,
)
    use_host_cg = sim.prob.b isa Vector{Float64} &&
                  length(sim.prob.b) >= HOST_CG_MIN_NODES &&
                  isempty(kwargs) && refine !== true
    alg = use_host_cg ? HostCG() : KrylovJL_CG()
    return solve(
        sim, alg; precond, reltol, abstol, maxiters, verbose, refine, kwargs...,
    )
end
