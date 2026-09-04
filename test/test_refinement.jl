# Contract tests for iterative refinement of a low-precision solve (`_refine`).
#
# test_gpu_e2e.jl already checks that refinement repairs a Float32 GPU solve,
# but that file runs only where a GPU backend is functional. On ordinary CI the
# whole of `_refine` — the code that rewrites the residual, the retcode and the
# stats every caller reads back — therefore executes never. This file drives the
# same code on the host by narrowing an assembled system to Float32, which is
# the same regime without the device.
#
# The promises pinned here are the ones a caller cannot check for themselves:
# the reported residual describes the vector handed back rather than the
# Float64 iterate kept internally, the borrowed cache is returned as it was
# found, and a solve that cannot be improved is returned unharmed rather than
# worse.

using Test
using LinearAlgebra
using SparseArrays
using Tortuosity
using Tortuosity:
    Imaginator,
    LinearSolve,
    _refine,
    _refines_by_default

# A Float32 system on the host. `SteadyDiffusionProblem` reaches for Float32 on
# the GPU path alone, so the element type is imposed after assembly; every other
# property of the problem is the one the constructor produced.
function float32_host_problem(img; axis=:x)
    sim = SteadyDiffusionProblem(img; axis=axis, gpu=false, warn_nonpercolating=false)
    A = sim.prob.A
    A32 = SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(A.rowval), Float32.(A.nzval))
    prob = LinearSolve.LinearProblem(A32, Float32.(sim.prob.b))
    return SteadyDiffusionProblem(sim.img, sim.axis, prob)
end

# Low porosity and an irregular pore space: refinement exists for the
# ill-conditioned case, and on an open box Float32 CG already lands on the
# answer and there is nothing left to repair.
function refinement_image()
    img = Imaginator.blobs(; shape=(32, 32, 32), porosity=0.5, blobiness=1, seed=1)
    return Array{Bool}(Imaginator.trim_nonpercolating_paths(Array{Bool}(img); axis=:x))
end

const REFINE_IMAGE = refinement_image()

# The quantity `_refine` promises to report: the residual of the vector it
# returns, accumulated in Float64 against the Float32 operator. Recomputed here
# from the returned solution rather than read off it, so the two are free to
# disagree.
function true_relative_residual(sim, u)
    A, b = sim.prob.A, sim.prob.b
    x64 = Float64.(u)
    r64 = similar(x64)
    mul!(r64, A, x64)
    r64 .= b .- r64
    return norm(r64) / Float64(norm(b))
end

base_solve(sim) = solve(sim.prob, KrylovJL_CG(); reltol=1.0f-6, abstol=0.0f0)

@testset "a Float32 host solve is refined by default" begin
    # The element type of `b` is what selects refinement, so a host Float32
    # problem must take the same branch a device one does.
    sim = float32_host_problem(REFINE_IMAGE)
    @test eltype(sim.prob.b) === Float32
    @test _refines_by_default(eltype(sim.prob.b))

    plain = solve(sim, KrylovJL_CG(); refine=false)
    refined = solve(sim, KrylovJL_CG())
    # Float32 CG stops on a recursively-updated residual that has drifted from
    # `b - A*x`, so it reports success while the true residual is still an order
    # of magnitude above the tolerance it claims to have met.
    @test Symbol(plain.retcode) === :Success
    @test true_relative_residual(sim, plain.u) > 1.0f-6
    # Refinement is worth having only if it recovers most of that gap.
    @test true_relative_residual(sim, refined.u) < true_relative_residual(sim, plain.u) / 5
    # And it costs iterations, which is how a silently-skipped refinement would
    # otherwise be indistinguishable from a working one.
    @test refined.iters > plain.iters
end

@testset "the reported residual describes the vector handed back" begin
    # `_refine` carries a Float64 iterate internally and narrows it to Float32
    # before returning. Reporting the iterate's residual would understate the
    # answer's by the narrowing error — the same class of defect refinement
    # exists to remove. The comparison is exact because the oracle above repeats
    # the function's own arithmetic in the same process.
    sim = float32_host_problem(REFINE_IMAGE)
    refined = solve(sim, KrylovJL_CG())
    @test refined.resid isa Base.RefValue
    @test refined.resid[] == true_relative_residual(sim, refined.u)
end

@testset "refinement hands the borrowed cache back as it found it" begin
    # The rounds point the cache's right-hand side at a scratch vector and
    # replace its tolerances for the correction equation. A caller who reuses
    # the cache afterwards — which is the whole reason refinement borrows it
    # rather than building its own — must not inherit any of those changes.
    sim = float32_host_problem(REFINE_IMAGE)
    sol = base_solve(sim)
    cache = sol.cache
    b_before, reltol_before, abstol_before = cache.b, cache.reltol, cache.abstol
    _refine(sol, sim, KrylovJL_CG())
    @test cache.b === b_before
    @test cache.reltol === reltol_before
    @test cache.abstol === abstol_before
end

@testset "a zero right-hand side is returned unrefined rather than as NaN" begin
    # Every residual in `_refine` is measured relative to `‖b‖`, and NaN
    # compares false against the shrink test, so without the guard all eight
    # rounds would run and report NaN for an answer that is exactly right.
    # A pore space touching neither Dirichlet face is the natural way to get
    # there: nothing is folded into the RHS, so `b` is exactly zero.
    img = falses(16, 8, 8)
    img[3:14, 3:6, 3:6] .= true
    sim = float32_host_problem(img)
    @test iszero(norm(sim.prob.b))

    sol = base_solve(sim)
    refined = _refine(sol, sim, KrylovJL_CG())
    @test refined === sol
    @test !isnan(refined.resid[])
end

@testset "a starved correction fails loudly and leaves no worse an answer" begin
    # Capping the cache at one iteration makes every correction return MaxIters.
    # Two things must follow: the failure reaches the caller as a retcode rather
    # than as a quietly-degraded solution, and the vector returned is still no
    # worse than the base solve it was built from.
    sim = float32_host_problem(REFINE_IMAGE)
    sol = base_solve(sim)
    resid_before = true_relative_residual(sim, sol.u)
    sol.cache.maxiters = 1
    # Each starved round logs a max-iterations warning. Those are left to print:
    # silencing them needs the `Logging` stdlib, which is in neither the package
    # deps nor the test target, and they are honest output of a solve this test
    # deliberately cripples.
    refined = _refine(sol, sim, KrylovJL_CG())
    @test Symbol(refined.retcode) === :Failure
    @test refined.stats.solved == false
    @test true_relative_residual(sim, refined.u) <= resid_before
end

@testset "success requires the returned vector to meet the requested residual" begin
    sim = float32_host_problem(REFINE_IMAGE)
    sol = base_solve(sim)
    sol.cache.reltol = eps(Float32)

    refined = _refine(sol, sim, KrylovJL_CG(); rounds=0)
    @test refined.resid[] > sol.cache.reltol
    @test Symbol(refined.retcode) === :Failure
    @test refined.stats.solved == false
end

@testset "a weak correction continues while the true residual improves" begin
    target = 2.5f-7
    short_sim = float32_host_problem(REFINE_IMAGE)
    short = base_solve(short_sim)
    short.cache.reltol = target
    stopped = _refine(short, short_sim, KrylovJL_CG(); rounds=1, shrink=0.45)

    full_sim = float32_host_problem(REFINE_IMAGE)
    full = base_solve(full_sim)
    full.cache.reltol = target
    continued = _refine(full, full_sim, KrylovJL_CG(); shrink=0.45)

    @test stopped.resid[] > target
    @test continued.resid[] <= target
    @test continued.iters > stopped.iters
    @test Symbol(continued.retcode) === :Success
end

@testset "a stalled loose correction falls back to the conservative tolerance" begin
    failed_sim = float32_host_problem(REFINE_IMAGE)
    failed = base_solve(failed_sim)
    without_fallback = _refine(
        failed, failed_sim, KrylovJL_CG(); correction_reltol=1.0f0,
    )

    recovered_sim = float32_host_problem(REFINE_IMAGE)
    recovered = base_solve(recovered_sim)
    with_fallback = _refine(
        recovered, recovered_sim, KrylovJL_CG();
        correction_reltol=1.0f0, fallback_reltol=0.5f0,
    )

    @test Symbol(without_fallback.retcode) === :Failure
    @test Symbol(with_fallback.retcode) === :Success
    @test with_fallback.resid[] <= recovered.cache.reltol
    @test with_fallback.iters > without_fallback.iters
end

@testset "absolute tolerance can satisfy the solver contract" begin
    sim = float32_host_problem(REFINE_IMAGE)
    sol = base_solve(sim)
    sol.cache.reltol = eps(Float32)
    abs_resid = true_relative_residual(sim, sol.u) * Float64(norm(sim.prob.b))
    sol.cache.abstol = Float32(1.01 * abs_resid)

    refined = _refine(sol, sim, KrylovJL_CG(); rounds=0)
    @test refined.resid[] > sol.cache.reltol
    @test refined.resid[] * Float64(norm(sim.prob.b)) <= sol.cache.abstol
    @test Symbol(refined.retcode) === :Success
    @test refined.stats.solved == true
end

@testset "relative and absolute tolerances contribute together" begin
    sim = float32_host_problem(REFINE_IMAGE)
    sol = base_solve(sim)
    rel_resid = true_relative_residual(sim, sol.u)
    abs_resid = rel_resid * Float64(norm(sim.prob.b))
    sol.cache.reltol = Float32(0.6 * rel_resid)
    sol.cache.abstol = Float32(0.5 * abs_resid)

    refined = _refine(sol, sim, KrylovJL_CG(); rounds=0)
    @test rel_resid > sol.cache.reltol
    @test abs_resid > sol.cache.abstol
    @test Symbol(refined.retcode) === :Success
    @test refined.stats.solved == true
end

@testset "corrections do not inherit the outer absolute tolerance" begin
    sim = float32_host_problem(REFINE_IMAGE)
    sol = base_solve(sim)
    base_iters = sol.iters
    base_abs_resid = true_relative_residual(sim, sol.u) * Float64(norm(sim.prob.b))
    sol.cache.reltol = eps(Float32)
    sol.cache.abstol = Float32(0.75 * base_abs_resid)

    refined = _refine(sol, sim, KrylovJL_CG())
    refined_abs_resid = refined.resid[] * Float64(norm(sim.prob.b))
    @test base_abs_resid > sol.cache.abstol
    @test refined_abs_resid <= sol.cache.abstol
    @test refined.iters > base_iters
    @test Symbol(refined.retcode) === :Success
end

@testset "the returned stats describe the base solve, not the last correction" begin
    # `sol.stats` is the workspace Krylov mutates in place, so each correction
    # overwrites it. Handing it back unchanged would report the last correction's
    # three iterations as the whole solve's count.
    sim = float32_host_problem(REFINE_IMAGE)
    plain = solve(sim, KrylovJL_CG(); refine=false)
    refined = solve(sim, KrylovJL_CG())

    @test refined.stats.niter == plain.iters
    @test refined.iters > refined.stats.niter
    @test refined.stats !== refined.cache.cacheval.stats
    @test occursin("refined", refined.stats.status)
    # `solved` has to agree with the retcode, or the two summaries of the same
    # solve contradict each other.
    @test refined.stats.solved == (Symbol(refined.retcode) === :Success)
end
