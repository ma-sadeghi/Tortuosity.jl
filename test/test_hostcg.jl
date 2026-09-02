# Contract tests for the CPU-specialized conjugate-gradient solver and its cache.

using Test
using LinearAlgebra
using Tortuosity
using Tortuosity: HostCG, HOST_CG_MIN_NODES

@testset "HostCG agrees with KrylovJL_CG" begin
    img = trues(24, 22, 20)
    img[9:12, 7:15, 8:13] .= false
    for matrixfree in (false, true), precond in (:none, :auto)
        sim = SteadyDiffusionProblem(
            img; axis=:x, gpu=false, matrixfree, warn_nonpercolating=false,
        )
        host = solve(sim, HostCG(); precond, reltol=1e-10)
        krylov = solve(sim, KrylovJL_CG(); precond, reltol=1e-10)
        @test Symbol(host.retcode) === :Success
        @test host.iters == krylov.iters
        @test host.u ≈ krylov.u rtol=1e-11
        @test tortuosity(host.u, sim) ≈ tortuosity(krylov.u, sim) rtol=1e-11
    end
end

@testset "HostCG honors callbacks and iteration limits" begin
    img = trues(32, 12, 10)
    sim = SteadyDiffusionProblem(
        img; axis=:x, gpu=false, matrixfree=true, warn_nonpercolating=false,
    )
    iterations = Ref(0)
    callback = workspace -> begin
        iterations[] += 1
        @test workspace.x isa Vector{Float64}
        return iterations[] == 3
    end
    stopped = solve(sim, HostCG(; callback); precond=:none, reltol=1e-14)
    @test Symbol(stopped.retcode) === :Terminated
    @test stopped.iters == 3

    limited = solve(sim, HostCG(); precond=:none, reltol=1e-14, maxiters=2)
    @test Symbol(limited.retcode) === :MaxIters
    @test limited.iters == 2
    @test limited.cache.maxiters == 2

    p_at_callback = Ref{Vector{Float64}}()
    at_cap = solve(
        sim, HostCG(; callback=workspace -> (p_at_callback[] = copy(workspace.p); false));
        precond=:none, reltol=1e-14, maxiters=1,
    )
    @test p_at_callback[] == at_cap.cache.workspace.p

    default_cap = solve(sim, HostCG(); precond=:none, reltol=1e-14)
    @test default_cap.cache.maxiters == length(sim.prob.b)
    zero_cap = solve(sim, HostCG(); precond=:none, reltol=1e-14, maxiters=0)
    @test zero_cap.cache.maxiters == 2 * length(sim.prob.b)
end

@testset "HostCG cache can be reused" begin
    img = trues(24, 16, 12)
    sim = SteadyDiffusionProblem(
        img; axis=:x, gpu=false, matrixfree=true, warn_nonpercolating=false,
    )
    first = solve(sim, HostCG(); precond=:none, reltol=1e-10)
    first_u = copy(first.u)
    again = solve!(first.cache)
    @test again.u == first_u
    @test again.iters == first.iters

    first.cache.b = 2 .* sim.prob.b
    scaled = solve!(first.cache)
    @test scaled.u ≈ 2 .* first_u rtol=1e-11
end

@testset "HostCG is invariant to uniform diffusivity scaling" begin
    img = trues(22, 22, 22)
    for matrixfree in (false, true)
        sim = SteadyDiffusionProblem(
            img; axis=:x, D=1e-20, gpu=false, matrixfree,
            warn_nonpercolating=false,
        )
        sol = solve(sim, HostCG(); precond=:none)
        @test Symbol(sol.retcode) === :Success
        @test sol.iters > 0
        @test tortuosity(sol.u, sim) ≈ 1.0 rtol=1e-9
    end
end

@testset "the package default routes only worthwhile host systems" begin
    small = SteadyDiffusionProblem(
        trues(20, 20, 20); axis=:x, gpu=false, matrixfree=true,
        warn_nonpercolating=false,
    )
    large = SteadyDiffusionProblem(
        trues(22, 22, 22); axis=:x, gpu=false, matrixfree=true,
        warn_nonpercolating=false,
    )
    @test length(small.prob.b) < HOST_CG_MIN_NODES
    @test length(large.prob.b) >= HOST_CG_MIN_NODES
    @test solve(small).alg isa typeof(KrylovJL_CG())
    @test solve(large).alg isa HostCG
    @test solve(large, KrylovJL_CG()).alg isa typeof(KrylovJL_CG())

    prob32 = Tortuosity.LinearSolve.LinearProblem(large.prob.A, Float32.(large.prob.b))
    large32 = SteadyDiffusionProblem(large.img, large.axis, prob32)
    @test solve(large32; refine=false).alg isa typeof(KrylovJL_CG())
    @test_throws ArgumentError solve(large32, HostCG())
end
