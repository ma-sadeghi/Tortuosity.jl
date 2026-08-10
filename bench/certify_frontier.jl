# Certifies the largest cubic image the matrix-free path can solve on this
# device: peak device memory, iteration count, wall time, tau and a true-residual
# check, one fresh process per size.
#
# Usage
# -----
#   julia --project=bench bench/certify_frontier.jl 1000
#   julia --project=bench bench/certify_frontier.jl 1100 --no-precond
#   julia --project=bench bench/certify_frontier.jl 1100 --generate-only
#
# A peak-memory number is only meaningful in a process that has allocated
# nothing else, so this script measures exactly one size and exits. Running two
# sizes in one process would report the high-water mark of the first.
#
# Flags:
#   --no-precond     solve without the two-level preconditioner
#   --generate-only  build and cache the blob fixture, measure nothing
#   --axis=x         transport axis (default x)
#   --reltol=1e-6    Krylov relative tolerance
#   --assembled      use the assembled path instead (expected to OOM past ~850)

using CUDA
using ImageFiltering
using LinearAlgebra
using LinearSolve
using Printf
using Tortuosity
using Tortuosity: Imaginator

const CACHE = get(ENV, "TORTUOSITY_BENCH_CACHE", joinpath(tempdir(), "tortuosity_bench_blobs"))
const POROSITY = 0.5
const BLOBINESS = 1.0
const SEED = 42

fixture_path(n) = joinpath(CACHE, "blobs_n$(n)_p$(POROSITY)_b$(BLOBINESS)_seed$(SEED).raw")

function cached_blobs(n)
    path = fixture_path(n)
    # A cache entry is only usable at its full size. An interrupted write leaves
    # a short file that `isfile` still accepts and `read!` then fails on for
    # good, so check the length and regenerate rather than inherit the stump.
    if isfile(path) && filesize(path) == n^3
        img = Array{Bool}(undef, n, n, n)
        read!(path, img)
        return img
    end
    isfile(path) && @warn "discarding truncated fixture ($(filesize(path)) of $(n^3) bytes)" path
    @info "Generating $(n)^3 fixture (this allocates a Float64 grid of $(round(8 * n^3 / 2^30; digits=1)) GiB)"
    img = Imaginator.blobs(;
        shape=(n, n, n), porosity=Float32(POROSITY), blobiness=Int(BLOBINESS), seed=SEED
    )
    mkpath(CACHE)
    # Write beside the target and rename, so the cached path only ever exists
    # complete — an interrupted run then costs a regeneration, not a wedge.
    tmp = path * ".partial"
    open(tmp, "w") do io
        write(io, img)
    end
    mv(tmp, path; force=true)
    return Array(img)
end

# Device memory is sampled from a task rather than read at the end, because the
# peak lands inside the solve and nothing afterwards remembers it.
mutable struct PeakMonitor
    task::Task
    stop::Threads.Atomic{Bool}
    peak::Threads.Atomic{Int}
end

function start_peak_monitor()
    stop = Threads.Atomic{Bool}(false)
    peak = Threads.Atomic{Int}(0)
    task = Threads.@spawn begin
        while !stop[]
            used = CUDA.total_memory() - CUDA.available_memory()
            used > peak[] && (peak[] = used)
            sleep(0.001)
        end
    end
    return PeakMonitor(task, stop, peak)
end

function stop_peak_monitor(m::PeakMonitor)
    m.stop[] = true
    wait(m.task)
    return m.peak[]
end

gib(x) = x / 2^30

function main(args)
    sizes = [parse(Int, a) for a in args if all(isdigit, a)]
    isempty(sizes) && (println("give exactly one cube size, e.g. 1000"); return)
    length(sizes) == 1 || error("measure one size per process — peak memory is per-process")
    n = sizes[1]
    precond = !("--no-precond" in args)
    assembled = "--assembled" in args
    axis = Symbol(something(findfirst(a -> startswith(a, "--axis="), args), 0) == 0 ? "x" :
                  split(args[findfirst(a -> startswith(a, "--axis="), args)], "=")[2])
    ridx = findfirst(a -> startswith(a, "--reltol="), args)
    reltol = isnothing(ridx) ? 1.0f-6 : parse(Float32, split(args[ridx], "=")[2])

    img = cached_blobs(n)
    "--generate-only" in args && (println("cached $(fixture_path(n))"); return)

    nnodes = count(img)
    @printf("n=%d  nvoxels=%d  nnodes=%d  porosity=%.4f  axis=%s  path=%s  precond=%s\n",
            n, length(img), nnodes, nnodes / length(img), axis,
            assembled ? "assembled" : "matrix-free", precond)

    base = CUDA.total_memory() - CUDA.available_memory()
    mon = start_peak_monitor()

    t_setup = @elapsed begin
        sim = SteadyDiffusionProblem(img; axis=axis, gpu=true, matrixfree=!assembled,
                                     warn_nonpercolating=false)
        CUDA.synchronize()
    end

    t_precond = 0.0
    Pl = nothing
    if precond
        t_precond = @elapsed begin
            Pl = two_level_preconditioner(sim)
            CUDA.synchronize()
        end
    end

    t_solve = @elapsed begin
        sol = isnothing(Pl) ? solve(sim.prob, KrylovJL_CG(); reltol=reltol) :
              solve(sim.prob, KrylovJL_CG(); Pl=Pl, reltol=reltol)
        CUDA.synchronize()
    end
    peak = stop_peak_monitor(mon)

    iters = try
        sol.iters
    catch
        -1
    end
    # Decision 5's guard: the recursive residual CG tracks can drift away from
    # the true one in Float32, and a converged retcode built on a drifted
    # residual is the failure mode that matters at this size.
    r = similar(sol.u)
    mul!(r, sim.prob.A, sol.u)
    r .= sim.prob.b .- r
    true_res = norm(r) / norm(sim.prob.b)

    u = Array(sol.u)
    tau = tortuosity(reconstruct_field(u, img), img; axis=axis)

    @printf("setup      %8.3f s\n", t_setup)
    @printf("precond    %8.3f s\n", t_precond)
    @printf("solve      %8.3f s   iters=%d   retcode=%s\n", t_solve, iters, sol.retcode)
    @printf("true resid %.3e (requested %.1e)\n", true_res, reltol)
    @printf("tau        %.6f   u: mean %.4f  extrema [%.4f, %.4f]\n",
            tau, sum(u) / length(u), minimum(u), maximum(u))
    @printf("device     base %.3f GiB  peak %.3f GiB  total %.3f GiB  headroom %.3f GiB\n",
            gib(base), gib(peak), gib(CUDA.total_memory()), gib(CUDA.total_memory() - peak))
    @printf("bytes per grid voxel (peak - base): %.2f\n", (peak - base) / length(img))
    return nothing
end

main(ARGS)
