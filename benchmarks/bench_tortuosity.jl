# Benchmark Tortuosity.jl on the shared image store.
#
#   julia --project=. bench_tortuosity.jl --device=gpu --operator=matrixfree
#   julia -t 1,1 --project=. bench_tortuosity.jl --device=cpu --measure=memory
#
# One device and one solver configuration per invocation, each writing its own
# CSV. Splitting them is not tidiness: a process that has already run large
# Float64 CPU sweeps carries a multi-gigabyte heap, and GPU sweeps that follow it
# in the same process stop being monotonic in the iteration count — a longer
# solve comes back faster, which is not something a solver can do.
#
# `--measure=time` reports wall time to each accuracy along the iteration ladder,
# traced from a single solve: a Krylov iterate is deterministic, so iterate k is
# the same vector whether the solve stopped there or carried on, and reading
# tortuosity off at each rung reports what one solve per rung would have
# reported. `--measure=memory` runs one fixed-length solve per case and reports
# sampled peak memory. They are separate because a timing must not be perturbed
# by a sampler and a peak cannot be measured without one.

include(joinpath(@__DIR__, "src", "BenchHarness.jl"))
using .BenchHarness
using CUDA
using Logging
using Printf
using Statistics
using Tortuosity

const BH = BenchHarness

args = parse_args(; options=["device", "operator", "precond", "measure", "timeout"])
cfg = load_config()

device = option(args, "device", "gpu")
device in ("cpu", "gpu") || error("--device must be cpu or gpu, got \"$device\"")
operator = option(args, "operator", "matrixfree")
operator in ("matrixfree", "assembled") || error("--operator must be matrixfree or assembled, got \"$operator\"")
precond = Symbol(option(args, "precond", "auto"))
precond in (:auto, :none) || error("--precond must be auto or none, got \"$precond\"")
measure = option(args, "measure", "time")
measure in ("time", "memory") || error("--measure must be time or memory, got \"$measure\"")

gpu = device == "gpu"
matrixfree = operator == "matrixfree"
axis = Symbol(cfg["campaign"]["axis"])
# The preconditioner is part of the configuration's identity, so a run without
# it lands in its own file rather than silently mixing with one that had it.
variant = operator * (precond == :none ? "-nopc" : "")
overwrite = flag(args, "overwrite")

gpu && !CUDA.functional() && error("--device=gpu but CUDA is not functional on this machine")

subdir = measure == "time" ? "timings" : "memory"
outpath = joinpath(cfg.root, "results", subdir, "tortuosity-$(device)-$(variant).csv")
columns = measure == "time" ? BH.TIMING_COLUMNS : BH.MEMORY_COLUMNS

target_error = Float64(cfg["sweep"]["target_error"])
n_repeats = Int(cfg["sweep"]["repeats"])
repeat_threshold = Float64(cfg["sweep"]["repeat_threshold_s"])
timeout_s = parse(Float64, option(args, "timeout", string(cfg["sweep"]["timeout_s"])))
ladder = BH.iteration_ladder(cfg)
cap_reltol = Float64(cfg["sweep"]["ladder"]["iters_mode_reltol"])
memory_iters = Int(cfg["memory"]["iters"])
sample_interval = Float64(cfg["memory"]["sample_interval_ms"])

manifest = read_manifest(cfg)
refs = read_references(cfg)
cases = select_cases(cfg, args)
measure == "memory" && (cases = BH.restrict_memory_blobiness(cfg, args, cases))

# Printed before anything is loaded or checked: this is how the orchestrator
# enumerates cases in order to run one process per case.
if flag(args, "list-cases")
    BH.list_cases(cases)
    exit(0)
end

missing_images = [c.id for c in cases if !haskey(manifest, c.id)]
isempty(missing_images) || error("no image for $(join(missing_images, ", ")) — run generate_images.jl first")

done = overwrite ? Set{String}() :
       (measure == "time" ? completed_cases(outpath; knob_name="iters") : BH.measured_cases(outpath))
solvable = [c for c in cases if manifest[c.id].nnodes > 0]
# A timing row is meaningless without ground truth to state its error against; a
# memory row needs no reference at all, which is what lets the memory stage cover
# sizes the accuracy stage cannot afford.
runnable = measure == "time" ? [c for c in solvable if haskey(refs, c.id)] : solvable
no_reference = [c.id for c in solvable if !haskey(refs, c.id)]
pending = [c for c in runnable if !(c.id in done)]

if flag(args, "dry-run")
    BH.report_plan(pending, "bench_tortuosity --device=$device --operator=$operator --measure=$measure";
                   skipped=sort(collect(done)))
    isempty(no_reference) || println("no reference yet, skipped: $(join(no_reference, ", "))")
    println("writing to $outpath")
    exit(0)
end

measure == "time" && BH.check_threads(cfg)
isempty(no_reference) || @warn "skipping cases with no ground truth — run compute_references.jl" cases = no_reference

"""Solve with warnings silenced.

Capping the iteration count makes LinearSolve warn that it stopped at the cap on
every rung. That is the sweep doing exactly what it was asked to; at eighteen
rungs per case it would be the bulk of the log.
"""
quiet_solve(prob, alg; kwargs...) =
    with_logger(ConsoleLogger(stderr, Logging.Error)) do
        solve(prob, alg; kwargs...)
    end

"""Build the problem for one case and run the solver for at most `maxiters` steps.

Construction sits inside the caller's measured region on purpose. Building the
coarse space for the preconditioner is a cost a user pays on every `solve(sim)`,
so charging it here is what keeps the comparison against another package honest.
"""
function solve_case(img, maxiters)
    sim = SteadyDiffusionProblem(
        img; axis=axis, gpu=gpu, matrixfree=matrixfree, warn_nonpercolating=false,
    )
    sol = quiet_solve(sim, KrylovJL_CG(); precond=precond, verbose=false,
                      maxiters=maxiters, reltol=cap_reltol)
    # Krylov methods read the residual norm back to the host every iteration, so
    # the device is very nearly synchronised already — but "very nearly" is not a
    # measurement, and the final update is not covered by it.
    gpu && CUDA.synchronize()
    tortuosity(sol.u, sim)
    return sim, sol
end

release!() = (GC.gc(true); gpu && CUDA.reclaim(); nothing)

# Monotonic, unlike `time()`. A campaign runs for hours, and a clock adjustment
# landing inside a measured region is indistinguishable from a slow solve. This
# is also what `@elapsed` uses, so the traced times stay on the same clock as
# every number this harness reported before.
now_s() = time_ns() / 1e9

row_prefix(case) = (; tool="tortuosity", device, variant, cpu_threads=Threads.nthreads(),
                    case_id=case.id, size=case.size, blobiness=case.blobiness,
                    porosity_target=case.porosity, porosity=manifest[case.id].porosity,
                    nnodes=manifest[case.id].nnodes,
                    host=gethostname(),
                    measured_at=BH.Dates.format(BH.Dates.now(), BH.Dates.dateformat"yyyy-mm-ddTHH:MM:SS"))

# ── Timing: trace the ladder from one solve ───────────────────────────

"""Trace one case's whole ladder from a single solve.

Construction sits inside the timed region for the same reason it does in
`solve_case`, so each rung's time is setup plus the iterations up to it. The
tortuosity evaluated at each rung is not work a user does, so its cost is
subtracted back out.

Stops at the first rung that meets the accuracy target, so a case that converges
early never pays for the rungs above it.
"""
function trace_case(img, tau_ref; rungs=ladder)
    rows, pending, excluded, k = NamedTuple[], copy(rungs), Ref(0.0), Ref(0)
    asked_to_stop = Ref(false)
    # Drain whatever the previous case left queued before starting the clock, for
    # the same reason every checkpoint reads it after a barrier: work that is
    # already in flight must not be charged to this solve.
    gpu && CUDA.synchronize()
    t0 = now_s()

    sim = SteadyDiffusionProblem(
        img; axis=axis, gpu=gpu, matrixfree=matrixfree, warn_nonpercolating=false,
    )
    # Construction is charged to every rung, so the padded rows below have to
    # carry it too — they re-solve, but they do not rebuild.
    setup_s = now_s() - t0

    cb = function (ws)
        k[] += 1
        (isempty(pending) || k[] != first(pending)) && return false
        popfirst!(pending)
        # Krylov reads its residual norm back every iteration, so the device is
        # nearly synchronised already — but "nearly" is not a measurement.
        gpu && CUDA.synchronize()
        mark = now_s()
        tau = tortuosity(ws.x, sim)
        elapsed = mark - t0 - excluded[]
        push!(rows, (; iters=k[], tau, time_s=elapsed))
        excluded[] += now_s() - mark
        asked_to_stop[] = abs(tau - tau_ref) / tau_ref <= target_error ||
                          elapsed > timeout_s || isempty(pending)
        return asked_to_stop[]
    end

    # `abstol=0` because the iteration count has to be the only stopping rule for
    # the trace to reach the rungs it was asked for. LinearSolve otherwise
    # defaults it to `sqrt(eps(T))`, which on `Float32` is 3.4e-4 — loose enough
    # to end the solve long before the cap, leaving most of the ladder
    # unreachable and every rung above the exit a copy of it.
    # `refine=false` because refinement reuses this solve's cache, and that cache
    # carries the algorithm — including `cb`. A refined solve would therefore fire
    # the callback on its correction rounds, where `ws.x` is a correction and the
    # tortuosity read off it is meaningless. The refined answer is measured below,
    # on its own, where it is the thing being asked for.
    sol = quiet_solve(sim, KrylovJL_CG(; callback=cb); precond=precond, verbose=false,
                      maxiters=last(rungs), reltol=cap_reltol, abstol=0.0, refine=false)
    gpu && CUDA.synchronize()

    # A solve that ended on its own — Krylov gives up once `Float32` rounding
    # stops the residual improving — leaves the rungs above it unvisited. They
    # are not missing measurements: asking for more iterations than the solver
    # will take returns this same answer at this same cost, which is exactly what
    # one solve per rung recorded for them.
    #
    # Read off the *final* iterate rather than the last checkpoint. The solve
    # runs on past the checkpoint before it stops, so padding with the checkpoint
    # would credit the run with neither the answer nor the cost it really had —
    # measured at 30% under on the time.
    #
    # On the GPU that final iterate is not what the package returns: a `Float32`
    # solve is refined against a `Float64` residual first, and the refined answer
    # is what a caller gets. The rungs above the exit are therefore padded with a
    # refined solve, timed on its own, because its wall clock is what reaching
    # that accuracy actually costs. Padding with the stalled iterate would report
    # a failure the shipped code does not have. On the CPU refinement never runs,
    # so nothing is re-solved and the original accounting stands.
    if !asked_to_stop[]
        local tau, elapsed
        if gpu
            CUDA.synchronize()
            t1 = now_s()
            refined = quiet_solve(sim, KrylovJL_CG(); precond=precond, verbose=false,
                                  maxiters=last(rungs), reltol=cap_reltol, abstol=0.0)
            CUDA.synchronize()
            mark = now_s()
            tau = tortuosity(refined.u, sim)
            # `setup_s` because every checkpointed rung's time is setup plus the
            # iterations up to it, and a padded rung has to be comparable with
            # them. The re-solve is timed on its own, but the problem it solves
            # was still built.
            elapsed = setup_s + (mark - t1)
            refined = nothing
        else
            mark = now_s()
            tau = tortuosity(sol.u, sim)
            elapsed = mark - t0 - excluded[]
        end
        for iters in pending
            push!(rows, (; iters, tau, time_s=elapsed))
        end
    end

    sim = sol = nothing
    release!()
    return rows
end

"""Sweep one case, writing a row per rung and stopping at the first conclusion."""
function sweep_case(w, case, img, tau_ref)
    traces = Vector{Vector{NamedTuple}}()
    for rep in 1:n_repeats
        rows = trace_case(img, tau_ref)
        # Thrown rather than returned: the caller's handler writes an error row.
        # Returning would leave the case with no row at all, and since resume keys
        # on a `stop_reason`, it would be retried on every resume with nothing
        # anywhere to say why.
        isempty(rows) && error("solve produced no checkpoints")
        push!(traces, rows)
        # A first repeat slower than the threshold abandons the rest: the median
        # of one is the value anyway at that size.
        rep == 1 && last(rows).time_s > repeat_threshold && break
    end

    # Repeats can stop one rung apart when a τ near the target lands either side
    # of it, so only rungs every repeat reached can be aggregated.
    n_rungs = minimum(length, traces)
    for rung in 1:n_rungs
        iters = traces[1][rung].iters
        all(t[rung].iters == iters for t in traces) || error("repeats disagree about the ladder")
        taus = [t[rung].tau for t in traces]
        times = [t[rung].time_s for t in traces]

        t_median = median(times)
        # Median over repeats, not the last value. The two-level preconditioner
        # accumulates its coarse operator with atomic float adds whose order is
        # not fixed across launches, so a GPU solve is not bit-reproducible and τ
        # moves by roughly the size of the accuracy target between runs.
        # Reporting one sample would make "did this case reach the target" partly
        # luck; the spread is written alongside so it can be quoted rather than
        # hidden. NaN when a single repeat could not measure a spread — a spread
        # of zero is the claim that repeats agreed exactly.
        tau_val = median(taus)
        spread = length(taus) > 1 ? (maximum(taus) - minimum(taus)) / tau_val : NaN
        rel_error = abs(tau_val - tau_ref) / tau_ref

        # `repeats_diverged` is the case that would otherwise leave no verdict at
        # all: repeats that disagree about whether the target was met stop at
        # different rungs, only their common prefix can be aggregated, and if the
        # target is not met inside it the loop ends with nothing written. Silence
        # then reads as "not measured" when what happened is that τ straddled the
        # target — which is exactly what the GPU preconditioner's atomics do.
        stop_reason = rel_error <= target_error ? "target_reached" :
                      t_median > timeout_s ? "timeout" :
                      iters == last(ladder) ? "ladder_exhausted" :
                      rung == n_rungs ? "repeats_diverged" : ""
        write_row!(w, (; row_prefix(case)..., knob_name="iters", knob=iters,
                       tau=tau_val, tau_ref, rel_error, time_s=t_median,
                       tau_spread=spread, repeats=length(times), stop_reason, note=""))
        @info @sprintf("  [%2d/%2d] iters=%-6d tau=%.4f err=%.2e spread=%.1e t=%.3fs %s",
                       rung, length(ladder), iters, tau_val, rel_error, spread, t_median, stop_reason)
        isempty(stop_reason) || return stop_reason
    end
    return "ladder_exhausted"
end

# ── Memory: one fixed-length solve, sampled ───────────────────────────

"""Measure one case's peak memory at a fixed iteration count."""
function probe_case(w, case, img)
    usage, status, note = nothing, "ok", ""
    try
        usage = with_peak_sampling(; interval_ms=sample_interval, gpu=gpu) do
            # Returned so the operator, the preconditioner and the Krylov
            # workspace are all still reachable when the closing sample is taken.
            solve_case(img, memory_iters)
        end
    catch e
        status = e isa CUDA.OutOfGPUMemoryError ? "oom" : "error"
        note = first(sprint(showerror, e), 200)
    end
    ok = usage !== nothing
    write_row!(w, (; row_prefix(case)..., iters=memory_iters,
                   time_s=ok ? usage.elapsed : NaN,
                   peak_rss_bytes=ok ? usage.peak_rss : 0,
                   baseline_rss_bytes=ok ? usage.baseline_rss : 0,
                   peak_device_bytes=ok ? usage.peak_device : 0,
                   pool_device_bytes=ok ? usage.peak_pool : 0,
                   status, note))
    if ok
        @info @sprintf("  %-6s t=%.2fs rss=%.2fGiB (base %.2f) device=%.2fGiB pool=%.2fGiB samples=%d",
                       status, usage.elapsed, usage.peak_rss / 2^30, usage.baseline_rss / 2^30,
                       usage.peak_device / 2^30, usage.peak_pool / 2^30, usage.samples)
    else
        @warn "  $status" note
    end
    usage = nothing
    release!()
    return status
end

# ── Warm up, then run ─────────────────────────────────────────────────

# Warmed on an image of its own rather than on a measured one: no reported number
# may include compilation, and re-solving a benchmark case to warm the path would
# double its cost for nothing. Julia specialises on types rather than on array
# sizes, so a small image compiles the code a large one runs. 64³ is the smallest
# that clears the pore count below which `precond=:auto` declines to build a
# coarse space, so this warms the preconditioner too when one is in use.
@info "Warming up" device operator precond measure threads = Threads.nthreads()
let warm = BH.build_image(cfg, Case(64, 1.0, 0.6))
    solve_case(warm, 5)
    # The timing stage runs a *different* path, and warming only the one above
    # would leave the first measured case carrying its compilation: a callback of
    # a fresh closure type respecializes `Krylov.cg!`, and the tortuosity read-off
    # inside it is compiled on first use. A τ target of `Inf` is never met, so the
    # short rung list is what ends this run.
    measure == "time" && trace_case(warm, Inf; rungs=[1, 2])
    release!()
end

record_environment(cfg; stage=measure, tool="tortuosity", device, variant,
                   accelerator=gpu ? string(CUDA.name(CUDA.device())) : "",
                   notes="ladder=$(length(ladder)) rungs, target=$target_error, timeout=$(timeout_s)s")

@info "Starting" measure n_pending = length(pending) already_done = length(done) out = outpath
# `replace_cases` so `--overwrite` drops only what this run re-measures. Passing
# `--cases=X --overwrite` after a long sweep must cost X's rows, not the sweep.
w = ResultsWriter(outpath, columns; overwrite=overwrite,
                  replace_cases=Set(c.id for c in pending))
try
    for (i, case) in enumerate(pending)
        entry = manifest[case.id]
        @info @sprintf("[%d/%d] %s  N=%d  blobiness=%.2f  porosity=%.4f  nodes=%d",
                       i, length(pending), case.id, case.size, case.blobiness,
                       entry.porosity, entry.nnodes)
        img = load_image(cfg, case)
        try
            measure == "time" ? sweep_case(w, case, img, refs[case.id]) : probe_case(w, case, img)
        catch e
            # One case failing must not take the rest of the run with it. At the
            # largest sizes the assembled path is *expected* to run out of device
            # memory, and that is a result worth recording rather than a crash
            # worth aborting on.
            kind = e isa CUDA.OutOfGPUMemoryError ? "out of device memory" : string(typeof(e))
            note = first(sprint(showerror, e), 200)
            @warn "[$i/$(length(pending))] $(case.id) failed — $kind" note
            measure == "time" && write_row!(w, (; row_prefix(case)..., knob_name="iters", knob=0,
                                                tau=NaN, tau_ref=refs[case.id], rel_error=NaN,
                                                time_s=NaN, tau_spread=NaN, repeats=0,
                                                stop_reason="error", note))
        end
        img = nothing
        release!()
    end
finally
    close_writer!(w)
end
@info "Done" out = outpath
