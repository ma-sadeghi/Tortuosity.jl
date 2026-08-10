# Matrix-free against assembled, side by side: apply cost, operator construction,
# end-to-end solve with and without the two-level preconditioner, peak GPU memory
# and the resulting tortuosity, at a range of cubic image sizes, emitted one CSV
# row per measurement so an OOM or a crash never loses the rows already collected.
#
# Usage
# -----
#   julia --project=bench bench/matrixfree_bench.jl              # 200 400 600 800
#   julia --project=bench bench/matrixfree_bench.jl 100 200      # explicit sizes
#   julia --project=bench bench/matrixfree_bench.jl --generate-only 1100
#   julia --project=bench bench/matrixfree_bench.jl --help
#
# Positional integer arguments are the cube edge lengths. Flags:
#   --paths=assembled,matrixfree  operator forms to measure (default both)
#   --passes=apply,setup,solve    measurements to take (default all three)
#   --no-precond                  skip the preconditioned solve
#   --precond                     take it (the default)
#   --axis=x                      transport direction
#   --reltol=1e-6                 solver tolerance; `auto` lets the package pick
#   --device=gpu                  gpu or cpu
#   --generate-only               cache the blob fixtures for SIZES and stop
#   --force                       re-measure cells already present in the CSV
#
# Environment overrides (flags win over env, env wins over the defaults):
#   TORTUOSITY_BENCH_CACHE     blob cache directory (see "Blob cache" below)
#   TORTUOSITY_BENCH_RESULTS   CSV path (default bench/results/matrixfree.csv)
#   TORTUOSITY_BENCH_PATHS, _PASSES, _PRECOND, _AXIS, _RELTOL, _DEVICE, _FORCE
#   TORTUOSITY_BENCH_SEED, _POROSITY, _BLOBINESS
#
# Output format
# -------------
# Long format, one row per (size, path, device, threads, pass, stage, repeat), so
# a stage can be added without breaking a reader. Four passes are recorded:
#
#   pass=apply          `mul!` alone, against a preallocated output vector and a
#                       random input, with the operator already built. Stage
#                       `mul`, one row per timed repeat; take the median.
#   pass=setup          building the operator on its own, after a pool reclaim.
#                       Stage `build`. Records nnodes, and nnz on the assembled
#                       path where the matrix has entries to count.
#   pass=solve          the whole path a user walks: `build`, `solve`, `post`
#                       (reconstruct_field + tortuosity). The pool is reclaimed
#                       before `build` and not again, so `peak_dev_bytes` on the
#                       `solve` stage is absolute device usage for the problem —
#                       the number the size ceiling is read off.
#   pass=solve_precond  the same with a two-level preconditioner, built in its
#                       own `precond` stage and handed to the solver. Kept a
#                       separate pass so the unpreconditioned numbers stay
#                       comparable whatever the preconditioner does.
#
# `tau` is recorded on the `post` stage of both solve passes, so the two operator
# forms can be compared for agreement and not only for speed. The two are not
# expected to agree bit for bit: the apply sums a row in a different association
# than a CSC `mul!` does, so at Float32 they part company in the last digits.
#
# Memory measurement
# ------------------
# `peak_dev_bytes` is the driver-reported device usage,
# `CUDA.total_memory() - CUDA.available_memory()`, sampled ~1 kHz from a
# background task for the duration of the stage. That quantity, not the sum of
# live Julia objects, is what decides whether an allocation throws: it includes
# blocks the CUDA.jl pool holds but has not handed out. The sampler cannot see a
# peak that opens and closes between two samples, so it is a lower bound;
# `retained_dev_bytes` (usage after the stage minus usage before it) is exact and
# complements it. `base_dev_bytes` is usage before the stage — subtract it for
# the stage's own cost.
#
# On the `apply` pass the figures cover the operator's construction and its
# warm-up applies, not the timed repeats: an apply allocates nothing, so those
# reach the same steady-state usage, and sampling the driver alongside a call
# that settles at a fraction of a millisecond costs more than the call.
#
# `maxrss_bytes` is `Sys.maxrss()`, the process-wide high-water mark of host
# memory. It is monotone over the whole process, so read it as "host memory had
# reached at least this much by the end of that stage".
#
# Blob cache
# ----------
# `Imaginator.blobs` costs ~60 s at 800^3, so generated images are cached as raw
# `Array{Bool}` bytes (one byte per voxel, N^3 bytes) keyed by the generation
# parameters, in the same location and under the same names `scaling_bench.jl`
# uses. The cache lives outside the repository — it is scratch, and 1000^3 alone
# is 1.0 GB. Override the location with TORTUOSITY_BENCH_CACHE.
#
# Generation is host-memory bound, not time bound: `blobs` works in `Float64`
# over the full grid, so the largest sizes need tens of gigabytes of free RAM.
# `--generate-only` runs the caching on its own, and a size whose estimate does
# not fit is reported and skipped rather than left to die mid-run.
#
# Deliberate choices worth knowing about
# --------------------------------------
# - `warn_nonpercolating=false` everywhere, for the reason `scaling_bench.jl`
#   gives: the check's automatic threshold would fire at some sizes and not at
#   others and make the construction times incomparable.
# - Images are not trimmed, so the fixtures are the ones the assembled baseline
#   was measured on and the two campaigns' numbers line up.
# - The `setup` pass and the `build` stage of the solve passes measure the same
#   call. The pass is the isolated number, taken after a reclaim with nothing
#   else resident; the stage is there so a solve pass stands on its own and so
#   the peak chain through the solve is unbroken.

using CUDA
using Dates
using ImageFiltering  # optional dependency, needed by Imaginator.blobs
using LinearAlgebra
using Printf
using Random
using SparseArrays
using Statistics
using Tortuosity
using Tortuosity: Imaginator

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_SIZES = [200, 400, 600, 800]
const DEFAULT_CACHE = joinpath(tempdir(), "tortuosity_bench_blobs")
const WARMUP_SIZE = 64
const ALL_PATHS = ["assembled", "matrixfree"]
const ALL_PASSES = ["apply", "setup", "solve"]

# CUSPARSE needs a long warm-up on this operator, far longer than a single
# discarded call: measured at 200^3 it runs 10-19 ms per `mul!` for the first
# seven or so calls and only then settles at 0.37 ms. Twenty-five discarded calls
# clear that transient on both paths; fifteen timed repeats then give a median
# that does not move between runs. Neither number is arbitrary — shortening the
# warm-up reports the transient as if it were the steady-state cost.
const APPLY_WARMUP = 25
const APPLY_REPS = 15

# `Imaginator.blobs` holds the grid in `Float64`: the noise field, the blurred
# copy and the normalised copy are 8 bytes per voxel each, and the Gaussian
# filter carries its own buffers on top. Four full-grid `Float64` arrays is the
# floor for the estimate, not the worst case.
const BLOB_HOST_BYTES_PER_VOXEL = 4 * sizeof(Float64)

# `path` is part of the key, not a footnote: the whole point of the file is that
# an assembled row and a matrix-free row at the same size are different
# measurements. So is `threads`, for the reason `scaling_bench.jl` records it —
# the CPU apply is KernelAbstractions-threaded and CPU SpMV is not.
const CSV_COLUMNS = [
    "run_id", "timestamp", "git_sha", "n", "nvoxels", "path", "device", "threads",
    "pass", "stage", "rep", "status", "wall_s", "peak_dev_bytes", "base_dev_bytes",
    "retained_dev_bytes", "maxrss_bytes", "nnodes", "nnz", "iters", "retcode",
    "tau", "note",
]

# --- Options ---------------------------------------------------------------

const USAGE = """
julia --project=bench bench/matrixfree_bench.jl [SIZES...] [FLAGS]

  SIZES                         cube edge lengths (default $(join(DEFAULT_SIZES, ' ')))
  --paths=assembled,matrixfree  operator forms to measure
  --passes=apply,setup,solve    measurements to take
  --no-precond                  skip the preconditioned solve
  --precond                     take it (the default)
  --axis=x                      transport direction
  --reltol=1e-6                 solver tolerance; `auto` lets the package pick
  --device=gpu                  gpu or cpu
  --generate-only               cache the blob fixtures for SIZES and stop
  --force                       re-measure cells already present in the CSV

Env overrides: TORTUOSITY_BENCH_{CACHE,RESULTS,PATHS,PASSES,PRECOND,AXIS,RELTOL,
DEVICE,FORCE,SEED,POROSITY,BLOBINESS}. See the header comment for the output
format.
"""

env(key, default) = get(ENV, "TORTUOSITY_BENCH_$(key)", default)
_flag(s) = startswith(s, "--")
_list(s) = [String(strip(x)) for x in split(s, ',') if !isempty(strip(x))]

function parse_args(args)
    sizes = Int[]
    opts = Dict{String,String}(
        "paths" => env("PATHS", join(ALL_PATHS, ',')),
        "passes" => env("PASSES", join(ALL_PASSES, ',')),
        "precond" => env("PRECOND", "1"),
        "axis" => env("AXIS", "x"),
        "reltol" => env("RELTOL", "auto"),
        "device" => env("DEVICE", "gpu"),
        "force" => env("FORCE", "0"),
        "generate-only" => "0",
    )
    for a in args
        if a in ("--help", "-h")
            print(USAGE)
            exit(0)
        elseif a == "--precond"
            opts["precond"] = "1"
        elseif a == "--no-precond"
            opts["precond"] = "0"
        elseif a == "--generate-only"
            opts["generate-only"] = "1"
        elseif a == "--force"
            opts["force"] = "1"
        elseif _flag(a) && occursin('=', a)
            k, v = split(a[3:end], '='; limit=2)
            # Values are checked below, but an unrecognised *key* would otherwise
            # be accepted and ignored — `--path=matrixfree` would silently run
            # the default set of both paths.
            haskey(opts, k) || error(
                "Unknown flag: --$(k) (expected one of $(join(sort(collect(keys(opts))), ", ")))"
            )
            opts[k] = v
        elseif _flag(a)
            error("Unknown flag: $a")
        else
            push!(sizes, parse(Int, a))
        end
    end
    isempty(sizes) && (sizes = copy(DEFAULT_SIZES))
    for p in _list(opts["paths"])
        p in ALL_PATHS || error("Unknown path: $p (expected one of $(join(ALL_PATHS, ", ")))")
    end
    for p in _list(opts["passes"])
        p in ALL_PASSES || error("Unknown pass: $p (expected one of $(join(ALL_PASSES, ", ")))")
    end
    return sort!(unique(sizes)), opts
end

# --- CSV -------------------------------------------------------------------

# Separators are replaced rather than quoted so that a plain `split(line, ',')`
# is always a correct parse — the summary reader below relies on that, and so
# will anyone who greps the file.
function csvfield(x)
    x === nothing && return ""
    x isa AbstractFloat && return isfinite(x) ? @sprintf("%.6g", x) : string(x)
    return replace(string(x), ',' => ';', '"' => '\'', '\n' => ' ', '\r' => ' ')
end

"""
Create the CSV if it is missing, and rotate it aside when its header is stale.

Appending a row with more fields than the header on disk would leave a file no
reader can parse, so a column added here retires the older results rather than
corrupting them.
"""
function ensure_csv(path)
    mkpath(dirname(path))
    header = join(CSV_COLUMNS, ",")
    if isfile(path) && filesize(path) > 0
        readline(path) == header && return nothing
        mv(path, "$(path).$(Dates.format(now(), "yyyymmdd-HHMMSS")).bak"; force=true)
        @warn "column layout changed; previous results moved aside" path
    end
    open(path, "w") do io
        println(io, header)
    end
    return nothing
end

"""Append one row and flush, so an OOM or a hard crash keeps every earlier row."""
function emit!(path, row::Dict{String,Any})
    open(path, "a") do io
        println(io, join((csvfield(get(row, c, nothing)) for c in CSV_COLUMNS), ","))
        flush(io)
    end
    return nothing
end

const TERMINAL_STAGE = Dict(
    "apply" => "mul", "setup" => "build", "solve" => "post", "solve_precond" => "post",
)

"""
Cells already measured to a deterministic outcome, so a re-run can skip them.

A cell counts as done only once its pass's *last* stage has been written — and
for the repeated `apply` pass, its last repeat. Every outcome the passes handle
themselves writes that row (a failure fills the stages and repeats after it in
as `skipped`), so the only way it is missing is that the process died part-way
through, which is the case that should be measured again rather than inherited.
"""
function completed_cells(path)
    done = Set{Tuple{Int,String,String,Int,String}}()
    isfile(path) || return done
    lines = readlines(path)
    length(lines) <= 1 && return done
    header = split(lines[1], ',')
    col = Dict(name => i for (i, name) in enumerate(header))
    for line in lines[2:end]
        f = split(line, ',')
        length(f) < length(header) && continue
        pass = f[col["pass"]]
        f[col["stage"]] == get(TERMINAL_STAGE, pass, "") || continue
        f[col["status"]] in ("ok", "oom", "oom_host", "error", "skipped") || continue
        pass == "apply" && parse(Int, f[col["rep"]]) != APPLY_REPS && continue
        push!(done, (
            parse(Int, f[col["n"]]), f[col["path"]], f[col["device"]],
            parse(Int, f[col["threads"]]), pass,
        ))
    end
    return done
end

# --- Device memory sampling ------------------------------------------------

device_used_bytes() = CUDA.functional() ? Int(CUDA.total_memory() - CUDA.available_memory()) : 0

mutable struct PeakMonitor
    peak::Int
    running::Bool
    task::Union{Nothing,Task}
end

function start_peak_monitor()
    m = PeakMonitor(device_used_bytes(), true, nothing)
    m.task = Threads.@spawn begin
        while m.running
            u = device_used_bytes()
            u > m.peak && (m.peak = u)
            sleep(0.001)
        end
        u = device_used_bytes()
        u > m.peak && (m.peak = u)
    end
    return m
end

function stop_peak_monitor(m::PeakMonitor)
    m.running = false
    try
        wait(m.task)
    catch
        # A sampler that died still leaves a usable high-water mark.
    end
    return m.peak
end

# --- Measurement -----------------------------------------------------------

"""Map a thrown exception onto a CSV status plus a short, single-line note."""
function classify(err)
    msg = try
        first(sprint(showerror, err), 240)
    catch
        string(typeof(err))
    end
    msg = replace(msg, r"\s+" => " ")
    if err isa CUDA.OutOfGPUMemoryError || occursin("OutOfGPUMemoryError", msg) ||
       occursin("out of GPU memory", msg)
        return "oom", msg
    elseif err isa OutOfMemoryError || occursin("OutOfMemoryError", msg)
        return "oom_host", msg
    end
    return "error", msg
end

"""
Run `f`, returning its value alongside wall time and device-memory figures.

`reclaim_first` returns pooled device memory to the driver before the baseline
is read; use it once at the head of a pass, not between the stages of one, or
the stage peaks stop being comparable to each other.
"""
function measure(f::Function; gpu::Bool, reclaim_first::Bool=false)
    if reclaim_first
        GC.gc(true)
        gpu && CUDA.functional() && CUDA.reclaim()
    end
    base = gpu ? device_used_bytes() : 0
    mon = gpu ? start_peak_monitor() : nothing
    val = nothing
    status = "ok"
    note = ""
    t0 = time_ns()
    try
        val = gpu ? CUDA.@sync(f()) : f()
    catch err
        status, note = classify(err)
    end
    wall = (time_ns() - t0) / 1e9
    peak = gpu ? stop_peak_monitor(mon) : 0
    after = gpu ? device_used_bytes() : 0
    return (;
        val, status, note, wall,
        peak=max(peak, after), base, retained=after - base, maxrss=Int(Sys.maxrss()),
    )
end

"""Drop everything the last pass left behind, so the next one starts from a floor."""
function release!(gpu::Bool)
    GC.gc(true)
    gpu && CUDA.functional() && CUDA.reclaim()
    return nothing
end

# --- Fixture ---------------------------------------------------------------

blob_name(n; porosity, blobiness, seed) = "blobs_n$(n)_p$(porosity)_b$(blobiness)_seed$(seed).raw"

"""
Report whether generating an `n^3` blob image is worth attempting on this host.

The estimate is a floor (see `BLOB_HOST_BYTES_PER_VOXEL`), so a size that clears
it can still run out — but a size that does not clear it will certainly fail,
and failing loudly here beats dying half an hour into an unattended run.
"""
function host_memory_ok(n)
    need = BLOB_HOST_BYTES_PER_VOXEL * n^3
    free = Int(Sys.free_memory())
    need <= free && return true
    @error "not generating $(n)^3: blob generation needs at least \
            $(round(need / 2^30; digits=1)) GiB of host RAM (a floor, not the \
            worst case) and $(round(free / 2^30; digits=1)) GiB is free. Free \
            memory and re-run with --generate-only $(n), or generate the \
            fixture on a larger host and drop the .raw file into the cache."
    return false
end

"""
Load the cached `n^3` blob image, generating and caching it if it is missing.

Returns `nothing` when the image is neither cached nor generatable here, so the
caller can skip that size and keep going rather than take the whole run down.
"""
function cached_blobs(n; porosity, blobiness, seed, cachedir)
    mkpath(cachedir)
    path = joinpath(cachedir, blob_name(n; porosity, blobiness, seed))
    nbytes = n^3
    if isfile(path) && filesize(path) == nbytes
        img = Array{Bool,3}(undef, n, n, n)
        open(io -> read!(io, img), path, "r")
        return img
    end
    host_memory_ok(n) || return nothing
    @info "generating $(n)^3 blob image (not cached yet); \
           $(round(nbytes / 2^30; digits=2)) GiB on disk when done"
    img = try
        Imaginator.blobs(; shape=(n, n, n), porosity=porosity, blobiness=blobiness, seed=seed)
    catch err
        status, msg = classify(err)
        @error "blob generation failed at $(n)^3 ($(status)): $(msg)"
        return nothing
    end
    tmp = path * ".tmp"
    open(io -> write(io, img), tmp, "w")
    mv(tmp, path; force=true)
    GC.gc(true)
    return img
end

"""Cache the fixtures for `sizes` and report what was and was not produced."""
function generate_only(sizes; porosity, blobiness, seed, cachedir)
    @info "fixture cache: $(cachedir)"
    for n in sizes
        path = joinpath(cachedir, blob_name(n; porosity, blobiness, seed))
        if isfile(path) && filesize(path) == n^3
            @info "$(n)^3 already cached — $(basename(path))"
            continue
        end
        t0 = time_ns()
        img = cached_blobs(n; porosity, blobiness, seed, cachedir)
        if img === nothing
            @warn "$(n)^3 NOT cached; see the message above"
        else
            @info "$(n)^3 cached in $(round((time_ns() - t0) / 1e9; digits=1)) s, \
                   porosity $(round(count(img) / length(img); digits=4))"
        end
        img = nothing
        GC.gc(true)
    end
    return nothing
end

# --- Passes ----------------------------------------------------------------

function base_row_for(base, path, pass)
    return merge(copy(base), Dict{String,Any}("path" => path, "pass" => pass))
end

function stage_row(base_row, stage, m; extra...)
    return merge(
        copy(base_row),
        Dict{String,Any}(
            "stage" => stage, "status" => m.status, "note" => m.note, "wall_s" => m.wall,
            "peak_dev_bytes" => m.peak, "base_dev_bytes" => m.base,
            "retained_dev_bytes" => m.retained, "maxrss_bytes" => m.maxrss,
        ),
        Dict{String,Any}(String(k) => v for (k, v) in extra),
    )
end

skipped_row(base_row, stage; rep=nothing) = merge(
    copy(base_row),
    Dict{String,Any}("stage" => stage, "status" => "skipped", "rep" => rep),
)

"""Build the operator the way a user does, with the bench's fixed conventions."""
function build_problem(img; gpu, matrixfree, axis)
    return SteadyDiffusionProblem(
        img; axis=axis, gpu=gpu, matrixfree=matrixfree, warn_nonpercolating=false,
    )
end

"""
`mul!` on its own: the operator is already built, the output vector preallocated
and the input random, so what is left is the apply and nothing else.

One row per timed repeat. The memory figures come from the construction and
warm-up block rather than from the repeats, for the reason the header comment
gives.
"""
function run_apply_pass!(csv, base, path; img, gpu, matrixfree, axis)
    br = base_row_for(base, path, "apply")
    # A failure before the repeats start has a duration, but it is not an apply
    # time — leave `wall_s` empty so nothing averages it in with the real ones.
    fail!(m) = begin
        emit!(csv, stage_row(br, "mul", merge(m, (; wall=nothing)); rep=1))
        for r in 2:APPLY_REPS
            emit!(csv, skipped_row(br, "mul"; rep=r))
        end
        release!(gpu)
        return m.status
    end

    prepared = measure(; gpu=gpu, reclaim_first=true) do
        sim = build_problem(img; gpu, matrixfree, axis)
        A, b = sim.prob.A, sim.prob.b
        x = similar(b)
        Random.rand!(x)
        y = fill!(similar(b), zero(eltype(b)))
        for _ in 1:APPLY_WARMUP
            mul!(y, A, x)
        end
        return sim, A, x, y
    end
    prepared.status == "ok" || return fail!(prepared)
    sim, A, x, y = prepared.val
    nnodes = length(sim.prob.b)

    # Nothing samples memory while the repeats run. The sampler polls the driver
    # for its reading, and against an apply that settles at a fraction of a
    # millisecond that poll is not free — it inflated the assembled path by more
    # than a millisecond a call, which is larger than the quantity being
    # measured. The figures carried on these rows come from `prepared` instead:
    # they cover construction and the warm-up applies, which reach the same
    # steady-state device usage a timed apply does, an apply allocating nothing.
    walls = Float64[]
    status, note = "ok", ""
    for _ in 1:APPLY_REPS
        t0 = time_ns()
        try
            gpu ? CUDA.@sync(mul!(y, A, x)) : mul!(y, A, x)
        catch err
            status, note = classify(err)
            break
        end
        push!(walls, (time_ns() - t0) / 1e9)
    end
    block = (;
        status, note, peak=prepared.peak, base=prepared.base,
        retained=prepared.retained, maxrss=Int(Sys.maxrss()),
    )
    for (rep, wall) in enumerate(walls)
        emit!(csv, stage_row(br, "mul", merge(block, (; wall)); rep=rep, nnodes=nnodes))
    end
    for rep in (length(walls) + 1):APPLY_REPS
        emit!(csv, stage_row(br, "mul", merge(block, (; wall=nothing)); rep=rep, nnodes=nnodes))
    end

    sim, A, x, y = nothing, nothing, nothing, nothing
    release!(gpu)
    return status
end

"""Operator construction on its own, measured from a reclaimed pool."""
function run_setup_pass!(csv, base, path; img, gpu, matrixfree, axis)
    br = base_row_for(base, path, "setup")
    m = measure(; gpu=gpu, reclaim_first=true) do
        build_problem(img; gpu, matrixfree, axis)
    end
    if m.status != "ok"
        emit!(csv, stage_row(br, "build", m; rep=1))
        release!(gpu)
        return m.status
    end
    sim = m.val
    # Only the assembled form has stored entries to count; leaving the column
    # empty on the matrix-free rows is the honest answer, not a missing one.
    nnz_A = matrixfree ? nothing : length(SparseArrays.nonzeros(sim.prob.A))
    emit!(csv, stage_row(br, "build", m; rep=1, nnodes=length(sim.prob.b), nnz=nnz_A))
    sim = nothing
    release!(gpu)
    return m.status
end

"""
The whole path a user walks: build, optionally build a preconditioner, solve,
postprocess to `tau`.

Emits one row per stage; a stage that fails marks the stages after it `skipped`
so the CSV records the whole cell rather than trailing off. The pool is
reclaimed once, before `build`, so the peaks accumulate through the pass and the
`solve` stage's peak is absolute device usage for the problem.
"""
function run_solve_pass!(csv, base, path; img, gpu, matrixfree, axis, precond, reltol)
    pass = precond ? "solve_precond" : "solve"
    br = base_row_for(base, path, pass)
    stages = precond ? ["build", "precond", "solve", "post"] : ["build", "solve", "post"]
    function bail!(stage, m; extra...)
        emit!(csv, stage_row(br, stage, m; rep=1, extra...))
        for s in stages[(findfirst(==(stage), stages) + 1):end]
            emit!(csv, skipped_row(br, s; rep=1))
        end
        release!(gpu)
        return m.status
    end

    built = measure(; gpu=gpu, reclaim_first=true) do
        build_problem(img; gpu, matrixfree, axis)
    end
    built.status == "ok" || return bail!("build", built)
    sim = built.val
    nnz_A = matrixfree ? nothing : length(SparseArrays.nonzeros(sim.prob.A))
    emit!(csv, stage_row(br, "build", built; rep=1, nnodes=length(sim.prob.b), nnz=nnz_A))

    Pl = :none
    if precond
        made = measure(; gpu=gpu) do
            two_level_preconditioner(sim)
        end
        made.status == "ok" || return bail!("precond", made)
        Pl = made.val
        emit!(csv, stage_row(br, "precond", made;
            rep=1, note=Pl === nothing ? "no coarse space" : "nc=$(Pl.nc) block=$(Pl.block)"))
        # A coarse space that could not be built is not a preconditioned run.
        Pl === nothing && (Pl = :none)
    end

    solved = measure(; gpu=gpu) do
        solve(sim, KrylovJL_CG(); precond=Pl, reltol=reltol, verbose=false)
    end
    solved.status == "ok" || return bail!("solve", solved)
    sol = solved.val
    iters = hasproperty(sol, :iters) ? sol.iters :
            (hasproperty(sol, :stats) ? sol.stats.niter : nothing)
    emit!(csv, stage_row(br, "solve", solved; rep=1, iters=iters, retcode=sol.retcode))

    tau = Ref{Any}(nothing)
    post = measure(; gpu=gpu) do
        c = reconstruct_field(sol.u, img)
        tau[] = tortuosity(c, img; axis=axis)
        return tau[]
    end
    emit!(csv, stage_row(br, "post", post; rep=1, tau=tau[]))

    sim, sol, Pl = nothing, nothing, nothing
    release!(gpu)
    return post.status
end

# --- Summary ---------------------------------------------------------------

function read_rows(path)
    lines = readlines(path)
    header = split(lines[1], ',')
    rows = Dict{String,String}[]
    for line in lines[2:end]
        f = split(line, ',')
        length(f) < length(header) && continue
        push!(rows, Dict(String(header[i]) => String(f[i]) for i in eachindex(header)))
    end
    return rows
end

_num(s) = isempty(s) ? nothing : tryparse(Float64, s)
gib(b) = b === nothing ? nothing : b / 2^30
_med(rs, col) = (v = Float64[x for x in (_num(r[col]) for r in rs) if x !== nothing];
                 isempty(v) ? nothing : median(v))
_maxof(rs, col) = (v = Float64[x for x in (_num(r[col]) for r in rs) if x !== nothing];
                   isempty(v) ? nothing : maximum(v))
_status(rs) = (s = unique([r["status"] for r in rs]);
               "oom" in s ? "OOM" : ("oom_host" in s ? "OOM_HOST" :
               ("error" in s ? "ERROR" : ("skipped" in s ? "PARTIAL" : "ok"))))
_fmt(x; digits=3) = x === nothing ? "-" : @sprintf("%.*f", digits, x)
_int(x) = x === nothing ? "-" : @sprintf("%.0f", x)

function cells_of(rows, pass)
    sel = filter(r -> r["pass"] == pass, rows)
    keys_ = unique([(parse(Int, r["n"]), r["path"], r["device"]) for r in sel])
    sort!(keys_; by=k -> (k[3], k[1], k[2]))
    return sel, keys_
end

function print_apply_summary(rows)
    sel, keys_ = cells_of(rows, "apply")
    isempty(sel) && return nothing
    println()
    println("=== pass=apply — mul! alone, median of $(APPLY_REPS) repeats after $(APPLY_WARMUP) discarded ===")
    @printf("%6s %12s %7s %8s %12s %11s %11s\n",
        "N", "path", "device", "status", "nnodes", "apply_ms", "spread_ms")
    for (n, path, dev) in keys_
        cell = filter(r -> parse(Int, r["n"]) == n && r["path"] == path && r["device"] == dev, sel)
        w = Float64[x for x in (_num(r["wall_s"]) for r in cell) if x !== nothing]
        med = isempty(w) ? nothing : 1e3 * median(w)
        spread = isempty(w) ? nothing : 1e3 * (maximum(w) - minimum(w))
        @printf("%6d %12s %7s %8s %12s %11s %11s\n",
            n, path, dev, _status(cell), _int(_med(cell, "nnodes")),
            _fmt(med), _fmt(spread))
    end
    return nothing
end

function print_setup_summary(rows)
    sel, keys_ = cells_of(rows, "setup")
    isempty(sel) && return nothing
    println()
    println("=== pass=setup — operator construction alone, from a reclaimed pool ===")
    @printf("%6s %12s %7s %8s %12s %14s %9s %10s %10s\n",
        "N", "path", "device", "status", "nnodes", "nnz", "setup_s", "peak_GiB", "held_GiB")
    for (n, path, dev) in keys_
        cell = filter(r -> parse(Int, r["n"]) == n && r["path"] == path && r["device"] == dev, sel)
        @printf("%6d %12s %7s %8s %12s %14s %9s %10s %10s\n",
            n, path, dev, _status(cell), _int(_med(cell, "nnodes")), _int(_med(cell, "nnz")),
            _fmt(_med(cell, "wall_s")), _fmt(gib(_maxof(cell, "peak_dev_bytes"))),
            _fmt(gib(_med(cell, "retained_dev_bytes"))))
    end
    return nothing
end

function print_solve_summary(rows)
    sel = filter(r -> r["pass"] in ("solve", "solve_precond"), rows)
    isempty(sel) && return nothing
    keys_ = unique([(parse(Int, r["n"]), r["path"], r["device"], r["pass"]) for r in sel])
    sort!(keys_; by=k -> (k[3], k[4], k[1], k[2]))
    println()
    println("=== pass=solve / solve_precond — end to end ===")
    println("peak = max device usage over the stages of the pass; e2e = their sum")
    @printf("%6s %12s %7s %14s %9s %9s %9s %9s %9s %9s %10s %7s %9s %10s\n",
        "N", "path", "device", "pass", "status", "build_s", "prec_s", "solve_s",
        "post_s", "e2e_s", "peak_GiB", "iters", "retcode", "tau")
    for (n, path, dev, pass) in keys_
        cell = filter(
            r -> parse(Int, r["n"]) == n && r["path"] == path && r["device"] == dev &&
                 r["pass"] == pass, sel,
        )
        stage_of(s) = filter(r -> r["stage"] == s, cell)
        t_build = _med(stage_of("build"), "wall_s")
        t_prec = _med(stage_of("precond"), "wall_s")
        t_solve = _med(stage_of("solve"), "wall_s")
        t_post = _med(stage_of("post"), "wall_s")
        e2e = sum(x for x in (t_build, t_prec, t_solve, t_post) if x !== nothing; init=0.0)
        retcodes = unique([r["retcode"] for r in stage_of("solve") if !isempty(r["retcode"])])
        @printf("%6d %12s %7s %14s %9s %9s %9s %9s %9s %9s %10s %7s %9s %10s\n",
            n, path, dev, pass, _status(cell), _fmt(t_build), _fmt(t_prec), _fmt(t_solve),
            _fmt(t_post), _fmt(e2e), _fmt(gib(_maxof(cell, "peak_dev_bytes"))),
            _int(_med(stage_of("solve"), "iters")),
            isempty(retcodes) ? "-" : first(retcodes),
            _fmt(_med(stage_of("post"), "tau"); digits=6))
    end
    return nothing
end

function print_summary(path)
    rows = read_rows(path)
    isempty(rows) && return nothing
    print_apply_summary(rows)
    print_setup_summary(rows)
    print_solve_summary(rows)
    return nothing
end

# --- Driver ----------------------------------------------------------------

function git_sha()
    try
        return strip(read(`git -C $REPO_ROOT rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function write_env_record(path, run_id, opts)
    cols = ["run_id", "timestamp", "git_sha", "julia", "threads", "gpu_name",
            "gpu_total_bytes", "cuda_runtime", "apply_warmup", "apply_reps", "options"]
    mkpath(dirname(path))
    isfile(path) || open(io -> println(io, join(cols, ",")), path, "w")
    rec = Dict{String,Any}(
        "run_id" => run_id, "timestamp" => string(now()), "git_sha" => git_sha(),
        "julia" => string(VERSION), "threads" => Threads.nthreads(),
        "gpu_name" => CUDA.functional() ? CUDA.name(CUDA.device()) : "none",
        "gpu_total_bytes" => CUDA.functional() ? Int(CUDA.total_memory()) : 0,
        "cuda_runtime" => CUDA.functional() ? string(CUDA.runtime_version()) : "none",
        "apply_warmup" => APPLY_WARMUP, "apply_reps" => APPLY_REPS,
        "options" => join(("$k=$v" for (k, v) in sort(collect(opts))), " "),
    )
    open(io -> println(io, join((csvfield(rec[c]) for c in cols), ",")), path, "a")
    return nothing
end

# Warmup rows are compilation noise, not measurements: send them to a throwaway
# file rather than polluting the results CSV.
function devnull_csv()
    path = joinpath(tempdir(), "tortuosity_matrixfree_warmup.csv")
    ensure_csv(path)
    return path
end

"""
Compile every kernel the run will touch, at a size small enough to be free.

The preconditioner is warmed inside a `try`: a path it cannot build for yet must
not stop the paths it can, and the failure is recorded properly when the real
pass reaches it.
"""
function warmup(paths, passes, precond; gpu, axis, reltol)
    img = Imaginator.blobs(; shape=(WARMUP_SIZE, WARMUP_SIZE, WARMUP_SIZE),
                           porosity=0.5, blobiness=1.0, seed=1)
    csv = devnull_csv()
    base = Dict{String,Any}("n" => WARMUP_SIZE, "rep" => 1)
    for path in paths
        matrixfree = path == "matrixfree"
        "apply" in passes && run_apply_pass!(csv, base, path; img, gpu, matrixfree, axis)
        "setup" in passes && run_setup_pass!(csv, base, path; img, gpu, matrixfree, axis)
        if "solve" in passes
            run_solve_pass!(csv, base, path;
                img, gpu, matrixfree, axis, precond=false, reltol=reltol)
            precond && run_solve_pass!(csv, base, path;
                img, gpu, matrixfree, axis, precond=true, reltol=reltol)
        end
    end
    release!(gpu)
    return nothing
end

function main(args)
    sizes, opts = parse_args(args)
    seed = parse(Int, env("SEED", "42"))
    porosity = parse(Float64, env("POROSITY", "0.5"))
    blobiness = parse(Float64, env("BLOBINESS", "1.0"))
    cachedir = env("CACHE", DEFAULT_CACHE)

    if opts["generate-only"] == "1"
        generate_only(sizes; porosity, blobiness, seed, cachedir)
        return nothing
    end

    paths = _list(opts["paths"])
    passes = _list(opts["passes"])
    do_precond = opts["precond"] == "1"
    force = opts["force"] == "1"
    axis = Symbol(opts["axis"])
    # `auto` hands the choice to the package's own solve entry point, which picks
    # by element type — the tolerance a Float32 device solve can actually reach.
    reltol = opts["reltol"] == "auto" ? nothing : parse(Float64, opts["reltol"])
    dev = opts["device"]
    dev in ("gpu", "cpu") || error("Unknown device: $(dev) (expected gpu or cpu)")
    gpu = dev == "gpu"
    gpu && !CUDA.functional() && error("gpu requested but CUDA is not functional")
    csv = env("RESULTS", joinpath(REPO_ROOT, "bench", "results", "matrixfree.csv"))
    envcsv = joinpath(dirname(csv), "matrixfree_env.csv")

    run_id = Dates.format(now(), "yyyymmdd-HHMMSS")
    ensure_csv(csv)
    write_env_record(envcsv, run_id, opts)
    done = force ? Set{Tuple{Int,String,String,Int,String}}() : completed_cells(csv)
    nthreads = Threads.nthreads()
    sha = git_sha()

    # `--passes=solve` covers two CSV passes when the preconditioner is on.
    wanted_passes = String[]
    "apply" in passes && push!(wanted_passes, "apply")
    "setup" in passes && push!(wanted_passes, "setup")
    if "solve" in passes
        push!(wanted_passes, "solve")
        do_precond && push!(wanted_passes, "solve_precond")
    end

    @info "matrixfree_bench run_id=$(run_id) sizes=$(sizes) paths=$(paths) \
           passes=$(passes) precond=$(do_precond) axis=$(axis) device=$(dev) csv=$(csv)"
    @info "warming up (compilation) at $(WARMUP_SIZE)^3"
    warmup(paths, passes, do_precond; gpu, axis, reltol)

    for n in sizes
        todo = [(p, pass) for p in paths for pass in wanted_passes
                if !((n, p, dev, nthreads, pass) in done)]
        if isempty(todo)
            @info "skip $(n)^3 — already in $(csv)"
            continue
        end

        img = cached_blobs(n; porosity, blobiness, seed, cachedir)
        if img === nothing
            @warn "skipping $(n)^3 — no fixture"
            continue
        end
        base = Dict{String,Any}(
            "run_id" => run_id, "timestamp" => string(now()), "git_sha" => sha,
            "n" => n, "nvoxels" => n^3, "device" => dev, "threads" => nthreads,
        )
        for (path, pass) in todo
            matrixfree = path == "matrixfree"
            # The assembled path indexes with Int32 on GPU and its largest index
            # is 7 * nnodes, so past this bound it does not run out of memory: it
            # faults with an illegal address and takes the process down, which no
            # `try` can catch and which would lose every row still to come. Skip
            # it, and record the skip so the gap in the table has a reason.
            if !matrixfree && gpu && 7 * count(img) + 1 > typemax(Int32)
                @warn "n=$(n) path=assembled pass=$(pass): skipped, 7*nnodes overflows Int32"
                row = base_row_for(base, path, pass)
                row["note"] = "7*nnodes exceeds typemax(Int32)"
                # Write the pass's own terminal stage, and for `apply` its last
                # repeat, so `completed_cells` counts the cell as settled — a
                # stage name of its own would leave every resume re-attempting
                # it and appending another row.
                rep = pass == "apply" ? APPLY_REPS : 1
                emit!(csv, skipped_row(row, TERMINAL_STAGE[pass]; rep=rep))
                continue
            end
            @info "n=$(n) path=$(path) pass=$(pass)"
            st = if pass == "apply"
                run_apply_pass!(csv, base, path; img, gpu, matrixfree, axis)
            elseif pass == "setup"
                run_setup_pass!(csv, base, path; img, gpu, matrixfree, axis)
            else
                run_solve_pass!(csv, base, path;
                    img, gpu, matrixfree, axis,
                    precond=(pass == "solve_precond"), reltol=reltol)
            end
            st in ("oom", "oom_host") &&
                @warn "n=$(n) path=$(path) pass=$(pass): $(st)"
        end
        img = nothing
        GC.gc(true)
    end

    print_summary(csv)
    @info "results: $(csv)"
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
