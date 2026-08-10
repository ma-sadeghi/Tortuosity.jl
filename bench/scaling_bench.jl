# Canonical scaling benchmark for the assembled-sparse-matrix path: peak GPU
# memory plus assembly, solve and end-to-end wall times at a range of cubic
# image sizes, on both the GPU and CPU paths, emitted one CSV row per
# measurement so an OOM or a crash never loses the rows already collected.
#
# Usage
# -----
#   julia --project=benchmarks bench/scaling_bench.jl              # 200 400 600 800
#   julia --project=benchmarks bench/scaling_bench.jl 100 200      # explicit sizes
#   julia --project=benchmarks bench/scaling_bench.jl --help
#
# Positional integer arguments are the cube edge lengths. Flags:
#   --devices=gpu,cpu   which paths to measure (default gpu,cpu)
#   --cpu-max=N         largest size the CPU path is run at (default 200)
#   --repeats=K         repeats of the API pass (default: auto, see REPEAT_PLAN)
#   --no-stages         skip the per-stage attribution pass
#   --force             re-measure cells already present in the CSV
#
# Environment overrides (flags win over env, env wins over the defaults):
#   TORTUOSITY_BENCH_CACHE     blob cache directory (see "Blob cache" below)
#   TORTUOSITY_BENCH_RESULTS   CSV path (default bench/results/scaling.csv)
#   TORTUOSITY_BENCH_DEVICES, _CPU_MAX, _REPEATS, _STAGES, _FORCE
#   TORTUOSITY_BENCH_SEED, _POROSITY, _BLOBINESS
#
# Output format
# -------------
# Long format, one row per (size, device, threads, pass, stage, repeat), so new
# stages can be added without breaking a reader. Two passes are recorded:
#
#   pass=api     the public API exactly as a user calls it. Stages: setup
#                (SteadyDiffusionProblem), solve (KrylovJL_CG), post
#                (reconstruct_field + tortuosity). This pass carries the
#                headline numbers; every later change reports deltas on it.
#   pass=precond the same path with a two-level preconditioner handed to the
#                solver, which is opt-in rather than the default. Adds a
#                `precond` stage for building it. Reported separately so the
#                default-path numbers stay comparable across the whole history.
#   pass=stages  the same assembly opened up into its internal steps, for
#                attribution: h2d, poreindex, count, colptr, entries. Mirrors
#                the body of SteadyDiffusionProblem and build_steady_system; if
#                either changes, update this pass to match.
#
# Memory measurement
# ------------------
# `peak_dev_bytes` is the driver-reported device usage,
# `CUDA.total_memory() - CUDA.available_memory()`, sampled ~1 kHz from a
# background task for the duration of the stage. That quantity, not the sum of
# live Julia objects, is what decides whether an allocation throws: it includes
# blocks the CUDA.jl pool holds but has not handed out. The pool is reclaimed
# once at the start of each repeat, never between stages, so the per-stage
# peaks are absolute device usage and the end-to-end peak is their maximum.
#
# The sampler is a task, so on the default single thread it only samples where
# the main task yields — which is every device synchronisation, i.e. exactly the
# points where the interesting allocations have just landed. Adding `-t 2` gives
# it a thread of its own and a tighter bound; the recorded baseline was taken
# without it, so keep the invocation the same when comparing.
#
# The sampler cannot see a peak that opens and closes between two samples, so
# `peak_dev_bytes` is a lower bound. `retained_dev_bytes` (usage after the
# stage minus usage before it) is exact and complements it, and a stage that
# overshot the device entirely is recorded as status=oom rather than guessed at.
#
# `maxrss_bytes` is `Sys.maxrss()`, the process-wide high-water mark of host
# memory. It is monotone over the whole process, so read it as "host memory had
# reached at least this much by the end of that stage", not as a per-stage cost.
#
# Blob cache
# ----------
# `Imaginator.blobs` costs ~60 s at 800^3, so generated images are cached as raw
# `Array{Bool}` bytes (one byte per voxel, N^3 bytes) keyed by the generation
# parameters. The cache lives outside the repository — it is scratch, and 800^3
# alone is 512 MB. Override the location with TORTUOSITY_BENCH_CACHE.
#
# Deliberate choices worth knowing about
# --------------------------------------
# - `warn_nonpercolating=false` everywhere. The default would run the
#   percolation check at 200^3 (under the 50M-voxel auto threshold) and skip it
#   at 400^3 and above, which would make the assembly times incomparable across
#   sizes. Nothing about the assembled system changes either way.
# - Images are not trimmed. The 800^3 OOM this campaign exists to fix is
#   reproduced from an untrimmed blob image, and `trim_nonpercolating_paths` is
#   CPU-only connected-component labelling that would dominate the wall clock.
# - The CPU path is capped at 200^3 by default. At 400^3 the CPU system carries
#   ~218M nonzeros in Float64, so a single unpreconditioned CG iteration moves
#   several GB through the memory system and the solve alone runs into tens of
#   minutes — a cost with no diagnostic value that would gate every later
#   re-run. Raise --cpu-max deliberately if a CPU change needs a larger point.

using CUDA
using Dates
using ImageFiltering  # optional dependency, needed by Imaginator.blobs
using Printf
using SparseArrays
using Statistics
using Tortuosity
using Tortuosity: Imaginator

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_SIZES = [200, 400, 600, 800]
const DEFAULT_CACHE = joinpath(tempdir(), "tortuosity_bench_blobs")
const WARMUP_SIZE = 64
const RELTOL = 1e-6

# Repeats fall off with size: the small sizes are cheap enough to average, the
# large ones are dominated by memory traffic rather than by timer noise and cost
# minutes each.
REPEAT_PLAN(n) = n <= 200 ? 3 : (n <= 400 ? 2 : 1)

# `threads` is per row rather than per run: CPU assembly is KernelAbstractions
# -threaded and CPU SpMV is not, so a CPU wall time means nothing without it,
# and a run-level record in scaling_env.csv is one join away from being read
# wrong. Every row carries the thread count that produced it.
const CSV_COLUMNS = [
    "run_id", "timestamp", "git_sha", "n", "nvoxels", "device", "threads", "pass",
    "stage", "rep", "status", "wall_s", "peak_dev_bytes", "base_dev_bytes",
    "retained_dev_bytes", "maxrss_bytes", "nnodes", "nedges", "nnz", "iters",
    "tau", "note",
]

# --- Options ---------------------------------------------------------------

const USAGE = """
julia --project=benchmarks bench/scaling_bench.jl [SIZES...] [FLAGS]

  SIZES               cube edge lengths (default $(join(DEFAULT_SIZES, ' ')))
  --devices=gpu,cpu   which paths to measure
  --cpu-max=N         largest size the CPU path is run at (default 200)
  --repeats=K         repeats of the API pass (default: auto by size)
  --no-stages         skip the per-stage attribution pass
  --no-precond        skip the preconditioned pass
  --force             re-measure cells already present in the CSV

Env overrides: TORTUOSITY_BENCH_{CACHE,RESULTS,DEVICES,CPU_MAX,REPEATS,STAGES,
PRECOND,FORCE,SEED,POROSITY,BLOBINESS}. See the header comment for the output
format.
"""

env(key, default) = get(ENV, "TORTUOSITY_BENCH_$(key)", default)
_flag(s) = startswith(s, "--")

function parse_args(args)
    sizes = Int[]
    opts = Dict{String,String}(
        "devices" => env("DEVICES", "gpu,cpu"),
        "cpu-max" => env("CPU_MAX", "200"),
        "repeats" => env("REPEATS", "auto"),
        "stages" => env("STAGES", "1"),
        "precond" => env("PRECOND", "1"),
        "force" => env("FORCE", "0"),
    )
    for a in args
        if a in ("--help", "-h")
            print(USAGE)
            exit(0)
        elseif a == "--no-stages"
            opts["stages"] = "0"
        elseif a == "--no-precond"
            opts["precond"] = "0"
        elseif a == "--force"
            opts["force"] = "1"
        elseif _flag(a) && occursin('=', a)
            k, v = split(a[3:end], '='; limit=2)
            opts[k] = v
        elseif _flag(a)
            error("Unknown flag: $a")
        else
            push!(sizes, parse(Int, a))
        end
    end
    isempty(sizes) && (sizes = copy(DEFAULT_SIZES))
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

const TERMINAL_STAGE = Dict("api" => "post", "precond" => "post", "stages" => "dirichlet")

"""
Cells already measured to a deterministic outcome, so a re-run can skip them.

A cell counts as done only once its pass's *last* stage has been written. Every
outcome the passes handle themselves — success, OOM, an error — writes that row
(OOM fills the remaining stages in as `skipped`), so the only way it is missing
is that the process died or was killed part-way through, which is exactly the
case that should be measured again rather than silently inherited.
"""
function completed_cells(path)
    done = Set{Tuple{Int,String,Int,String,Int}}()
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
        f[col["status"]] in ("ok", "oom", "oom_host", "skipped") || continue
        push!(done, (
            parse(Int, f[col["n"]]), f[col["device"]], parse(Int, f[col["threads"]]),
            pass, parse(Int, f[col["rep"]]),
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
is read; use it once per repeat, not between the stages of one repeat, or the
stage peaks stop being comparable to each other.
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

# --- Fixture ---------------------------------------------------------------

function cached_blobs(n; porosity, blobiness, seed, cachedir)
    mkpath(cachedir)
    name = "blobs_n$(n)_p$(porosity)_b$(blobiness)_seed$(seed).raw"
    path = joinpath(cachedir, name)
    nbytes = n^3
    if isfile(path) && filesize(path) == nbytes
        img = Array{Bool,3}(undef, n, n, n)
        open(io -> read!(io, img), path, "r")
        return img
    end
    @info "generating $(n)^3 blob image (not cached yet)"
    img = Imaginator.blobs(; shape=(n, n, n), porosity=porosity, blobiness=blobiness, seed=seed)
    tmp = path * ".tmp"
    open(io -> write(io, img), tmp, "w")
    mv(tmp, path; force=true)
    return img
end

# --- Passes ----------------------------------------------------------------

"""
The public API path, stage by stage: this is the pass every later change is
compared against. Emits one row per stage; a stage that fails marks the stages
after it `skipped` so the CSV records the whole cell rather than trailing off.
"""
function run_api_pass!(csv, base_row, img; gpu::Bool, precond::Bool=false)
    pass = precond ? "precond" : "api"
    stages = precond ? ["setup", "precond", "solve", "post"] : ["setup", "solve", "post"]
    row(stage, m; extra...) = merge(
        copy(base_row),
        Dict{String,Any}(
            "pass" => pass, "stage" => stage, "status" => m.status, "note" => m.note,
            "wall_s" => m.wall, "peak_dev_bytes" => m.peak, "base_dev_bytes" => m.base,
            "retained_dev_bytes" => m.retained, "maxrss_bytes" => m.maxrss,
        ),
        Dict{String,Any}(String(k) => v for (k, v) in extra),
    )
    skip_after!(stage) = for s in stages[(findfirst(==(stage), stages) + 1):end]
        emit!(csv, merge(
            copy(base_row),
            Dict{String,Any}("pass" => pass, "stage" => s, "status" => "skipped"),
        ))
    end

    setup = measure(; gpu=gpu, reclaim_first=true) do
        SteadyDiffusionProblem(img; axis=:x, gpu=gpu, warn_nonpercolating=false)
    end
    if setup.status != "ok"
        emit!(csv, row("setup", setup))
        skip_after!("setup")
        return setup.status
    end
    sim = setup.val
    emit!(csv, row("setup", setup;
        nnodes=length(sim.prob.b), nnz=length(SparseArrays.nonzeros(sim.prob.A))))

    Pl = nothing
    if precond
        built = measure(; gpu=gpu) do
            two_level_preconditioner(sim)
        end
        if built.status != "ok"
            emit!(csv, row("precond", built))
            skip_after!("precond")
            return built.status
        end
        Pl = built.val
        emit!(csv, row("precond", built;
            note=Pl === nothing ? "no coarse space" : "nc=$(Pl.nc) block=$(Pl.block)"))
    end

    solved = measure(; gpu=gpu) do
        Pl === nothing ? solve(sim.prob, KrylovJL_CG(); reltol=RELTOL, verbose=false) :
        solve(sim.prob, KrylovJL_CG(); Pl=Pl, reltol=RELTOL, verbose=false)
    end
    if solved.status != "ok"
        emit!(csv, row("solve", solved))
        skip_after!("solve")
        return solved.status
    end
    sol = solved.val
    emit!(csv, row("solve", solved;
        iters=hasproperty(sol, :iters) ? sol.iters : nothing))

    tau = Ref{Any}(nothing)
    post = measure(; gpu=gpu) do
        c = reconstruct_field(sol.u, img)
        tau[] = tortuosity(c, img; axis=:x)
        return tau[]
    end
    emit!(csv, row("post", post; tau=tau[]))
    return post.status
end

"""
The assembly opened up into its internal steps, for attribution. Mirrors the
body of `SteadyDiffusionProblem`; keep the two in step when that body changes.
"""
function run_stages_pass!(csv, base_row, img; gpu::Bool)
    row(stage, m; extra...) = merge(
        copy(base_row),
        Dict{String,Any}(
            "pass" => "stages", "stage" => stage, "status" => m.status, "note" => m.note,
            "wall_s" => m.wall, "peak_dev_bytes" => m.peak, "base_dev_bytes" => m.base,
            "retained_dev_bytes" => m.retained, "maxrss_bytes" => m.maxrss,
        ),
        Dict{String,Any}(String(k) => v for (k, v) in extra),
    )
    skipped(stage) = merge(
        copy(base_row), Dict{String,Any}("pass" => "stages", "stage" => stage, "status" => "skipped"),
    )
    remaining = ["h2d", "poreindex", "count", "colptr", "entries"]
    function bail!(stage, m)
        emit!(csv, row(stage, m))
        for s in remaining[(findfirst(==(stage), remaining) + 1):end]
            emit!(csv, skipped(s))
        end
        return m.status
    end

    T = gpu ? Float32 : Float64
    Ti = gpu ? Int32 : Int
    nnodes = count(img)
    nx, ny, nz = size(img)
    bcdim, nbc = 1, nx                      # axis=:x
    D0 = one(T)
    wg = (64, 4, 1)

    m = measure(; gpu=gpu, reclaim_first=true) do
        d = gpu ? Tortuosity._gpu_adapt[](img) : img
        gpu && CUDA.synchronize()
        return d
    end
    m.status == "ok" || return bail!("h2d", m)
    img_dev = m.val
    emit!(csv, row("h2d", m; nnodes=nnodes))

    m = measure(; gpu=gpu) do
        idx = similar(img_dev, Ti)
        cumsum!(vec(idx), vec(img_dev))
        idx .*= img_dev
        return idx
    end
    m.status == "ok" || return bail!("poreindex", m)
    idx = m.val
    backend = Tortuosity.get_backend(idx)
    emit!(csv, row("poreindex", m))

    m = measure(; gpu=gpu) do
        counts = similar(idx, Ti, nnodes)
        b = similar(idx, T, nnodes)
        Tortuosity._steady_count_kernel!(backend, wg)(
            counts, b, idx, nothing, nx, ny, nz, bcdim, nbc, D0; ndrange=(nx, ny, nz),
        )
        Tortuosity.KernelAbstractions.synchronize(backend)
        return counts, b
    end
    m.status == "ok" || return bail!("count", m)
    counts, b = m.val
    emit!(csv, row("count", m))

    m = measure(; gpu=gpu) do
        scan = accumulate(+, counts)
        Tortuosity._free!(counts)
        cp = similar(idx, Ti, nnodes + 1)
        Tortuosity._build_colptr_kernel!(backend)(cp, scan, nnodes; ndrange=max(nnodes, 1))
        Tortuosity.KernelAbstractions.synchronize(backend)
        Tortuosity._free!(scan)
        return cp
    end
    m.status == "ok" || return bail!("colptr", m)
    colptr = m.val
    nnz_A = Int(Array(@view colptr[end:end])[1]) - 1
    emit!(csv, row("colptr", m; nnz=nnz_A))

    m = measure(; gpu=gpu) do
        rowval = similar(idx, Ti, nnz_A)
        nzval = similar(b, T, nnz_A)
        Tortuosity._steady_fill_kernel!(backend, wg)(
            rowval, nzval, colptr, idx, nothing, nx, ny, nz, bcdim, nbc, D0;
            ndrange=(nx, ny, nz),
        )
        Tortuosity.KernelAbstractions.synchronize(backend)
        return rowval, nzval
    end
    m.status == "ok" || return bail!("entries", m)
    emit!(csv, row("entries", m; nnz=nnz_A))
    return "ok"
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

function print_summary(path)
    rows = read_rows(path)
    api = filter(r -> r["pass"] in ("api", "precond"), rows)
    isempty(api) && return nothing
    # Threads are part of the key, not a footnote: a CPU row measured on one
    # thread and one measured on four are different measurements. So is the
    # pass, which decides whether the solver was given a preconditioner.
    keys_ = unique([(parse(Int, r["n"]), r["device"], r["threads"], r["pass"]) for r in api])
    sort!(keys_; by=k -> (k[2], k[4], k[1], k[3]))

    println()
    println("=== bench/scaling_bench.jl — API pass (median over repeats) ===")
    println("peak = max device usage over the stages of the pass; e2e = their sum")
    println("base = device usage before setup (CUDA context etc.); subtract it for problem-only memory")
    println("pass=api is the default path; pass=precond hands the solver a two-level preconditioner")
    @printf(
        "%5s %7s %7s %8s %8s %12s %14s %9s %9s %9s %9s %9s %8s %8s\n",
        "N", "device", "threads", "pass", "status", "nnodes", "nnz", "setup_s",
        "prec_s", "solve_s", "e2e_s", "peak_GiB", "iters", "tau",
    )
    for (n, dev, nthreads, pass) in keys_
        cell = filter(
            r -> parse(Int, r["n"]) == n && r["device"] == dev &&
                 r["threads"] == nthreads && r["pass"] == pass,
            api,
        )
        stage_of(s) = filter(r -> r["stage"] == s, cell)
        med(rs, col) = (v = Float64[x for x in (_num(r[col]) for r in rs) if x !== nothing];
                        isempty(v) ? nothing : median(v))
        setup, solve_, post = stage_of("setup"), stage_of("solve"), stage_of("post")
        statuses = unique([r["status"] for r in cell])
        status = "oom" in statuses ? "OOM" : ("error" in statuses ? "ERROR" : "ok")
        t_setup, t_solve, t_post = med(setup, "wall_s"), med(solve_, "wall_s"), med(post, "wall_s")
        t_prec = med(stage_of("precond"), "wall_s")
        e2e = sum(x for x in (t_setup, t_prec, t_solve, t_post) if x !== nothing; init=0.0)
        peaks = Float64[x for x in (_num(r["peak_dev_bytes"]) for r in cell) if x !== nothing]
        peak = maximum(peaks; init=0.0)
        f(x) = x === nothing ? "-" : @sprintf("%.3f", x)
        g(x) = x === nothing ? "-" : @sprintf("%.0f", x)
        @printf(
            "%5d %7s %7s %8s %8s %12s %14s %9s %9s %9s %9s %9s %8s %8s\n",
            n, dev, nthreads, pass, status,
            g(med(setup, "nnodes")), g(med(setup, "nnz")),
            f(t_setup), f(t_prec), f(t_solve), f(e2e), f(gib(peak)),
            g(med(solve_, "iters")), f(med(post, "tau")),
        )
    end

    stages = filter(r -> r["pass"] == "stages", rows)
    isempty(stages) && return nothing
    println()
    println("=== stage attribution (pass=stages) ===")
    @printf("%5s %7s %11s %9s %10s %12s %12s\n",
        "N", "device", "stage", "status", "wall_s", "peak_GiB", "retained_GiB")
    for (n, dev) in sort!(unique([(parse(Int, r["n"]), r["device"]) for r in stages]); by=k -> (k[2], k[1]))
        for r in filter(r -> parse(Int, r["n"]) == n && r["device"] == dev, stages)
            f(x) = x === nothing ? "-" : @sprintf("%.3f", x)
            @printf("%5d %7s %11s %9s %10s %12s %12s\n",
                n, dev, r["stage"], r["status"], f(_num(r["wall_s"])),
                f(gib(_num(r["peak_dev_bytes"]))), f(gib(_num(r["retained_dev_bytes"]))))
        end
    end
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
            "gpu_total_bytes", "cuda_runtime", "reltol", "options"]
    mkpath(dirname(path))
    isfile(path) || open(io -> println(io, join(cols, ",")), path, "w")
    rec = Dict{String,Any}(
        "run_id" => run_id, "timestamp" => string(now()), "git_sha" => git_sha(),
        "julia" => string(VERSION), "threads" => Threads.nthreads(),
        "gpu_name" => CUDA.functional() ? CUDA.name(CUDA.device()) : "none",
        "gpu_total_bytes" => CUDA.functional() ? Int(CUDA.total_memory()) : 0,
        "cuda_runtime" => CUDA.functional() ? string(CUDA.runtime_version()) : "none",
        "reltol" => RELTOL,
        "options" => join(("$k=$v" for (k, v) in sort(collect(opts))), " "),
    )
    open(io -> println(io, join((csvfield(rec[c]) for c in cols), ",")), path, "a")
    return nothing
end

function warmup(devices)
    img = Imaginator.blobs(; shape=(WARMUP_SIZE, WARMUP_SIZE, WARMUP_SIZE),
                           porosity=0.5, blobiness=1.0, seed=1)
    for dev in devices
        gpu = dev == "gpu"
        sim = SteadyDiffusionProblem(img; axis=:x, gpu=gpu, warn_nonpercolating=false)
        sol = solve(sim.prob, KrylovJL_CG(); reltol=RELTOL, verbose=false)
        tortuosity(reconstruct_field(sol.u, img), img; axis=:x)
        run_stages_pass!(devnull_csv(), Dict{String,Any}(), img; gpu=gpu)
    end
    GC.gc(true)
    CUDA.functional() && CUDA.reclaim()
    return nothing
end

# Warmup rows are compilation noise, not measurements: send them to a throwaway
# file rather than polluting the results CSV.
function devnull_csv()
    path = joinpath(tempdir(), "tortuosity_bench_warmup.csv")
    ensure_csv(path)
    return path
end

function main(args)
    sizes, opts = parse_args(args)
    devices = [strip(d) for d in split(opts["devices"], ',') if !isempty(strip(d))]
    cpu_max = parse(Int, opts["cpu-max"])
    do_stages = opts["stages"] == "1"
    do_precond = opts["precond"] == "1"
    force = opts["force"] == "1"
    seed = parse(Int, env("SEED", "42"))
    porosity = parse(Float64, env("POROSITY", "0.5"))
    blobiness = parse(Float64, env("BLOBINESS", "1.0"))
    cachedir = env("CACHE", DEFAULT_CACHE)
    csv = env("RESULTS", joinpath(REPO_ROOT, "bench", "results", "scaling.csv"))
    envcsv = joinpath(dirname(csv), "scaling_env.csv")

    "gpu" in devices && !CUDA.functional() && error("gpu requested but CUDA is not functional")

    run_id = Dates.format(now(), "yyyymmdd-HHMMSS")
    ensure_csv(csv)
    write_env_record(envcsv, run_id, opts)
    done = force ? Set{Tuple{Int,String,Int,String,Int}}() : completed_cells(csv)
    nthreads = Threads.nthreads()
    sha = git_sha()

    @info "scaling_bench run_id=$(run_id) sizes=$(sizes) devices=$(devices) cpu_max=$(cpu_max) csv=$(csv)"
    @info "warming up (compilation) at $(WARMUP_SIZE)^3"
    warmup(devices)

    for n in sizes
        active = [d for d in devices if d == "gpu" || n <= cpu_max]
        isempty(active) && continue
        nreps = opts["repeats"] == "auto" ? REPEAT_PLAN(n) : parse(Int, opts["repeats"])
        wanted = [(d, "api", r) for d in active for r in 1:nreps]
        do_precond && append!(wanted, [(d, "precond", r) for d in active for r in 1:nreps])
        do_stages && append!(wanted, [(d, "stages", 1) for d in active])
        if all(w -> (n, w[1], nthreads, w[2], w[3]) in done, wanted)
            @info "skip $(n)^3 — already in $(csv)"
            continue
        end

        img = cached_blobs(n; porosity, blobiness, seed, cachedir)
        for dev in active
            gpu = dev == "gpu"
            base = Dict{String,Any}(
                "run_id" => run_id, "timestamp" => string(now()), "git_sha" => sha,
                "n" => n, "nvoxels" => n^3, "device" => dev,
                "threads" => Threads.nthreads(),
            )
            for pass in (do_precond ? ("api", "precond") : ("api",))
                for rep in 1:nreps
                    (n, dev, nthreads, pass, rep) in done && continue
                    @info "n=$(n) dev=$(dev) pass=$(pass) rep=$(rep)/$(nreps)"
                    st = run_api_pass!(
                        csv, merge(base, Dict{String,Any}("rep" => rep)), img;
                        gpu=gpu, precond=(pass == "precond"),
                    )
                    GC.gc(true)
                    gpu && CUDA.reclaim()
                    if st in ("oom", "oom_host")
                        @warn "n=$(n) dev=$(dev): $(st) on the $(pass) pass — not repeating"
                        break
                    end
                end
            end
            if do_stages && !((n, dev, nthreads, "stages", 1) in done)
                @info "n=$(n) dev=$(dev) pass=stages"
                run_stages_pass!(csv, merge(base, Dict{String,Any}("rep" => 1)), img; gpu=gpu)
                GC.gc(true)
                gpu && CUDA.reclaim()
            end
        end
        img = nothing
        GC.gc(true)
    end

    print_summary(csv)
    @info "results: $(csv)"
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
