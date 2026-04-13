# =============================================================================
#  Tortuosity GPU benchmark
# =============================================================================
# Compares OLD CUDA-specific code (frozen in bench/old_baseline.jl) against the
# NEW KernelAbstractions-based code, across operations and problem sizes.
#
# Goals:
#   1. Track speed and memory regressions over time, reproducibly.
#   2. Save results as JSON baselines (in bench/baselines/) for git tracking.
#   3. Compare any current run against any saved baseline.
#
# Quick start:
#   julia --project=bench bench/gpu_bench.jl                       # run all, print table
#   julia --project=bench bench/gpu_bench.jl --filter spmv         # run a subset
#   julia --project=bench bench/gpu_bench.jl --save                # save to default path
#   julia --project=bench bench/gpu_bench.jl --compare-with PATH   # regression check
#   julia --project=bench bench/gpu_bench.jl --help                # full CLI reference
#
# Adding a new operation: append a single Op(...) entry to OPERATIONS in
# section 5. The harness handles dispatch, fixturing, formatting, and
# baseline tracking automatically.
module GPUBench

# =============================================================================
# 1. Imports
# =============================================================================
using CUDA
using Tortuosity:
    Tortuosity,
    PortableSparseCSC,
    SteadyDiffusionProblem,
    TransientDiffusionProblem,
    create_connectivity_list,
    create_adjacency_matrix,
    laplacian,
    apply_dirichlet_bc_fast!,
    set_diag!,
    get_diag,
    zero_rows_cols!,
    zero_rows!,
    dropzeros!,
    multihotvec,
    init_state,
    stop_at_time,
    KrylovJL_CG
# `solve` and `solve!` are heavily ambiguous across the SciML ecosystem; pull
# them from Tortuosity so the bench code stays unambiguous.
import Tortuosity: solve, solve!
using LinearAlgebra: mul!
using LinearSolve: LinearProblem
using SparseArrays: nonzeros
using BenchmarkTools
using Printf
using Dates
using JSON

# Old baseline lives in a sibling file. Loading it brings OldBaseline into scope.
include(joinpath(@__DIR__, "old_baseline.jl"))
using .OldBaseline

# =============================================================================
# 2. Configuration
# =============================================================================
"Default BenchmarkTools parameters for the harness."
const DEFAULT_PARAMS = (seconds=3.0, samples=50)

"Default regression threshold (in percent) when comparing against a baseline."
const DEFAULT_THRESHOLD_PCT = 5.0

"Where saved baselines live (one JSON per branch by convention)."
const BASELINE_DIR = joinpath(@__DIR__, "baselines")

"""
Problem sizes to benchmark. Each entry is `(name, n)` where the cube is n×n×n
and `name` is the human label used in CLI filtering and tables.
"""
const SIZES = [
    (name="100K", n=47),    # 47^3 =   103,823 voxels
    (name="250K", n=63),    # 63^3 =   250,047 voxels
    (name="1M",   n=100),   # 100^3 = 1,000,000 voxels
]

# =============================================================================
# 3. Run metadata capture
# =============================================================================
"""
    capture_metadata() -> NamedTuple

Snapshot the environment at the time of a run: git commit/branch, GPU info,
CUDA versions, Julia version, hostname, timestamp. Embedded in saved baselines
so future readers can tell what the numbers came from.
"""
function capture_metadata()
    git_commit = try
        strip(read(`git rev-parse --short HEAD`, String))
    catch
        "unknown"
    end
    git_branch = try
        strip(read(`git rev-parse --abbrev-ref HEAD`, String))
    catch
        "unknown"
    end
    git_dirty = try
        !isempty(strip(read(`git status --porcelain`, String)))
    catch
        false
    end
    return (
        timestamp = string(now()),
        hostname = gethostname(),
        git_commit = git_commit,
        git_branch = git_branch,
        git_dirty = git_dirty,
        gpu_name = string(CUDA.name(CUDA.device())),
        cuda_driver = string(CUDA.driver_version()),
        cuda_runtime = string(CUDA.runtime_version()),
        julia_version = string(VERSION),
        bench_seconds = DEFAULT_PARAMS.seconds,
        bench_samples = DEFAULT_PARAMS.samples,
    )
end

# =============================================================================
# 4. Fixture builder
# =============================================================================
"""
    build_fixture(size_spec) -> NamedTuple

Construct everything an operation might need at a given size:
- `img_gpu`: cubic boolean image on GPU
- `img_cpu`: same on CPU (for transient pipeline which expects CPU input)
- `nnodes`, `nedges`
- `conns`: connectivity list (built once with old code; both old and new
  consume the same Int32 connectivity matrix, so this is fair)
- `weights`: vector of unit weights
- `am_old`, `am_new`: adjacency matrices via each pipeline
- `L_old`, `L_new`: Laplacians via each pipeline
- `bc_n`, `bc_nodes`: boundary conditions for Dirichlet ops

The fixture is built ONCE per size and shared across all operations at that
size. This keeps the bench fast and ensures every operation sees the same
starting state.
"""
function build_fixture(size_spec)
    n = size_spec.n
    img_cpu = ones(Bool, n, n, n)
    img_gpu = CuArray(img_cpu)
    nnodes = n * n * n

    conns = OldBaseline.create_connectivity_list_old(img_gpu)
    nedges = size(conns, 1)
    weights = CUDA.fill(1.0f0, nedges)

    am_old = OldBaseline.create_adjacency_matrix_old(conns; n=nnodes, weights=weights)
    am_new = create_adjacency_matrix(conns; n=nnodes, weights=weights)

    L_old = OldBaseline.laplacian_old(am_old)
    L_new = laplacian(am_new)

    bc_n = min(1000, nnodes ÷ 2)
    bc_nodes = collect(Int32, 1:bc_n)

    return (;
        img_cpu, img_gpu, n, nnodes, nedges,
        conns, weights, am_old, am_new, L_old, L_new,
        bc_n, bc_nodes,
    )
end

# =============================================================================
# 5. Operation registry (declarative)
# =============================================================================
"""
    Op(name, category; old, new, [setup])

A single benchmark entry. Both `old` and `new` are functions that take
`(setup, fixture)` and execute the operation under `CUDA.@sync`. The `setup`
function (optional, defaults to returning an empty NamedTuple) prepares
per-call state and is called BEFORE each `@benchmark` invocation; its result
is passed to both `old` and `new`. Use `setup` for state that must be fresh
for each run (e.g., a copy of the matrix that the operation will mutate).

`category` is one of:
- `:component` — a single primitive (kernel, sparse op, or helper)
- `:integration` — a higher-level user-facing function

Adding a new operation = appending one entry to OPERATIONS. The harness
handles everything else.
"""
struct Op
    name::String
    category::Symbol
    setup::Function
    old::Function
    new::Function
end

Op(name, category; old, new, setup = _ -> NamedTuple()) =
    Op(name, category, setup, old, new)

# --- Multi-line setup helpers (extracted because `f -> begin ... end` doesn't
# parse cleanly inside a function-call kwarg position) ---

_setup_dropzeros(f) = let
    L_old = OldBaseline.laplacian_old(f.am_old)
    L_new = laplacian(f.am_new)
    drop_idxs = CuArray(collect(1:10:length(L_old.nzVal)))
    L_old.nzVal[drop_idxs] .= 0.0f0
    nonzeros(L_new)[drop_idxs] .= 0.0f0
    (; L_old, L_new)
end

_setup_apply_dirichlet_bc(f) = let
    L_old = OldBaseline.laplacian_old(f.am_old)
    L_new = laplacian(f.am_new)
    b_old = CUDA.zeros(Float32, f.nnodes)
    b_new = CUDA.zeros(Float32, f.nnodes)
    bc_vals = CuArray(fill(1.0f0, f.bc_n))
    (; L_old, L_new, b_old, b_new, bc_vals)
end

_setup_solve_steady(f) = let
    A_old, b_old = OldBaseline.tortuosity_simulation_old(f.img_gpu)
    ts_new = SteadyDiffusionProblem(f.img_gpu; axis=:x, gpu=true)
    prob_old = LinearProblem(A_old, b_old)
    (; prob_old, prob_new = ts_new.prob)
end

_setup_solve_transient(f) = let
    prob_old = OldBaseline.transient_problem_old(f.img_cpu, 0.01; axis=:z, dtype=Float32)
    prob_new = TransientDiffusionProblem(f.img_cpu, 0.01; axis=:z, dtype=Float32, gpu=true)
    state_old = OldBaseline.init_state_old(prob_old)
    state_new = init_state(prob_new)
    (; prob_old, prob_new, state_old, state_new)
end

const OPERATIONS = Op[
    # ---------- Components ----------
    Op("create_connectivity_list", :component;
       old = (s, f) -> OldBaseline.create_connectivity_list_old(f.img_gpu),
       new = (s, f) -> create_connectivity_list(f.img_gpu)),

    Op("create_adjacency_matrix", :component;
       old = (s, f) -> OldBaseline.create_adjacency_matrix_old(f.conns; n=f.nnodes, weights=f.weights),
       new = (s, f) -> create_adjacency_matrix(f.conns; n=f.nnodes, weights=f.weights)),

    Op("laplacian", :component;
       old = (s, f) -> OldBaseline.laplacian_old(f.am_old),
       new = (s, f) -> laplacian(f.am_new)),

    Op("SpMV mul!", :component;
       setup = f -> (
           x = CUDA.ones(Float32, f.nnodes),
           y_old = CUDA.zeros(Float32, f.nnodes),
           y_new = CUDA.zeros(Float32, f.nnodes),
       ),
       old = (s, f) -> mul!(s.y_old, f.L_old, s.x),
       new = (s, f) -> mul!(s.y_new, f.L_new, s.x)),

    Op("get_diag", :component;
       old = (s, f) -> OldBaseline.get_diag_old(f.L_old),
       new = (s, f) -> get_diag(f.L_new)),

    Op("set_diag!", :component;
       setup = f -> (
           diag_v = CUDA.ones(Float32, f.nnodes),
           L_old_copy = OldBaseline.laplacian_old(f.am_old),
           L_new_copy = laplacian(f.am_new),
       ),
       old = (s, f) -> OldBaseline.set_diag_old!(s.L_old_copy, s.diag_v),
       new = (s, f) -> set_diag!(s.L_new_copy, s.diag_v)),

    Op("zero_rows_cols!", :component;
       setup = f -> (
           L_old_copy = OldBaseline.laplacian_old(f.am_old),
           L_new_copy = laplacian(f.am_new),
       ),
       old = (s, f) -> OldBaseline.zero_rows_cols_old!(s.L_old_copy, f.bc_nodes),
       new = (s, f) -> zero_rows_cols!(s.L_new_copy, f.bc_nodes)),

    Op("zero_rows!", :component;
       setup = f -> (
           L_old_copy = OldBaseline.laplacian_old(f.am_old),
           L_new_copy = laplacian(f.am_new),
       ),
       old = (s, f) -> OldBaseline.zero_rows_old!(s.L_old_copy, f.bc_nodes),
       new = (s, f) -> zero_rows!(s.L_new_copy, f.bc_nodes)),

    Op("dropzeros!", :component;
       setup = _setup_dropzeros,
       old = (s, f) -> OldBaseline.dropzeros_old!(s.L_old),
       new = (s, f) -> dropzeros!(s.L_new)),

    Op("multihotvec", :component;
       setup = f -> (
           # Pass GPU vals to both — matches how production code uses multihotvec
           # via apply_dirichlet_bc_fast! (which transfers vals to GPU before
           # calling multihotvec).
           vals_gpu = CUDA.fill(1.0f0, f.bc_n),
           template = CUDA.zeros(Float32, f.nnodes),
       ),
       old = (s, f) -> OldBaseline.multihotvec_old(f.bc_nodes, f.nnodes; vals=s.vals_gpu, gpu=true),
       new = (s, f) -> multihotvec(f.bc_nodes, f.nnodes; vals=s.vals_gpu, template=s.template)),

    Op("apply_dirichlet_bc_fast!", :component;
       setup = _setup_apply_dirichlet_bc,
       old = (s, f) -> OldBaseline.apply_dirichlet_bc_old!(s.L_old, s.b_old; nodes=f.bc_nodes, vals=s.bc_vals),
       new = (s, f) -> apply_dirichlet_bc_fast!(s.L_new, s.b_new; nodes=f.bc_nodes, vals=s.bc_vals)),

    # ---------- Integration / workflow ----------
    Op("SteadyDiffusionProblem", :integration;
       old = (s, f) -> OldBaseline.tortuosity_simulation_old(f.img_gpu),
       new = (s, f) -> SteadyDiffusionProblem(f.img_gpu; axis=:x, gpu=true)),

    Op("solve (steady-state Krylov)", :integration;
       setup = _setup_solve_steady,
       old = (s, f) -> solve(s.prob_old, KrylovJL_CG(); reltol=1e-6),
       new = (s, f) -> solve(s.prob_new, KrylovJL_CG(); reltol=1e-6)),

    Op("TransientDiffusionProblem", :integration;
       old = (s, f) -> OldBaseline.transient_problem_old(f.img_cpu, 0.01; axis=:z, dtype=Float32),
       new = (s, f) -> TransientDiffusionProblem(f.img_cpu, 0.01; axis=:z, dtype=Float32, gpu=true)),

    Op("solve! (transient, 5 dt-steps)", :integration;
       setup = _setup_solve_transient,
       old = (s, f) -> solve!(s.state_old, s.prob_old, stop_at_time(0.05); max_iter=5),
       new = (s, f) -> solve!(s.state_new, s.prob_new, stop_at_time(0.05); max_iter=5)),
]

# =============================================================================
# 6. Benchmark engine
# =============================================================================
"A single (op, size) measurement: timing + memory + allocations for both paths."
struct Measurement
    op_name::String
    category::Symbol
    size_name::String
    nnodes::Int
    nedges::Int
    t_old::Float64    # nanoseconds (median)
    t_new::Float64
    mem_old::Int      # bytes (Julia heap)
    mem_new::Int
    alloc_old::Int    # number of allocations
    alloc_new::Int
end

"All measurements from a single run, plus metadata."
struct RunResult
    metadata::NamedTuple
    measurements::Vector{Measurement}
end

"""
    bench_pair(op, fixture; params) -> Measurement

Benchmark a single Op against its fixture. Runs `op.old` and `op.new`
separately under `CUDA.@sync`, using BenchmarkTools' `setup=` keyword so
that per-iteration state preparation happens OUTSIDE the timed region. This
matters for ops that mutate the matrix (set_diag!, zero_rows!, dropzeros!,
etc.) — without `setup=`, the laplacian rebuild would dominate the timing.

BenchmarkTools handles its own warmup; the first sample reflects steady-state
performance, not JIT compilation cost.
"""
function bench_pair(op::Op, fixture; params=DEFAULT_PARAMS)
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = params.seconds
    BenchmarkTools.DEFAULT_PARAMETERS.samples = params.samples

    # Capture function values into locals so the @benchmark macro's
    # interpolation has stable references to splice in.
    setup_fn = op.setup
    old_fn = op.old
    new_fn = op.new

    b_old = @benchmark(
        (CUDA.@sync $old_fn(s, $fixture)),
        setup = (s = $setup_fn($fixture))
    )
    b_new = @benchmark(
        (CUDA.@sync $new_fn(s, $fixture)),
        setup = (s = $setup_fn($fixture))
    )

    return Measurement(
        op.name, op.category,
        "", 0, 0,    # size_name/nnodes/nedges filled in by run_size
        median(b_old).time, median(b_new).time,
        b_old.memory, b_new.memory,
        b_old.allocs, b_new.allocs,
    )
end

"""
    run_size(size_spec, ops; params) -> Vector{Measurement}

Build the fixture once and run all `ops` against it.
"""
function run_size(size_spec, ops::Vector{Op}; params=DEFAULT_PARAMS)
    fixture = build_fixture(size_spec)
    measurements = Measurement[]
    for op in ops
        m = bench_pair(op, fixture; params=params)
        push!(measurements, Measurement(
            m.op_name, m.category,
            size_spec.name, fixture.nnodes, fixture.nedges,
            m.t_old, m.t_new, m.mem_old, m.mem_new, m.alloc_old, m.alloc_new,
        ))
    end
    return measurements
end

"""
    run_all(; sizes, ops, params) -> RunResult

Top-level entry. Runs every op at every size and returns a complete
RunResult ready for printing, saving, or comparing.
"""
function run_all(; sizes=SIZES, ops::Vector{Op}=OPERATIONS, params=DEFAULT_PARAMS)
    metadata = capture_metadata()
    all_measurements = Measurement[]
    for size_spec in sizes
        println(">>> Building fixture for $(size_spec.name) ($(size_spec.n)^3)...")
        ms = run_size(size_spec, ops; params=params)
        append!(all_measurements, ms)
        print_size_table(size_spec, ms)
    end
    return RunResult(metadata, all_measurements)
end

# =============================================================================
# 7. Output: human-readable formatting
# =============================================================================
const TABLE_WIDTH = 116

# Box-drawing helpers (UTF-8; modern terminals only — confirmed by Amin)
const HBAR  = "═"
const HBAR2 = "─"
const SEP   = " · "

format_int_commas(n::Integer) = replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+(?:\.|$))" => ",")

function format_time(ns)
    ns < 1_000          && return @sprintf("%6.0f ns", ns)
    ns < 1_000_000      && return @sprintf("%6.1f µs", ns / 1_000)
    ns < 1_000_000_000  && return @sprintf("%6.2f ms", ns / 1_000_000)
    return @sprintf("%6.3f s ", ns / 1_000_000_000)
end

function format_memory(bytes)
    bytes < 1024      && return @sprintf("%6d B ", bytes)
    bytes < 1024^2    && return @sprintf("%6.1f KB", bytes / 1024)
    bytes < 1024^3    && return @sprintf("%6.2f MB", bytes / 1024^2)
    return @sprintf("%6.2f GB", bytes / 1024^3)
end

format_ratio(r) = @sprintf("%6.2f×", r)

# Status marker — `ok` if new is at least as good as old (within threshold).
# Speed: ratio = old/new, > 1 means new is faster.
# Memory: ratio = old/new, > 1 means new uses less.
function status_marker(ratio; threshold_ratio=0.95)
    ratio >= threshold_ratio && return " ok "
    return "WARN"
end

function print_run_header(metadata)
    println()
    println(HBAR^TABLE_WIDTH)
    println("  Tortuosity GPU benchmark")
    println(HBAR2^TABLE_WIDTH)
    println(@sprintf("  Date          %s", metadata.timestamp))
    println(@sprintf("  Hostname      %s", metadata.hostname))
    println(@sprintf("  Git           %s @ %s%s",
        metadata.git_commit, metadata.git_branch,
        metadata.git_dirty ? " (dirty)" : ""))
    println(@sprintf("  GPU           %s", metadata.gpu_name))
    println(@sprintf("  CUDA          driver %s, runtime %s",
        metadata.cuda_driver, metadata.cuda_runtime))
    println(@sprintf("  Julia         %s", metadata.julia_version))
    println(@sprintf("  BenchmarkTools  seconds=%.1f, samples=%d",
        metadata.bench_seconds, metadata.bench_samples))
    println(HBAR^TABLE_WIDTH)
end

function print_size_table(size_spec, measurements::Vector{Measurement})
    isempty(measurements) && return
    f = first(measurements)
    println()
    println(HBAR^TABLE_WIDTH)
    title = @sprintf("  %s  %s  %s voxels  %s  %s edges",
        size_spec.name, SEP, format_int_commas(f.nnodes), SEP, format_int_commas(f.nedges))
    println(title)
    println(HBAR2^TABLE_WIDTH)
    println(@sprintf("  %-30s  %10s  %10s  %8s        %10s  %10s  %8s",
        "Operation", "Old time", "New time", "Speed×", "Old mem", "New mem", "Mem×"))
    println(HBAR2^TABLE_WIDTH)

    # Print components first, then a separator, then integration ops
    components = filter(m -> m.category === :component, measurements)
    integrations = filter(m -> m.category === :integration, measurements)

    for m in components
        print_measurement_row(m)
    end
    if !isempty(components) && !isempty(integrations)
        println(HBAR2^TABLE_WIDTH)
    end
    for m in integrations
        print_measurement_row(m)
    end

    println(HBAR^TABLE_WIDTH)
end

function print_measurement_row(m::Measurement)
    speed = m.t_old / m.t_new
    mem_ratio = m.mem_old == 0 ? (m.mem_new == 0 ? 1.0 : 0.0) : m.mem_old / max(m.mem_new, 1)
    speed_marker = status_marker(speed)
    mem_marker = status_marker(mem_ratio)
    println(@sprintf("  %-30s  %10s  %10s  %s %s   %10s  %10s  %s %s",
        m.op_name, format_time(m.t_old), format_time(m.t_new),
        format_ratio(speed), speed_marker,
        format_memory(m.mem_old), format_memory(m.mem_new),
        format_ratio(mem_ratio), mem_marker))
end

# =============================================================================
# 8. JSON serialization (save / load baselines)
# =============================================================================
function to_dict(m::Measurement)
    return Dict(
        "op_name" => m.op_name,
        "category" => string(m.category),
        "size_name" => m.size_name,
        "nnodes" => m.nnodes,
        "nedges" => m.nedges,
        "t_old_ns" => m.t_old,
        "t_new_ns" => m.t_new,
        "mem_old_bytes" => m.mem_old,
        "mem_new_bytes" => m.mem_new,
        "alloc_old" => m.alloc_old,
        "alloc_new" => m.alloc_new,
    )
end

function from_dict(d::AbstractDict)
    return Measurement(
        d["op_name"], Symbol(d["category"]),
        d["size_name"], d["nnodes"], d["nedges"],
        d["t_old_ns"], d["t_new_ns"],
        d["mem_old_bytes"], d["mem_new_bytes"],
        d["alloc_old"], d["alloc_new"],
    )
end

function to_dict(r::RunResult)
    # NamedTuple → Dict for JSON serialization
    meta_dict = Dict{String,Any}(string(k) => v for (k, v) in pairs(r.metadata))
    return Dict(
        "metadata" => meta_dict,
        "measurements" => [to_dict(m) for m in r.measurements],
    )
end

function from_run_dict(d::AbstractDict)
    meta = (; (Symbol(k) => v for (k, v) in d["metadata"])...)
    measurements = [from_dict(m) for m in d["measurements"]]
    return RunResult(meta, measurements)
end

"""
    save_baseline(result::RunResult, path::String)

Write a run result to a pretty-printed JSON file. Creates parent directories
if needed. Use `--save` from the CLI to invoke this with the default branch
path.
"""
function save_baseline(result::RunResult, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, to_dict(result), 2)
    end
    println()
    println("Baseline saved to: ", path)
end

"""
    load_baseline(path::String) -> RunResult

Read a previously-saved baseline file.
"""
function load_baseline(path::String)
    isfile(path) || error("Baseline file not found: $path")
    return from_run_dict(JSON.parse(read(path, String)))
end

"""
    default_baseline_path() -> String

The conventional save target: `bench/baselines/<branch>.json`.
"""
function default_baseline_path()
    branch = try
        strip(read(`git rev-parse --abbrev-ref HEAD`, String))
    catch
        "default"
    end
    safe_branch = replace(branch, "/" => "-")
    return joinpath(BASELINE_DIR, "$(safe_branch).json")
end

# =============================================================================
# 9. Regression comparison
# =============================================================================
"A single comparison row between baseline and current."
struct ComparisonRow
    op_name::String
    size_name::String
    metric::Symbol             # :time or :memory
    baseline_value::Float64
    current_value::Float64
    delta_pct::Float64          # (current - baseline) / baseline * 100
    status::Symbol              # :regression, :improvement, :unchanged
end

"""
    compare(current::RunResult, baseline::RunResult; threshold_pct=5.0) -> Vector{ComparisonRow}

Compare each (op, size) measurement in `current` against the same in
`baseline`. A row is flagged as `:regression` if `current` is more than
`threshold_pct` slower (time) or larger (memory) than baseline. The reverse
is flagged as `:improvement`.
"""
function compare(current::RunResult, baseline::RunResult; threshold_pct=DEFAULT_THRESHOLD_PCT)
    rows = ComparisonRow[]
    by_key = Dict((m.op_name, m.size_name) => m for m in baseline.measurements)
    for m in current.measurements
        key = (m.op_name, m.size_name)
        haskey(by_key, key) || continue
        b = by_key[key]
        # Compare the NEW (post-refactor) values; that's what we want to track
        # over time. The OLD values are constant (frozen reference) and don't
        # change between runs.
        for (metric, b_val, c_val) in (
            (:time,   b.t_new,   m.t_new),
            (:memory, Float64(b.mem_new), Float64(m.mem_new)),
        )
            b_val == 0 && continue
            delta_pct = (c_val - b_val) / b_val * 100
            status = if delta_pct > threshold_pct
                :regression
            elseif delta_pct < -threshold_pct
                :improvement
            else
                :unchanged
            end
            push!(rows, ComparisonRow(m.op_name, m.size_name, metric,
                b_val, c_val, delta_pct, status))
        end
    end
    return rows
end

function print_comparison(rows::Vector{ComparisonRow}, baseline_path::String,
                          baseline_meta, threshold_pct)
    regressions = filter(r -> r.status === :regression, rows)
    improvements = filter(r -> r.status === :improvement, rows)
    unchanged = filter(r -> r.status === :unchanged, rows)

    println()
    println(HBAR^TABLE_WIDTH)
    println("  Comparison vs baseline: ", baseline_path)
    println(HBAR2^TABLE_WIDTH)
    println(@sprintf("  Baseline date    %s", baseline_meta.timestamp))
    println(@sprintf("  Baseline commit  %s @ %s",
        baseline_meta.git_commit, baseline_meta.git_branch))
    println(@sprintf("  Threshold        %.1f%%", threshold_pct))
    println(HBAR^TABLE_WIDTH)

    if !isempty(regressions)
        println()
        println("REGRESSIONS:")
        println(HBAR2^TABLE_WIDTH)
        println(@sprintf("  %-6s  %-30s  %-7s  %14s  %14s  %10s",
            "Size", "Operation", "Metric", "Baseline", "Current", "Delta"))
        println(HBAR2^TABLE_WIDTH)
        for r in regressions
            print_comparison_row(r)
        end
    end

    if !isempty(improvements)
        println()
        println("IMPROVEMENTS:")
        println(HBAR2^TABLE_WIDTH)
        println(@sprintf("  %-6s  %-30s  %-7s  %14s  %14s  %10s",
            "Size", "Operation", "Metric", "Baseline", "Current", "Delta"))
        println(HBAR2^TABLE_WIDTH)
        for r in improvements
            print_comparison_row(r)
        end
    end

    println()
    println(HBAR^TABLE_WIDTH)
    println(@sprintf("  Summary: %d regressions, %d improvements, %d unchanged",
        length(regressions), length(improvements), length(unchanged)))
    println(HBAR^TABLE_WIDTH)
end

function print_comparison_row(r::ComparisonRow)
    fmt = r.metric === :time ? format_time : format_memory
    sign = r.delta_pct >= 0 ? "+" : ""
    println(@sprintf("  %-6s  %-30s  %-7s  %14s  %14s  %s%6.1f%%",
        r.size_name, r.op_name, string(r.metric),
        fmt(r.baseline_value), fmt(r.current_value),
        sign, r.delta_pct))
end

# =============================================================================
# 10. CLI parser
# =============================================================================
const HELP_TEXT = """
Tortuosity GPU benchmark — single-source-of-truth performance harness.

USAGE
  julia --project=bench bench/gpu_bench.jl [options]

OPTIONS
  --filter NAMES         Comma-separated substrings; only run ops whose name
                         contains any of them. Case-insensitive.
                         Example: --filter spmv,laplacian

  --sizes NAMES          Comma-separated size names from SIZES.
                         Example: --sizes 100K,1M

  --save [PATH]          Save the run as a baseline JSON. PATH defaults to
                         bench/baselines/<git-branch>.json.

  --compare-with PATH    Load PATH as a baseline and print a regression
                         report. Exits with code 1 if any regression > threshold.

  --threshold PCT        Regression threshold in percent (default: 5.0).

  --json [PATH]          Write machine-readable JSON to PATH (or `-` for stdout).
                         Suppresses the regression exit code.

  --no-table             Suppress the human-readable table output. Useful with
                         --json or --save when piping.

  --list                 Print the operation registry and exit.

  --help                 Show this message.

EXAMPLES
  # Full run, save to default baseline path
  julia --project=bench bench/gpu_bench.jl --save

  # Run only SpMV at 1M, compare with main baseline
  julia --project=bench bench/gpu_bench.jl --filter spmv --sizes 1M \\
      --compare-with bench/baselines/main.json
"""

"""
    parse_args(args::Vector{String}) -> NamedTuple

Hand-rolled CLI parser. Returns a NamedTuple of resolved options. Unknown
options or `--help` cause this function to print help and return `nothing`.
"""
function parse_args(args::Vector{String})
    opts = Dict{String,Any}(
        "filter" => nothing,
        "sizes" => nothing,
        "save" => false,
        "save_path" => nothing,
        "compare_with" => nothing,
        "threshold" => DEFAULT_THRESHOLD_PCT,
        "json" => nothing,
        "no_table" => false,
        "list" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--help" || a == "-h"
            println(HELP_TEXT)
            return nothing
        elseif a == "--list"
            opts["list"] = true; i += 1
        elseif a == "--no-table"
            opts["no_table"] = true; i += 1
        elseif a == "--save"
            opts["save"] = true
            # Optional positional path
            if i + 1 <= length(args) && !startswith(args[i + 1], "--")
                opts["save_path"] = args[i + 1]; i += 2
            else
                i += 1
            end
        elseif a == "--filter"
            opts["filter"] = args[i + 1]; i += 2
        elseif a == "--sizes"
            opts["sizes"] = args[i + 1]; i += 2
        elseif a == "--compare-with"
            opts["compare_with"] = args[i + 1]; i += 2
        elseif a == "--threshold"
            opts["threshold"] = parse(Float64, args[i + 1]); i += 2
        elseif a == "--json"
            opts["json"] = args[i + 1]; i += 2
        else
            println(stderr, "Unknown option: ", a)
            println(stderr, HELP_TEXT)
            error("CLI parse error")
        end
    end
    return (;
        filter = opts["filter"],
        sizes = opts["sizes"],
        save = opts["save"],
        save_path = opts["save_path"],
        compare_with = opts["compare_with"],
        threshold = opts["threshold"],
        json = opts["json"],
        no_table = opts["no_table"],
        list = opts["list"],
    )
end

function filter_ops(ops::Vector{Op}, filter_str)
    isnothing(filter_str) && return ops
    needles = lowercase.(strip.(split(filter_str, ",")))
    return filter(op -> any(occursin(n, lowercase(op.name)) for n in needles), ops)
end

function filter_sizes(sizes, sizes_str)
    isnothing(sizes_str) && return sizes
    wanted = strip.(split(sizes_str, ","))
    return filter(s -> s.name in wanted, sizes)
end

function print_op_list()
    println()
    println(HBAR^TABLE_WIDTH)
    println("  Operation registry  ($(length(OPERATIONS)) ops × $(length(SIZES)) sizes)")
    println(HBAR2^TABLE_WIDTH)
    components = filter(o -> o.category === :component, OPERATIONS)
    integrations = filter(o -> o.category === :integration, OPERATIONS)
    println("  Components ($(length(components))):")
    for op in components
        println("    - ", op.name)
    end
    println()
    println("  Integration ($(length(integrations))):")
    for op in integrations
        println("    - ", op.name)
    end
    println(HBAR2^TABLE_WIDTH)
    println("  Sizes:")
    for s in SIZES
        println(@sprintf("    - %-6s  (%d^3 = %s voxels)",
            s.name, s.n, format_int_commas(s.n^3)))
    end
    println(HBAR^TABLE_WIDTH)
end

# =============================================================================
# 11. Main entry point
# =============================================================================
"""
    main(args::Vector{String}=ARGS) -> Int

Top-level CLI entry. Returns an exit code: 0 on success, 1 if a regression
was detected via `--compare-with`. Suitable for use as a script or as
`exit(GPUBench.main(ARGS))`.
"""
function main(args::Vector{String}=ARGS)
    parsed = parse_args(args)
    isnothing(parsed) && return 0  # --help printed, exit cleanly

    if parsed.list
        print_op_list()
        return 0
    end

    @assert CUDA.functional() """
    CUDA must be functional to run this benchmark.
    Install CUDA.jl: julia -e 'using Pkg; Pkg.add("CUDA")'
    """

    # Resolve filtered ops and sizes
    selected_ops = filter_ops(OPERATIONS, parsed.filter)
    selected_sizes = filter_sizes(SIZES, parsed.sizes)
    isempty(selected_ops) && error("No operations matched --filter $(parsed.filter)")
    isempty(selected_sizes) && error("No sizes matched --sizes $(parsed.sizes)")

    # Brief CUDA warmup (just one trivial op to trigger init; BenchmarkTools
    # handles per-op JIT warmup automatically)
    let
        x = CUDA.zeros(Float32, 32)
        x .+= 1
        CUDA.synchronize()
    end

    # Print header
    metadata = capture_metadata()
    parsed.no_table || print_run_header(metadata)

    # Run
    result = run_all(sizes=selected_sizes, ops=selected_ops)

    # Save
    if parsed.save
        save_path = something(parsed.save_path, default_baseline_path())
        save_baseline(result, save_path)
    end

    # JSON output
    if !isnothing(parsed.json)
        if parsed.json == "-"
            JSON.print(stdout, to_dict(result), 2)
            println()
        else
            mkpath(dirname(parsed.json))
            open(parsed.json, "w") do io
                JSON.print(io, to_dict(result), 2)
            end
            println()
            println("JSON written to: ", parsed.json)
        end
    end

    # Compare against baseline
    exit_code = 0
    if !isnothing(parsed.compare_with)
        baseline = load_baseline(parsed.compare_with)
        rows = compare(result, baseline; threshold_pct=parsed.threshold)
        print_comparison(rows, parsed.compare_with, baseline.metadata, parsed.threshold)
        if !isnothing(parsed.json)
            # JSON mode: don't fail on regressions, just report
            exit_code = 0
        elseif any(r -> r.status === :regression, rows)
            exit_code = 1
        end
    end

    return exit_code
end

end  # module GPUBench

# Run main() only when invoked as a script (not when included from REPL)
if abspath(PROGRAM_FILE) == @__FILE__
    exit(GPUBench.main(ARGS))
end
