# Result files: fixed schemas, append-only writes, and the rule for resuming.
#
# Every stage writes one tidy CSV per (tool, device, variant). The identifying
# columns are repeated on every row rather than encoded only in the filename, so
# the post-processing stage can concatenate the whole directory and never has to
# parse a name to know what it is reading.

const TIMING_COLUMNS = [
    :tool, :device, :variant, :cpu_threads,
    :case_id, :size, :blobiness, :porosity_target, :porosity, :nnodes,
    :knob_name, :knob, :tau, :tau_ref, :rel_error, :time_s, :tau_spread, :repeats,
    :stop_reason, :host, :measured_at, :note,
]

const MEMORY_COLUMNS = [
    :tool, :device, :variant, :cpu_threads,
    :case_id, :size, :blobiness, :porosity_target, :porosity, :nnodes,
    :iters, :time_s, :peak_rss_bytes, :baseline_rss_bytes,
    :peak_device_bytes, :pool_device_bytes, :status, :host, :measured_at, :note,
]

const REFERENCE_COLUMNS = [
    :case_id, :size, :blobiness, :porosity_target, :porosity, :nnodes,
    :tau_ref, :reltol, :solve_time_s, :host, :measured_at,
]

const ENVIRONMENT_COLUMNS = [
    :measured_at, :host, :stage, :tool, :device, :variant,
    :runtime, :runtime_version, :cpu_threads, :accelerator, :notes,
]

"""Render one value as a CSV field.

Floats get ten significant digits — far beyond any quantity measured here, and
short enough that the file stays readable. `NaN` is written literally because it
is a result: a spread that a single repeat could not measure is not the same
thing as a spread of zero, and a blank would read as the latter.
"""
csv_field(x::AbstractFloat) = isnan(x) ? "NaN" : (isinf(x) ? string(x) : @sprintf("%.10g", x))
csv_field(x::Integer) = string(x)
csv_field(x::Nothing) = ""
csv_field(x::Missing) = ""
csv_field(x) = csv_quote(string(x))

"""Quote a field only when it needs it, so ordinary rows stay plain text."""
function csv_quote(s::AbstractString)
    clean = replace(s, r"[\r\n]+" => " ")
    return any(c -> c in (',', '"'), clean) ? '"' * replace(clean, '"' => "\"\"") * '"' : clean
end

"""An open results file with a fixed schema.

Refuses to append to a file whose header is not the current schema. Silently
appending rows in one shape under a header of another is the failure mode worth
guarding: the file still parses, the columns simply mean something else.
"""
mutable struct ResultsWriter
    io::IO
    path::String
    columns::Vector{Symbol}
end

function ResultsWriter(path::AbstractString, columns::Vector{Symbol}; overwrite::Bool=false,
                       replace_cases=nothing)
    mkpath(dirname(path))
    header = join(string.(columns), ",")
    has_rows = isfile(path) && filesize(path) > 0
    # `--overwrite` means "do not resume", not "throw the file away". Given the
    # set of cases this run will measure, drop exactly those rows and keep the
    # rest: re-measuring one stubborn case after a multi-hour sweep must not cost
    # the sweep. Without a case set there is nothing to key on, so it falls back
    # to truncating, which is the right reading of `--overwrite` on a full grid.
    keeping = has_rows && (!overwrite || replace_cases !== nothing)
    if keeping
        existing = strip(open(readline, path))
        existing == header || error(
            "$(basename(path)) has header\n  $existing\nbut this harness writes\n  $header\n" *
            "Move the file aside or pass --overwrite; appending would produce a file whose rows " *
            "and header disagree.")
        overwrite && drop_cases!(path, header, columns, replace_cases)
    end
    io = open(path, keeping ? "a" : "w")
    keeping || (println(io, header); flush(io))
    return ResultsWriter(io, path, columns)
end

"""Rewrite `path` without the rows belonging to `cases`.

Splits on commas without honouring quotes, for the same reason [`read_column`](@ref)
does: the only column that can contain one is `note`, and every schema keeps it
last, so no key column can be displaced by it.

Writes a sibling file and moves it into place, so an interrupted rewrite leaves
the original results intact rather than a half-copied file.
"""
function drop_cases!(path::AbstractString, header::AbstractString,
                     columns::Vector{Symbol}, cases)
    ci = findfirst(==(:case_id), columns)
    ci === nothing && return 0
    wanted = Set(String.(cases))
    isempty(wanted) && return 0
    kept, dropped = String[], 0
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue
        isempty(strip(line)) && continue
        f = split(line, ",")
        if length(f) >= ci && String(f[ci]) in wanted
            dropped += 1
            continue
        end
        push!(kept, line)
    end
    tmp = path * ".rewriting"
    open(tmp, "w") do io
        println(io, header)
        for line in kept
            println(io, line)
        end
    end
    mv(tmp, path; force=true)
    return dropped
end

"""Append one row, given as a NamedTuple covering the schema."""
function write_row!(w::ResultsWriter, row::NamedTuple)
    missing_cols = setdiff(w.columns, collect(keys(row)))
    isempty(missing_cols) || error("row is missing $(join(missing_cols, ", ")) for $(basename(w.path))")
    println(w.io, join((csv_field(row[c]) for c in w.columns), ","))
    flush(w.io)
    return w
end

close_writer!(w::ResultsWriter) = close(w.io)

"""Read one column of a CSV, keyed by another. Returns a `Dict{String,String}`.

Deliberately a hand-rolled reader rather than a CSV dependency: these files are
written by this harness in a known shape, and the Julia stages should not carry
a parser they need for nothing else.

Splits on commas without honouring quotes, which is sound only because the one
column that can ever contain one — `note`, holding an exception message — is last
in every schema. Keep it last.
"""
function read_column(path::AbstractString, key::Symbol, value::Symbol)
    out = Dict{String,String}()
    isfile(path) || return out
    open(path, "r") do io
        header = split(strip(readline(io)), ",")
        ki = findfirst(==(string(key)), header)
        vi = findfirst(==(string(value)), header)
        (ki === nothing || vi === nothing) && return out
        for line in eachline(io)
            isempty(strip(line)) && continue
            f = split(line, ",")
            length(f) >= max(ki, vi) || continue
            out[f[ki]] = f[vi]
        end
    end
    return out
end

"""Case ids whose sweep ran to a conclusion.

A case is finished when one of its rows carries a `stop_reason` — the ladder
reached the accuracy target, timed out, or was exhausted. Keying resume on the
mere presence of a row, as an earlier version of this harness did, silently
treats a case interrupted halfway up its ladder as complete, and a partial
ladder is indistinguishable from a converged one once it is in the file.
"""
function completed_cases(path::AbstractString; knob_name=nothing)
    if !isnothing(knob_name)
        # Every sweep shares one header, so a file swept on another axis resumes
        # cleanly and measures nothing. Refuse it instead.
        seen = Set(String(strip(k)) for k in values(read_column(path, :case_id, :knob_name))
                   if !isempty(strip(k)))
        if !isempty(setdiff(seen, Set([String(knob_name)])))
            axes = join(sort(collect(seen)), "/")
            error("$path was swept on $axes, this run sweeps $knob_name. Rerun with --overwrite, " *
                  "or move the file aside; resuming would mix two axes under one header.")
        end
    end
    done = Set{String}()
    for (case, reason) in read_column(path, :case_id, :stop_reason)
        isempty(strip(reason)) || push!(done, case)
    end
    return done
end

"""Case ids present at all — the resume rule for stages that write one row each."""
function measured_cases(path::AbstractString)
    return Set(keys(read_column(path, :case_id, :case_id)))
end

# ── Ground truth ─────────────────────────────────────────────────────

references_path(cfg::Config) = joinpath(cfg.root, "results", "references.csv")

"""Cached ground-truth tortuosity by case id."""
function read_references(cfg::Config)
    refs = Dict{String,Float64}()
    for (case, tau) in read_column(references_path(cfg), :case_id, :tau_ref)
        val = tryparse(Float64, tau)
        val === nothing || (refs[case] = val)
    end
    return refs
end

"""Append one reference the moment it is solved.

Written immediately rather than at the end of the stage because a reference is
by far the most expensive thing the campaign computes — at the largest sizes it
is hours of single-precision-free `Float64` work — and an interrupted run that
had to recompute one it had already paid for would be the costliest possible
failure.
"""
function record_reference(cfg::Config, case::Case, entry, tau_ref, reltol, elapsed)
    w = ResultsWriter(references_path(cfg), REFERENCE_COLUMNS)
    try
        write_row!(w, (; case_id=case.id, size=case.size, blobiness=case.blobiness,
                       porosity_target=case.porosity, porosity=entry.porosity,
                       nnodes=entry.nnodes, tau_ref=tau_ref, reltol=reltol,
                       solve_time_s=elapsed, host=gethostname(),
                       measured_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS")))
    finally
        close_writer!(w)
    end
    return tau_ref
end

"""Record what produced a batch of rows, so results from two machines stay apart.

Timings are only comparable within one machine and one software stack. The
campaign spans a laptop and a rented GPU host by design, which makes this the
difference between a dataset that can be audited and one that cannot.
"""
function record_environment(cfg::Config; stage, tool, device, variant="", accelerator="", notes="")
    path = joinpath(results_output_dir(cfg), "environment.csv")
    w = ResultsWriter(path, ENVIRONMENT_COLUMNS)
    try
        write_row!(w, (; measured_at=Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"),
                       host=gethostname(), stage, tool, device, variant,
                       runtime="julia", runtime_version=string(VERSION),
                       cpu_threads=Threads.nthreads(), accelerator, notes))
    finally
        close_writer!(w)
    end
    return path
end
