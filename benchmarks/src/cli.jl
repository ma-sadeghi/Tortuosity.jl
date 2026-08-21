# The command-line surface shared by every Julia stage.
#
# One parser, one set of selection flags, and — deliberately — an error on any
# argument the script does not know. A silently ignored typo is the expensive
# failure here: `--sweep-iter` instead of `--sweep-iters` once meant a run that
# looked healthy for hours and produced the wrong sweep.

struct Args
    flags::Set{String}
    options::Dict{String,String}
end

"""Selection flags every stage accepts, so a case list is chosen the same way."""
const SELECTION_OPTIONS = ["grid", "sizes", "porosities", "blobiness", "cases"]
const SELECTION_FLAGS = ["dry-run", "overwrite", "list-cases"]

"""Parse `--flag` and `--option=value` arguments, rejecting anything unknown."""
function parse_args(argv=ARGS; flags=String[], options=String[])
    known_flags = Set(vcat(SELECTION_FLAGS, flags, "help"))
    known_options = Set(vcat(SELECTION_OPTIONS, options))
    seen_flags, seen_options = Set{String}(), Dict{String,String}()
    for arg in argv
        startswith(arg, "--") || error("unexpected argument \"$arg\"; every argument is a --flag or --option=value")
        body = arg[3:end]
        if occursin('=', body)
            name, value = split(body, '='; limit=2)
            name in known_options || error("unknown option --$name; this stage accepts " *
                                           join(sort(collect(known_options)), ", "))
            seen_options[name] = value
        else
            body in known_flags || error("unknown flag --$body; this stage accepts " *
                                         join(sort(collect(known_flags)), ", "))
            push!(seen_flags, body)
        end
    end
    return Args(seen_flags, seen_options)
end

flag(a::Args, name::AbstractString) = name in a.flags
option(a::Args, name::AbstractString, default) = get(a.options, name, default)

function option_list(a::Args, name::AbstractString, parse_one)
    raw = get(a.options, name, "")
    return isempty(raw) ? nothing : [parse_one(strip(s)) for s in split(raw, ",") if !isempty(strip(s))]
end

"""The cases this invocation should run, cheapest first.

Filters compose: `--sizes=200,400 --blobiness=1.0` selects the intersection.
`--cases=` names grid points outright and overrides the rest, which is how a
single stubborn case gets re-measured without disturbing the others.
"""
function select_cases(cfg::Config, a::Args)
    grid = option(a, "grid", cfg["campaign"]["grid"])
    cases = case_grid(cfg, grid)

    wanted_ids = option_list(a, "cases", String)
    if wanted_ids !== nothing
        by_id = Dict(c.id => c for c in cases)
        unknown = [id for id in wanted_ids if !haskey(by_id, id)]
        isempty(unknown) || error("no such case in grid \"$grid\": $(join(unknown, ", "))")
        return [by_id[id] for id in wanted_ids]
    end

    keep_sizes = option_list(a, "sizes", s -> parse(Int, s))
    keep_blob = option_list(a, "blobiness", s -> parse(Float64, s))
    keep_por = option_list(a, "porosities", s -> parse(Float64, s))
    keep_sizes === nothing || (cases = filter(c -> c.size in keep_sizes, cases))
    keep_blob === nothing || (cases = filter(c -> any(≈(c.blobiness), keep_blob), cases))
    keep_por === nothing || (cases = filter(c -> any(≈(c.porosity), keep_por), cases))
    isempty(cases) && error("no cases match the selection in grid \"$grid\"")
    return cases
end

"""Narrow `cases` to the structures the memory stage measures, unless asked otherwise.

Memory tracks pore count, which at a fixed porosity barely moves with blobiness,
so measuring every structure would spend processes re-measuring one curve. An
explicit `--blobiness` always wins.
"""
function restrict_memory_blobiness(cfg::Config, a::Args, cases::AbstractVector{Case})
    haskey(a.options, "blobiness") && return cases
    wanted = Float64.(get(cfg["memory"], "blobinesses", cfg["image"]["blobinesses"]))
    kept = filter(c -> any(≈(c.blobiness), wanted), cases)
    return isempty(kept) ? cases : kept
end

"""Print the selected case ids, one per line, and nothing else.

Exists so a shell can drive one process per case. The memory stage needs that:
peak resident set is only that case's peak in a process that has not already
faulted in comparable pages, and an allocator that reuses them — torch's on the
CPU especially — makes a within-process reading report page faults rather than
footprint.
"""
function list_cases(cases::AbstractVector{Case})
    for c in cases
        println(c.id)
    end
    return nothing
end

"""Print the selected cases and return, for checking a plan before paying for it."""
function report_plan(cases::AbstractVector{Case}, stage::AbstractString; skipped=String[])
    println("$stage would run $(length(cases)) case(s):")
    for c in cases
        @printf("  %-18s N=%-5d blobiness=%.2f porosity=%.2f\n", c.id, c.size, c.blobiness, c.porosity)
    end
    isempty(skipped) || println("skipping $(length(skipped)) already complete: $(join(skipped, ", "))")
    return nothing
end

"""Check the process got the thread count the campaign configured.

Julia fixes its thread count at startup, so this can only verify. It is worth
verifying: `julia --project=.` starts with one thread whether or not that was
intended, and a CPU comparison run at a different count than the rows already in
the file is not comparable with them.

With the campaign's `"auto"` setting, what this catches is the case that would
otherwise be invisible — a run that never received `-t auto`, quietly measuring
one thread against tools that took every core the machine has.
"""
function check_threads(cfg::Config; expected=cfg["cpu"]["threads"])
    actual = Threads.nthreads()
    if expected == "auto"
        actual > 1 && return true
        @warn "configured to use the whole machine but this process has one thread — pass " *
              "`-t auto`; these rows are not comparable with ones measured on every core"
        return false
    end
    actual == Int(expected) && return true
    @warn "thread count does not match the configured CPU budget — this run is not comparable " *
          "with rows measured at the configured count" configured = expected actual
    return false
end
