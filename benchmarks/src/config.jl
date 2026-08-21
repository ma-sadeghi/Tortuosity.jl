# The campaign configuration and the case grid it defines.

"""Directory holding `config.toml`, `data/`, `results/` and the entry points."""
benchdir() = normpath(joinpath(@__DIR__, ".."))

"""One benchmark image: a point on the (size, blobiness, porosity) grid.

`porosity` is the *target* the image was thresholded to, not what it ended up
with — trimming non-percolating clusters removes pore voxels, so the realised
value is always a little lower and differs between sizes. Every stage keys on
the target, which is exact and identical everywhere, rather than on the realised
value, which is a float that has to survive a round trip through three CSV
writers in two languages before two tools can agree they solved the same image.
"""
struct Case
    id::String
    size::Int
    blobiness::Float64
    porosity::Float64
end

"""The parsed `config.toml`, plus the paths derived from its location."""
struct Config
    raw::Dict{String,Any}
    root::String
end

Base.getindex(cfg::Config, key::AbstractString) = cfg.raw[key]

"""Identifier for a grid point, e.g. `n200_b100_p020`.

Blobiness and porosity are written as hundredths so the identifier is a plain
string with no decimal point: at two decimals it is exact for every value the
campaign uses, and it stays safe as a filename and as a CSV field.
"""
function case_id(size::Integer, blobiness::Real, porosity::Real)
    return @sprintf("n%d_b%03d_p%03d", size, round(Int, 100blobiness), round(Int, 100porosity))
end

Case(size, blobiness, porosity) =
    Case(case_id(size, blobiness, porosity), Int(size), Float64(blobiness), Float64(porosity))

"""Read `config.toml`, defaulting to the one beside this harness."""
function load_config(path=joinpath(benchdir(), "config.toml"))
    isfile(path) || error("no config at $path")
    return Config(TOML.parsefile(path), dirname(abspath(path)))
end

"""Sizes of the named grid, e.g. `sizes(cfg, "smoke")`."""
function sizes(cfg::Config, grid::AbstractString)
    grids = cfg["grid"]
    haskey(grids, grid) || error("unknown grid \"$grid\"; config defines $(join(sort(collect(keys(grids))), ", "))")
    return Int.(grids[grid])
end

"""Every case of a named grid, ordered cheapest first.

Ordering matters more than it looks: every stage resumes from its own results
file, so a run that is interrupted — or deliberately stopped once a rented
machine has cost enough — leaves the small cases complete rather than a scatter
of half-finished large ones.
"""
function case_grid(cfg::Config, grid::AbstractString=cfg["campaign"]["grid"])
    img = cfg["image"]
    cases = [Case(n, b, p)
             for n in sizes(cfg, grid)
             for b in Float64.(img["blobinesses"])
             for p in Float64.(img["porosities"])]
    return sort(cases; by=c -> (c.size, c.blobiness, c.porosity))
end

"""Log-spaced iteration ladder from the config, deduplicated and ascending."""
function iteration_ladder(cfg::Config)
    spec = cfg["sweep"]["ladder"]["iters"]
    lo, hi, n = Float64(spec["min"]), Float64(spec["max"]), Int(spec["count"])
    return sort(unique(round.(Int, 10 .^ range(log10(lo), log10(hi); length=n))))
end

"""Log-spaced tolerance ladder from the config, loosest first."""
function tolerance_ladder(cfg::Config)
    spec = cfg["sweep"]["ladder"]["tolerance"]
    lo, hi, n = Float64(spec["min"]), Float64(spec["max"]), Int(spec["count"])
    return 10.0 .^ range(log10(lo), log10(hi); length=n)
end
