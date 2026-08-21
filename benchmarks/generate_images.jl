# Build the shared image store every stage and every tool reads from.
#
#   julia --project=. generate_images.jl [--grid=smoke] [--sizes=200,400] [--dry-run]
#
# Append-only: an image already in the store is left alone, so this is safe to
# re-run and safe to interrupt. `--overwrite` rebuilds the selected cases and
# `--verify` re-hashes what is there without building anything.

include(joinpath(@__DIR__, "src", "BenchHarness.jl"))
using .BenchHarness
using Printf

const BH = BenchHarness

args = parse_args(; flags=["verify"])
cfg = load_config()
cases = select_cases(cfg, args)
overwrite = flag(args, "overwrite")
manifest = read_manifest(cfg)

if flag(args, "dry-run")
    pending = [c for c in cases if overwrite || !haskey(manifest, c.id) || !isfile(image_path(cfg, c))]
    BH.report_plan(pending, "generate_images")
    exit(0)
end

if flag(args, "verify")
    bad, missing_ids = String[], String[]
    for case in cases
        # An image that is not there cannot match the manifest, and a partial
        # copy onto a rented machine is the failure this mode exists to catch —
        # so a missing file is reported rather than skipped.
        if !haskey(manifest, case.id) || !isfile(image_path(cfg, case))
            push!(missing_ids, case.id)
            continue
        end
        try
            load_image(cfg, case)
            @info "ok $(case.id)"
        catch e
            push!(bad, case.id)
            @error "$(case.id) failed verification" exception = e
        end
    end
    isempty(missing_ids) ||
        error("$(length(missing_ids)) of $(length(cases)) selected image(s) are absent: " *
              join(missing_ids, ", "))
    isempty(bad) || error("$(length(bad)) image(s) do not match the manifest: $(join(bad, ", "))")
    @info "all selected images match the manifest" n = length(cases)
    exit(0)
end

@info "Generating images" grid = option(args, "grid", cfg["campaign"]["grid"]) n_cases = length(cases) overwrite
for (i, case) in enumerate(cases)
    cached = haskey(manifest, case.id) && isfile(image_path(cfg, case))
    if cached && !overwrite
        @info @sprintf("[%d/%d] %s cached", i, length(cases), case.id)
        continue
    end
    t = @elapsed entry = ensure_image!(cfg, case; force=overwrite)
    # A case whose percolating pore space is empty is a real outcome of the grid,
    # not a failure: coarse structures at low porosity can leave no path across
    # the domain at all. It is recorded with zero nodes and every later stage
    # skips it, so the grid stays complete and the gap stays explained.
    msg = @sprintf("[%d/%d] %s  porosity=%.4f  nodes=%d  (%.1fs)",
                   i, length(cases), case.id, entry.porosity, entry.nnodes, t)
    entry.nnodes == 0 ? (@warn "$msg — no percolating pore space") : (@info msg)
end
@info "Image store ready" dir = BH.imagedir(cfg) manifest = manifest_path(cfg)
