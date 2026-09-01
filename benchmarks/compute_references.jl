# Ground truth for every case: a Float64 CPU solve at the configured tolerance.
#
#   julia -t auto --project=. compute_references.jl [--grid=smoke] [--sizes=200]
#
# Deliberately its own stage and its own process. Ground truth depends only on
# the image, so it is computed once and reused by every tool, on every device,
# for the life of the dataset — and it is the most expensive thing the campaign
# does, so it must survive an interruption. Each value is appended to
# `results/references.csv` the moment it is solved.
#
# Run this with as many threads as the machine has. A reference is a value, not
# a timing, so its thread count cannot change the answer — which is exactly why
# the fairness argument that pins the sweeps to one thread does not apply here.

include(joinpath(@__DIR__, "src", "BenchHarness.jl"))
using .BenchHarness
using Logging
using Printf
using Tortuosity

const BH = BenchHarness

args = parse_args()
cfg = load_config()
cases = select_cases(cfg, args)
axis = Symbol(cfg["campaign"]["axis"])
reltol = Float64(cfg["reference"]["reltol"])
overwrite = flag(args, "overwrite")

manifest = read_manifest(cfg)
known = overwrite ? Set{String}() : Set(keys(read_references(cfg)))
missing_images = [c.id for c in cases if !haskey(manifest, c.id)]
isempty(missing_images) || error("no image for $(join(missing_images, ", ")) — run generate_images.jl first")

pending = [c for c in cases if !(c.id in known) && manifest[c.id].nnodes > 0]
empty_cases = [c.id for c in cases if manifest[c.id].nnodes == 0]

if flag(args, "dry-run")
    BH.report_plan(pending, "compute_references"; skipped=sort(collect(known)))
    isempty(empty_cases) || println("no percolating pore space, nothing to solve: $(join(empty_cases, ", "))")
    exit(0)
end

record_environment(cfg; stage="references", tool="tortuosity", device="cpu",
                   variant="matrixfree", notes="Float64 reference at reltol=$reltol")

@info "Computing references" n_pending = length(pending) already = length(known) reltol threads = Threads.nthreads()

for (i, case) in enumerate(pending)
    entry = manifest[case.id]
    @info @sprintf("[%d/%d] %s  N=%d  nodes=%d", i, length(pending), case.id, case.size, entry.nnodes)
    img = load_image(cfg, case)
    # Matrix-free and preconditioned. Neither changes the converged answer — the
    # operator is the same one and the preconditioner only changes the path to
    # the same fixed point — but together they turn a reference at the largest
    # sizes from a job of hours into one of minutes. What makes this ground
    # truth is `Float64` and the tolerance, and both are untouched.
    elapsed = @elapsed begin
        sim = SteadyDiffusionProblem(
            img; axis=axis, gpu=false, matrixfree=true, warn_nonpercolating=false,
        )
        sol = with_logger(ConsoleLogger(stderr, Logging.Error)) do
            solve(sim, KrylovJL_CG(); verbose=false, reltol=reltol, precond=:auto)
        end
        tau_ref = tortuosity(sol.u, sim)
    end
    record_reference(cfg, case, entry, tau_ref, reltol, elapsed)
    @info @sprintf("      tau_ref = %.6f  (%.1fs)", tau_ref, elapsed)
    sim = sol = img = nothing
    GC.gc(true)
end

@info "References written to $(references_path(cfg))"
