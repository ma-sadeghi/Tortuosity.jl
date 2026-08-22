# AGENTS.md

Agent instructions for Tortuosity.jl. Canonical for every agent; Claude Code reads it through `CLAUDE.md`, which imports this file.

## Running Julia

**Do not invoke `julia` from Bash or PowerShell** to evaluate code, run tests, or run a script. Use the persistent Julia MCP session (`julia_eval`), and load the **`julia-workflow`** skill for the full rules before doing Julia work. Measured on this package: a fresh process costs **173.9 s** from a source edit to a green test file; the warm session costs **1.13 s**.

Three things specific to this repository:

- **Pass `env_path` as `<repo>/test/`** — the trailing `test/` makes the server run `using TestEnv; TestEnv.activate()` and treat the parent as the project root. This package uses `[extras]`/`[targets]` rather than a `test/Project.toml`, so `--project=.` cannot `using Test`. Getting this wrong is the usual reason the session "doesn't work".
- **After editing any `.jl`, reload and verify the reload applied** before trusting the result. A missed reload returns results for code that was never compiled, and reads exactly like a passing test. Assert a value the edit changed.
- **`julia_restart` with the same `env_path`** after a struct or `const` redefinition error, a world-age error, a branch switch, or any `Project.toml` / `Manifest.toml` change.

A fresh process is correct for exactly four things: peak-memory or size-ceiling measurements, a published benchmark table someone else must reproduce, the full `Pkg.test()` release gate, and anything measuring startup or precompile time itself. Everything else — including day-to-day benchmarking — belongs in the warm session, warming up and discarding the first run. Compile time is the cost of using Julia, not a property of the code under test, so a timing that includes it measures the wrong thing.

Do not lower the optimisation level (`-O1`, `-O0`) as a speed lever. It changes floating-point codegen; `test/test_transient.jl` asserts exact float equality and flips from pass to fail at `-O1`.

## Benchmarking

`benchmarks/` is the only benchmark harness. It compares Tortuosity.jl against taufactor and PuMA over domain size, porosity and microstructure, on both devices, for time and for memory, and it is part of the JOSS submission. Run it under `--project=benchmarks`; read `benchmarks/README.md` before changing anything in it, and `benchmarks/run/ORCHESTRATION.md` before driving a campaign.

There used to be a second directory, `bench/`, holding a frozen copy of the pre-KernelAbstractions implementation and the old-versus-new comparison built around it. It was migration scaffolding, the migration landed, and it was deleted along with `test/test_gpu_parity.jl`. Correctness is covered by properties and golden values in `test/`, which do not need a duplicate implementation to compare against; performance is covered end to end by `benchmarks/`. Do not reintroduce a second harness — a performance assertion inside `Pkg.test()` is flaky by construction, and a per-machine baseline file nobody re-measures is worse than no baseline.

## Testing

Full suite: `Pkg.test()`, roughly three minutes. It exercises the CUDA GPU path on this machine, so a green local run covers CPU and GPU. Never weaken, loosen or skip a test to make a change pass — if a test fails, the change is wrong.

## Releasing

Release flow: bump `version` in `Project.toml` on `main` (commit message: `chore: release vX.Y.Z`), then post a `@JuliaRegistrator register` comment on that commit. Registrator opens a PR against `JuliaRegistries/General`; the AutoMerge bot then decides whether to merge without human review.

**Release-notes keyword requirement.** AutoMerge classifies any version bump that changes the leftmost non-zero component as breaking, and refuses to auto-merge a breaking release whose notes don't contain the literal substring `breaking` or `changelog`. Since `v0.1.0` that component is `y`, so a minor bump triggers the rule and a patch bump does not:

- `0.1.0 → 0.2.0` — `y` is the leftmost non-zero, so this **is** breaking by Julia SemVer.
- `0.1.0 → 0.1.1` — patch under a non-zero `y`, **not** breaking.
- `0.0.6 → 0.0.7` (historical, pre-`0.1`) — `z` was the leftmost non-zero, so back then every patch bump was breaking.
- `1.2.3 → 1.2.4` (hypothetical, post-`1.0`) — patch, **not** breaking.

So the Registrator trigger comment must always include release notes with one of those keywords — even just `No breaking API changes.` Format:

```markdown
@JuliaRegistrator register

Release notes:

## Bug fixes
- ...

No breaking API changes.
```

If the General PR gets blocked because notes were missing, **re-post the same trigger comment with notes added on the same release commit**. Registrator updates the existing General PR in place; do not bump the version again.

After AutoMerge passes, a new version of an already-registered package merges in the next round — about 20 minutes, measured at 17.6 for v0.1.0. The three-day waiting period applies to registering a *brand-new package*, not to a version bump.

**Do not comment on the General PR.** Any comment blocks auto-merging unless its body contains the literal `[noblock]`, and an existing blocking comment can be edited to add it.

TagBot then tags `vX.Y.Z` and creates the GitHub release without further action. It has no cron here: it fires on the `issue_comment` JuliaTagBot leaves once the registration merges, and `workflow_dispatch` (input `lookback`, in days) is the manual fallback if that comment never arrives.
