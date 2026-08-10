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

## Directories that are easy to confuse

- `bench/` — this package's benchmark harness. Bench scripts run under `--project=bench`, never `--project=benchmarks`.
- `benchmarks/` — the JOSS submission's directory. **Do not modify anything under it** unless the task is explicitly JOSS work.

## Testing

Full suite: `Pkg.test()`, currently **11576 assertions**, ~223 s. It exercises the CUDA GPU path on this machine, so a green local run covers CPU and GPU. Never weaken, loosen or skip a test to make a change pass — if a test fails, the change is wrong.

## Releasing

Release flow: bump `version` in `Project.toml` on `main` (commit message: `chore: release vX.Y.Z`), then post a `@JuliaRegistrator register` comment on that commit. Registrator opens a PR against `JuliaRegistries/General`; the AutoMerge bot then decides whether to merge without human review.

**Release-notes keyword requirement.** AutoMerge classifies any version bump that changes the leftmost non-zero component as breaking, and refuses to auto-merge a breaking release whose notes don't contain the literal substring `breaking` or `changelog`. For this package (pre-1.0) every patch bump triggers the rule:

- `0.0.6 → 0.0.7` — `y` is the leftmost non-zero, so this **is** breaking by Julia SemVer.
- `0.5.2 → 0.5.3` (hypothetical, post-`0.1`) — patch under a non-zero `x`, **not** breaking.
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

After AutoMerge passes, the General PR enters a 3-day registry waiting period and then merges. TagBot then tags `vX.Y.Z` and creates the GitHub release without further action.
