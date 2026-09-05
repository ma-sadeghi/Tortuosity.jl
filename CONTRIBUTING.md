# Contributing to Tortuosity.jl

Thank you for your interest in contributing to Tortuosity.jl.

## Code of Conduct

This project is governed by the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you are expected to uphold it.

## Reporting issues

If you find a bug or have a feature request, please open an issue on [GitHub](https://github.com/ma-sadeghi/Tortuosity.jl/issues). Include:

- A minimal reproducible example
- The Julia version and OS you are using
- The output of `Pkg.status("Tortuosity")`

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes. Add tests for new functionality.
3. Run the test suite to make sure nothing is broken:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```
4. Open a pull request against `main`.

## Development setup

```bash
git clone https://github.com/ma-sadeghi/Tortuosity.jl.git
cd Tortuosity.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Project layout and where things are documented

| Path | What it holds |
| --- | --- |
| `src/`, `ext/` | The package. GPU backends are package extensions under `ext/`. |
| `test/` | The test suite that `Pkg.test()` runs. |
| `docs/src/` | The user-facing documentation site; `docs/make.jl` builds it. |
| `docs/design.md` | Maintainer documentation: the non-obvious design decisions, with the measurements behind them. Read it before changing the solver, the preconditioner or the memory layout. |
| `docs/plans/` | Design and execution plans, kept after they finish so that what was tried and what was rejected stay on the record. `docs/plans/README.md` explains the convention. |
| `benchmarks/` | The cross-tool benchmark harness. Start with `benchmarks/README.md`, and read `benchmarks/run/ORCHESTRATION.md` before driving a campaign. |
| `AGENTS.md` | The repository's working rules: how to run Julia here, how to benchmark, how a release is cut. Written for coding agents, and the short version for everyone else. |
| `.JuliaFormatter.toml` | The formatting config (blue style, 4-space indent, 92-column margin). Match it in new code. |

## Questions

If you have questions about the code or how to contribute, open a discussion or issue on GitHub.
