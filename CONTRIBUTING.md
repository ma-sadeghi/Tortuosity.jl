# Contributing to Tortuosity.jl

Thank you for your interest in contributing to Tortuosity.jl.

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

## Questions

If you have questions about the code or how to contribute, open a discussion or issue on GitHub.
