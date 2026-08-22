#!/usr/bin/env bash
# Prepare a machine to run the campaign: resolve both environments, then check
# that the accelerator and every tool actually work before anything is measured.
#
#   ./run/setup.sh
#
# Written to be run on a freshly rented machine as much as on a laptop. It
# changes nothing outside the two package environments.

set -eu
cd "$(dirname "$0")/.."

JULIA="${JULIA:-julia}"
PIXI="${PIXI:-}"
if [ -z "$PIXI" ]; then
  if command -v pixi > /dev/null 2>&1; then PIXI=pixi
  elif [ -x "$HOME/.pixi/bin/pixi" ]; then PIXI="$HOME/.pixi/bin/pixi"
  elif [ -x "$HOME/.pixi/bin/pixi.exe" ]; then PIXI="$HOME/.pixi/bin/pixi.exe"
  else echo "pixi not found — install it from https://pixi.sh, or set PIXI=" >&2; exit 2
  fi
fi

echo "=== julia environment"
# `--project=.` and not the package above it: `bench_tortuosity.jl` needs CUDA,
# which is a *weak* dependency of Tortuosity.jl. Adding it to the package's own
# Project.toml would turn the CUDA extension into a hard dependency of the
# released package, which is the opposite of what the extension is for.
#
# `develop` before `instantiate`: the manifest is committed so that every other
# package resolves to the same version everywhere, but the `Tortuosity` entry
# records an absolute path to wherever the manifest was last written. That path
# is meaningless on any other machine, so it is re-pointed at the parent
# directory first. Idempotent — on the machine that wrote it, it rewrites the
# same path.
$JULIA --startup-file=no --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'

echo "=== python environment"
# taufactor is checked out under vendor/ at a pinned commit and installed
# *editable*, so that a patch to the fork takes effect without a reinstall.
# Getting that wrong is silent — the benchmark imports a stale copy and reports
# numbers for code nobody edited — so the check below asserts the import
# resolves into the working tree.
#
# A clone rather than a git submodule: a gitlink makes this repository's git tree
# hash disagree with the one Pkg recomputes from an installed package's files, so
# a submodule here would cost every Linux and macOS user of Tortuosity.jl a
# warning and a git-clone fallback on `Pkg.add`. See README, "The taufactor fork".
TAUFACTOR_URL=https://github.com/ma-sadeghi/taufactor.git
TAUFACTOR_COMMIT=a4bc5f9ed5a9d92f315d64e3d0872d1673bc0c94
if [ ! -e vendor/taufactor/.git ]; then
  mkdir -p vendor
  git clone "$TAUFACTOR_URL" vendor/taufactor
fi
# Fetch only when the pin is missing, so a machine that already has it can run
# this offline.
if ! git -C vendor/taufactor cat-file -e "$TAUFACTOR_COMMIT^{commit}" 2> /dev/null; then
  git -C vendor/taufactor fetch --quiet origin
fi
git -C vendor/taufactor checkout --quiet --detach "$TAUFACTOR_COMMIT"
$PIXI install

echo "=== checks"
$JULIA --startup-file=no --project=. -e '
    using CUDA
    @info "julia" version=VERSION threads=Threads.nthreads()
    if CUDA.functional()
        @info "cuda" device=CUDA.name(CUDA.device()) memory_gib=round(CUDA.total_memory()/2^30; digits=1)
    else
        @warn "CUDA is not functional — GPU stages will refuse to run"
    end'
$PIXI run python -c "
import inspect, pathlib, sys, torch, taufactor
print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')
src = pathlib.Path(taufactor.__file__ or '')
print('taufactor imports from', src or '(namespace package — the install is broken)')
# The fork's patches are what the campaign measures, so an import that resolves
# anywhere but the working tree, or a Solver missing the checkpoint argument,
# means the numbers would describe code nobody edited.
ok = src.is_file() and 'vendor' in src.parts and \
     'checkpoints' in inspect.signature(taufactor.Solver.solve).parameters
print('vendored fork live and patched:', ok)
sys.exit(0 if ok else 1)"
# pumapy is checked in its own process: it and torch each link an OpenMP runtime
# and abort on the duplicate when imported together.
$PIXI run python -c "
import pumapy, scipy
print('pumapy ok, scipy', scipy.__version__)"

echo
echo "ready. Validate the machinery before paying for a large machine:"
echo "  ./run/campaign.sh --grid=smoke"
echo "then run the real grid with:"
echo "  ./run/campaign.sh --grid=full"
