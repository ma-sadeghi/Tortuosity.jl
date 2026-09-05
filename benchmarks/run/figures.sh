#!/usr/bin/env bash
# Redraw every figure from the result CSVs.
#
#   ./run/figures.sh
#   ./run/figures.sh --only=memory,summary --no-publish
#
# Reads `results/` and nothing else — no images, no solvers, no GPU. This is the
# whole point of keeping the stages apart: data is generated once, on whatever
# machine can afford it, and the figures are rebuilt as often as the paper needs
# on whatever machine is to hand.

set -eu
cd "$(dirname "$0")/.."

PIXI="${PIXI:-}"
if [ -z "$PIXI" ]; then
  if command -v pixi > /dev/null 2>&1; then PIXI=pixi
  elif [ -x "$HOME/.pixi/bin/pixi" ]; then PIXI="$HOME/.pixi/bin/pixi"
  elif [ -x "$HOME/.pixi/bin/pixi.exe" ]; then PIXI="$HOME/.pixi/bin/pixi.exe"
  else echo "pixi not found; set PIXI=/path/to/pixi" >&2; exit 2
  fi
fi

$PIXI run python make_figures.py "$@"
