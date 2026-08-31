#!/usr/bin/env bash
# Generate benchmark data, unattended. Produces CSVs; draws nothing.
#
#   ./run/campaign.sh --grid=smoke                        # validate the machinery
#   ./run/campaign.sh --grid=full                         # the real thing
#   ./run/campaign.sh --grid=full --stages=images,references
#   ./run/campaign.sh --grid=full --stages=timings --tools=tortuosity --sizes=800,1000
#
# Runs identically on a laptop and on a rented machine — the only difference is
# how long it takes. Figures are deliberately not drawn here: `run/figures.sh`
# reads the CSVs and needs no GPU, so the expensive machine is only ever paying
# for measurement.
#
# Every stage resumes from its own results file, so this can be interrupted and
# re-run without repeating completed work, and stopped early once a rented
# machine has cost enough. Cases run cheapest first for the same reason.
#
# The stages must not overlap. They contend for the same GPU and the same cores,
# and concurrency contaminates the very timings the benchmark exists to measure.

set -u
cd "$(dirname "$0")/.."

GRID="full"
STAGES="images,references,timings,memory"
# PuMA and PoreSpy are left out of the default. Both are CPU only and both are
# slow enough that sweeping them over the size grid costs days for a gap already
# decided at the smallest size, so they are run deliberately and at one size:
# `--tools=puma,porespy --sizes=200`. Naming them here instead would make the
# default campaign the expensive one.
TOOLS="tortuosity,taufactor"
# Which devices the timing and memory stages cover. Sweeping the sizes on the GPU
# first and returning for the CPU later is a normal way to run this: the GPU
# stages carry the headline numbers and cost hours where the CPU ones cost days.
# References ignore this — they are a Float64 CPU solve whatever is being timed.
DEVICES="gpu,cpu"
PASSTHROUGH=""

for arg in "$@"; do
  case "$arg" in
    --grid=*) GRID="${arg#*=}" ;;
    --stages=*) STAGES="${arg#*=}" ;;
    --tools=*) TOOLS="${arg#*=}" ;;
    --devices=*) DEVICES="${arg#*=}" ;;
    --sizes=*|--porosities=*|--blobiness=*|--cases=*|--timeout=*|--overwrite|--dry-run)
      PASSTHROUGH="$PASSTHROUGH $arg" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

JULIA="${JULIA:-julia}"
# No `startup.jl`, ever. A benchmark process must not inherit an interactive REPL
# setup: Revise puts a file watcher beside the thing being timed, and the rest is
# load time charged to every one of the processes the memory stage spawns, one
# per case.
JLFLAGS="--startup-file=no --project=."
PIXI="${PIXI:-}"
if [ -z "$PIXI" ]; then
  if command -v pixi > /dev/null 2>&1; then PIXI=pixi
  elif [ -x "$HOME/.pixi/bin/pixi" ]; then PIXI="$HOME/.pixi/bin/pixi"
  elif [ -x "$HOME/.pixi/bin/pixi.exe" ]; then PIXI="$HOME/.pixi/bin/pixi.exe"
  else echo "pixi not found; set PIXI=/path/to/pixi" >&2; exit 2
  fi
fi

# Read straight from the configuration so what Julia is launched with cannot
# drift from what the result rows claim. "auto" — the campaign's setting — hands
# Julia the machine, which is what the Python tools take by default anyway.
#
# Read from under [cpu] specifically: [reference] carries a `threads` of its own,
# and although both say "auto" today they answer different questions and are
# allowed to diverge.
THREADS=$(sed -n '/^\[cpu\]/,/^\[/p' config.toml |
          sed -n 's/^threads *= *"\{0,1\}\([^"]*\)"\{0,1\} *$/\1/p')
THREADS="${THREADS:-auto}"

LOGS=logs
mkdir -p "$LOGS" results data/images
SELECT="--grid=$GRID$PASSTHROUGH"

# Refuse to start alongside another campaign. Two of these contend for the same
# GPU and the same cores, which silently corrupts every timing rather than
# failing — and the way it happens is not obvious: killing a stage's solver
# process leaves this script alive, so it simply proceeds to the next stage while
# a replacement campaign is already running.
LOCK="$LOGS/.campaign.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2> /dev/null; then
  echo "a campaign is already running (pid $(cat "$LOCK"))." >&2
  echo "stop it first, or remove $LOCK if you are sure it is dead." >&2
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

has_stage() { case ",$STAGES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
has_tool()  { case ",$TOOLS,"  in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
# taufactor names its accelerator `cuda` where the Julia harness says `gpu`. The
# selector speaks one language — `--devices=gpu` covers both.
has_device() {
  case "$1" in cuda) set -- gpu ;; esac
  case ",$DEVICES," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

# Deliberately no `set -e`: a configuration that dies at the largest size still
# leaves every smaller size measured, and the stages after it are independent.
step() {
  local name=$1; shift
  echo "=== $name  ($(date '+%H:%M:%S'))"
  "$@" > "$LOGS/$name.log" 2>&1
  echo "    exit $? — $(date '+%H:%M:%S')  → $LOGS/$name.log"
}

# ── Images ───────────────────────────────────────────────────────────
# Generated on whichever machine runs the campaign rather than copied to it: the
# store reaches tens of gigabytes at the largest sizes, generation is
# deterministic in the configured seed, and every image carries a SHA-256 in the
# manifest so identity across machines is checked rather than assumed.
if has_stage images; then
  step images $JULIA $JLFLAGS generate_images.jl $SELECT
fi

# ── Ground truth ─────────────────────────────────────────────────────
# Its own stage and its own process. A reference depends only on the image, so it
# is computed once and reused by every tool on every device for the life of the
# dataset — and it is the most expensive thing here, so it must not share a heap
# with anything whose timing matters.
#
# Run with every core on purpose. A reference is a value, not a timing, so its
# thread count cannot change the answer, which is exactly why the fairness
# argument that pins the sweeps to one thread does not apply.
if has_stage references; then
  step references $JULIA $JLFLAGS -t auto compute_references.jl $SELECT
fi

# ── Timings ──────────────────────────────────────────────────────────
# Ordered by what the paper cannot do without, because there may not be time for
# all of it. The GPU pair comes first: together they carry the headline timings
# and the operator comparison, which needs the large sizes most and which no
# other tool participates in.
if has_stage timings; then
  if has_tool tortuosity; then
    for device in gpu cpu; do
      has_device $device || continue
      for operator in matrixfree assembled; do
        step "timings_tortuosity_${device}_${operator}" \
          $JULIA $JLFLAGS -t "$THREADS" bench_tortuosity.jl \
          --device=$device --operator=$operator --measure=time $SELECT
      done
    done
  fi
  if has_tool taufactor; then
    for device in cuda cpu; do
      has_device $device || continue
      step "timings_taufactor_${device}" \
        $PIXI run python bench_taufactor.py --device=$device --measure=time $SELECT
    done
  fi
  if has_tool puma && has_device cpu; then
    step timings_puma $PIXI run python bench_puma.py --measure=time $SELECT
  fi
  if has_tool porespy && has_device cpu; then
    step timings_porespy $PIXI run -e porespy python bench_porespy.py --measure=time $SELECT
  fi
fi

# ── Memory ───────────────────────────────────────────────────────────
# A separate stage from the timings, and not a tidiness choice: a timing must not
# be perturbed by a sampler and a peak cannot be measured without one. Cheap next
# to the sweeps — one short fixed-length solve per case — and it needs no ground
# truth, which is what lets it cover sizes the accuracy sweeps cannot afford.
#
# The Julia stages get an extra interactive thread (`-t N,1`) so the sampler has
# somewhere to run while the solver saturates the default pool.
#
# **One process per case, and this is not optional.** Peak resident set is only
# that case's peak in a process that has not already faulted in comparable pages.
# Measured within one process the readings are worthless: torch's CPU allocator
# reuses pages it already holds, so an 80³ solve holding several full-grid
# tensors reported 0.6 MB, and the series was not even monotonic in the domain
# size. It is the host-side twin of the CUDA pool problem — a caching allocator
# defeats a delta measurement — and isolation is the only honest fix.
#
# Each stage below runs `--list-cases` once to enumerate its work, then spawns a
# process per case. Resume still applies, so an interrupted stage picks up where
# it stopped.
isolated() {
  local name=$1; shift
  echo "=== $name  ($(date '+%H:%M:%S'))"
  : > "$LOGS/$name.log"

  local cases status
  cases=$("$@" $SELECT --list-cases 2>> "$LOGS/$name.log")
  status=$?
  if [ $status -ne 0 ] || [ -z "$cases" ]; then
    echo "    could not enumerate cases (exit $status) — see $LOGS/$name.log"
    return
  fi

  case "$PASSTHROUGH" in
    *--dry-run*)
      echo "    would run $(echo "$cases" | wc -w) case(s), one process each"
      return ;;
  esac

  # `--overwrite` drops only the rows of the cases a process is about to measure,
  # so every process in this loop can carry it: each clears its own case and
  # leaves its predecessors' rows alone. This used to hand `--overwrite` to the
  # first process only, because the flag truncated the whole file — which meant
  # every case after the first silently resumed instead of re-measuring.

  # Report per-case exit status. Counting iterations alone cannot distinguish a
  # stage that measured everything from one whose every process died on startup:
  # both print the same count and leave the failure only in the log.
  local n=0 failed=0
  for case_id in $cases; do
    # `$SELECT` is passed alongside `--cases` for `--grid`, which is what tells
    # the harness which grid to resolve the id against. `--cases` overrides the
    # size, porosity and blobiness filters, so repeating them is harmless.
    if ! "$@" $SELECT --cases="$case_id" >> "$LOGS/$name.log" 2>&1; then
      failed=$((failed + 1))
      echo "    !! $case_id exited non-zero"
    fi
    n=$((n + 1))
  done
  if [ "$failed" -gt 0 ]; then
    echo "    $n case(s), $failed FAILED — $(date '+%H:%M:%S')  → $LOGS/$name.log"
  else
    echo "    $n case(s) — $(date '+%H:%M:%S')  → $LOGS/$name.log"
  fi
}

if has_stage memory; then
  if has_tool tortuosity; then
    for device in gpu cpu; do
      has_device $device || continue
      for operator in matrixfree assembled; do
        isolated "memory_tortuosity_${device}_${operator}" \
          $JULIA $JLFLAGS -t "$THREADS,1" bench_tortuosity.jl \
          --device=$device --operator=$operator --measure=memory
      done
    done
  fi
  if has_tool taufactor; then
    for device in cuda cpu; do
      has_device $device || continue
      isolated "memory_taufactor_${device}" \
        $PIXI run python bench_taufactor.py --device=$device --measure=memory
    done
  fi
  if has_tool puma && has_device cpu; then
    isolated memory_puma $PIXI run python bench_puma.py --measure=memory
  fi
  if has_tool porespy && has_device cpu; then
    isolated memory_porespy $PIXI run -e porespy python bench_porespy.py --measure=memory
  fi
fi

echo "campaign complete — results in results/, logs in $LOGS/"
echo "draw the figures with ./run/figures.sh (no GPU needed)"
