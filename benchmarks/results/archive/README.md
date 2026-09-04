# Archived result sets

Datasets measured before a change that made them incomparable with the current ones. They are kept because a before-and-after is the only way to show what a change cost or bought, and because a claim in the documentation rests on one of them. They are **not** part of the published campaign: no figure is drawn from them, no stage resumes against them, and their rows must never be concatenated with `results/timings/` or `results/memory/`.

Each directory is named for the change it precedes.

## `pre-rewrite-2026-08-17/`

Tortuosity.jl measured 2026-08-11 to 2026-08-14 on the code as it stood before the operator and preconditioner work landed, over sizes 200 through 800 on both devices and both operators. Superseded by the re-measurement campaign of 2026-08-17 to 2026-08-18, which added 1000³ and re-ran everything below it. Same schema as the current files, so the two look concatenable and are not — the solver underneath them is different, which is the whole point of keeping these separately. `environment.csv` contains the matching CPU and GPU Tortuosity.jl invocation records from that period.

## `pre-refine-2026-08-20/`

GPU timings and memory from the 2026-08-17 campaign, taken before `solve` began refining single-precision solutions against a double-precision residual by default. Verified byte-identical to the originals before the re-run overwrote them. `environment.csv` contains the 60 matching GPU Tortuosity.jl invocation records from 2026-08-17 to 2026-08-18.

This is the load-bearing one. `docs/src/benchmark.md` cites it as the operator-cost figure against the current files' user-cost figure, and differencing the two is what identified the device guard in `_refine` as the reason the 1000³ memory row at ε = 0.95 is faster than its neighbours: refinement adds exactly 20 bytes per pore node in four cases and exactly 16 in the fifth, where the third allocation did not fit. Deleting this directory would leave both of those results unsupported.

## `pre-hostcg-2026-09-02/`

CPU Tortuosity.jl timings, memory, occupancy, and matching environment rows from the 2026-08-17 to 2026-08-18 campaign, before the benchmark harness switched from `KrylovJL_CG` to the bandwidth-fused `HostCG` implementation. The active result directories accept only variants carrying `-hostcg`; loading one of these archived rows as current data raises an error instead of silently publishing the superseded solver.

## `unmatched-environment.csv`

Three CPU matrix-free timing invocations from 2026-08-21 whose output rows are not present in any committed result file. They remain available as an audit trail but are kept separate rather than assigned to a campaign they cannot substantiate.
