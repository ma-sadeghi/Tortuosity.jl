---
title: Assembled 64-bit index path
created: 2026-08-10
updated: 2026-08-10
status: complete
outcome: The assembled path scales past the Int32 wall on both backends instead of refusing; a prerequisite defect that silently narrowed 64-bit device indices to 32-bit is fixed; measured cost of the wide path is +16.9 % per SpMV.
branch: joss
supersedes: "-"
superseded-by: "-"
related: 2026-08-08-matrix-free-operator.md, 2026-08-08-matrix-path-optimization.md
---

> **Status: complete**, 2026-08-10, on branch `joss`. Closes [#86](https://github.com/ma-sadeghi/Tortuosity.jl/issues/86). `build_steady_system` past **306,783,378 pore voxels** now widens its CSC offsets to `Int64` on GPU as it already did on CPU, rather than raising an `ArgumentError` naming `matrixfree=true`. Only the offsets widen: pore ordinals are bounded by `nnodes` rather than `7 * nnodes`, so the grid-sized `idx` array keeps `Int32` a factor of seven further out, worth **2.3 GB of assembly peak at 850³**. Getting there required fixing a defect the investigation uncovered — `_as_cusparse` converted any non-`Int32` device index array **down** to `Int32`, which past the wall is the exact silent corruption the bound exists to prevent. Measured price of the wide path: one `Float32` SpMV at 400³ goes **3.45 ms → 4.03 ms (+16.9 %)** for **0.93 GiB → 1.85 GiB** of index storage. **139 new assertions**, full suite green including GPU.
>
> **The alternative was weighed and rejected.** The issue proposed `wontfix` on the grounds that `matrixfree=true` covers the regime. It does, and measurably better — 6.31 s / 7.92 GB against 8.33 s / 19.09 GB at 600³, at identical iteration counts — and the documentation now says so in those words. But "the other path is faster" is not a reason for the default path to refuse to run. Amin's ruling, 2026-08-10: the assembled path scales as far as the hardware allows, and **nothing routes to the matrix-free operator automatically**.
>
> **What the investigation got wrong going in:** the suspicion that the wall came from a worst-case `7 * dim³` preallocation. It does not — `rowval` and `nzval` are sized from the measured `colptr[end]`, exact to the entry, and the bound appears only in the *type decision*. And CPU was never blocked at all; the host branch already widened, so the 128 GB machine could always run those sizes. The CPU benchmark CSVs stop at 400³ because the sweeps were capped for time.

# Assembled 64-bit index path plan

Issue [#86](https://github.com/ma-sadeghi/Tortuosity.jl/issues/86) was filed out of the matrix-free campaign, whose progress log recorded the original defect: at 1100³ the GPU branch short-circuited past the index-type check, `colptr` wrapped negative, and `_steady_fill_kernel!` wrote through `@inbounds` at the true offsets — surfacing as an asynchronous `ERROR_ILLEGAL_ADDRESS` that no `try` can catch. That was fixed by applying the bound to both branches, which left the device *refusing* past the wall rather than *scaling* past it. This plan closes that gap.

## Governing constraints

Set by Amin on 2026-08-10, after the analysis below was presented. They are rulings, not preferences.

**1. The assembled path scales; it does not refuse.** It is the default and the explicit choice. Where the hardware has the room, it runs. This is what rejected the `wontfix` the issue itself proposed.

**2. No auto-routing, in either direction.** Selecting between the assembled and matrix-free paths stays the caller's decision. The matrix-free operator may be *recommended* in documentation, never substituted silently. (There was no auto-routing to remove — `matrixfree::Bool=false` was already passed straight through with no size logic. The constraint governs what must not be added.)

**3. CPU must work wherever the RAM allows.** A user who forgoes GPU and has 128 GB should not be blocked by an index-type decision. Investigation found this constraint already satisfied; it remains a constraint on any future change to the bound.

## The problem, measured

Four measurements decided the design. All are reproducible from the working tree.

**The `7 * nnodes` bound is nearly tight.** Measured `nnz / nnodes` for `build_steady_system` on blob images (128³, seed 42, blobiness 1), plus one 400³ point:

| ε | 0.3 | 0.5 | 0.7 | 0.9 | 0.95 | 0.5 @ 400³ |
| --- | --- | --- | --- | --- | --- | --- |
| nnz/nnodes | 6.25 | 6.46 | 6.63 | 6.78 | 6.82 | 6.80 |

Blobs are spatially correlated, so nearly every pore voxel keeps close to six pore neighbours. The worst-case bound is within 3–11 % of the truth across the whole porosity range.

**Allocation was never wasteful.** `rowval` and `nzval` are sized from the measured `colptr[end]` (`src/assembly.jl`), exact to the entry. The `7 * nnodes` expression appears only in the type decision. Since the wall is a **pore count** rather than an edge length, porosity decides which image reaches it: roughly 1150³ at ε = 0.2, 915³ at 0.4, 800³ at 0.6, 690³ at 0.95. That is why the GPU benchmark sweeps — run at ε ≈ 0.95 — hit it between 600³ and 800³.

**64-bit CUSPARSE works on this stack.** CUDA.jl 5.11.3 maps `Int64` to `CUSPARSE_INDEX_64I`, and `cusparseCreateCsr` takes the offsets type and the index type as separate arguments. Verified on the RTX PRO 5000: a `CuSparseMatrixCSR{Float32,Int64}` SpMV agrees with the `Int32` one to 9.5e-7.

**Memory at the wall** (`nnodes` = 306,783,378 ≈ 850³ at ε = 0.5, `nnz` ≈ 1.98e9, `Float32`). Assembly peak is the matrix plus the grid-sized `idx`; solve peak is the matrix plus the Krylov vectors plus the preconditioner's aggregate map.

| path | matrix | assembly peak | solve peak |
| --- | --- | --- | --- |
| assembled, `Int32` (if the bound allowed it) | 15.9 GB | 18.2 GB | 22.2 GB |
| assembled, `Int64` throughout | 24.4 GB | 29.0 GB | 30.7 GB |
| assembled, `Int64` offsets + `Int32` ordinals — **shipped** | 24.4 GB | **26.7 GB** | 30.7 GB |
| matrix-free | — | — | 8.6 GB |

## Design

### The type decision splits in two

`_assembled_index_type(nnodes)` loses its `on_gpu` argument along with the refusal branch; both backends widen identically past `7 * nnodes + 1 > typemax(Int32)`.

`_ordinal_index_type(nnodes)` is new. A pore ordinal is bounded by `nnodes` where an offset is bounded by `7 * nnodes`, so the two walls sit a factor of seven apart and `idx` — which spans the whole grid rather than the pore space — has no reason to widen when the offsets do. The assembly kernels already read their types from the arrays they are handed (`Ti = eltype(counts)`, `eltype(rowval)`), so an `Int32` ordinal written into an `Int64` `rowval` slot is a widening convert on `setindex!` and needs no kernel change.

Deliberately **not** shared with `_operator_index_type`, which applies the same bound but throws on GPU. Assembly has no reason to throw: past 2³¹ pore voxels everything simply widens further.

### The `Ti` keyword

`build_steady_system(img; …, Ti=nothing)` and `SteadyDiffusionProblem(img; …, Ti=nothing)`.

Naming follows the surrounding code and the ecosystem: the signature already takes the value type as `T`; SparseArrays uses `Tv`/`Ti` as the domain notation (`spzeros(Tv, Ti, m, n)`); type-valued keyword arguments are established SciML practice (FiniteDiff's `returntype = eltype(x)`); and LinearSolve's SuperLUDIST extension carries a private `_superlu_index_type(A)` helper of exactly the shape used here.

Semantics, in `_resolve_index_type`: `nothing` takes the automatic choice; `Int64` is always honoured, since asking for more range than the image needs costs memory and nothing else; `Int32` is honoured only while the bound holds, because granting it past the bound would reinstate the wrap-around corruption the bound exists to prevent — not something a keyword gets to switch off. Anything else refuses. Paired with `matrixfree=true` it refuses, rather than being silently ignored.

Without this keyword the wide path could not be exercised below 307M pore voxels and would ship untested, which was the issue's own central objection. The ordinal type stays automatic regardless, so forcing `Ti=Int64` at 24³ exercises *exactly* the production wide configuration: `Int32` ordinals, `Int64` offsets.

### The CUSPARSE prerequisite

`_as_cusparse`'s fast path widens from `Int32` to `Int32`/`Int64`, so a wide matrix is wrapped as it stands. The converting fallback remains for index types CUSPARSE does not take at all, and gains a host-side range check: narrowing an index that does not fit raises `InexactError` inside a device broadcast kernel, where nothing can catch it and the message names neither the matrix nor the reason.

## Work items

| id | item | status |
| --- | --- | --- |
| I1 | `_assembled_index_type` drops the device refusal; both backends widen | **done** |
| I2 | `_ordinal_index_type` — decouple grid ordinals from CSC offsets | **done** — 2.3 GB of assembly peak at 850³ |
| I3 | `_as_cusparse` takes `Int64` natively; range guard on the converting fallback | **done** — the prerequisite; see Final report |
| I4 | `Ti` keyword on `build_steady_system` and `SteadyDiffusionProblem` | **done** — `_resolve_index_type`, refused alongside `matrixfree=true` |
| I5 | Host and device parity tests under forced `Ti=Int64` | **done** — 130 assertions |
| I6 | Regression test pinning the narrowing defect | **done** — 9 assertions, `test_gpu_parity.jl` |
| I7 | Documentation re-truing — the "~850³ ceiling" claim | **done** — `simulations.jl`, `docs/src/api.md` |
| I8 | Measure the wide path's SpMV cost so docstrings quote a number | **done** — +16.9 % at 400³ |
| I9 | Widen `_operator_index_type` for symmetry | **rejected** — see below |

### Rejected, with reasoning

| item | verdict | evidence |
| --- | --- | --- |
| **Close #86 as `wontfix`** | **rejected by Amin's ruling 1** | The measurement supporting it is real and stands: matrix-free is 6.31 s / 7.92 GB against 8.33 s / 19.09 GB at 600³, at identical 106-iteration counts. It is now the documented recommendation. It is not a reason for the default path to refuse. |
| **Exact `nnz` instead of the `7 * nnodes` bound** — scan wide, then narrow `colptr` if it fits | **rejected — measured, buys under 10 %** | Real `nnz / nnodes` is 6.25–6.82, so the worst-case bound is within 3–11 % of the truth. Would cost a wide transient scan in the common case to move the wall by a tenth. |
| **Split index types inside `PortableSparseCSC`** — `Int64` `colptr` + `Int32` `rowval` | **rejected — cannot reach CUSPARSE** | Worth ~35 % more headroom, since `rowval` is the long array and holds ordinals. But CUDA.jl's `CuSparseMatrixCSR{Tv,Ti}` types both arrays with one `Ti`, so the mixed case would lose the CUSPARSE fast path entirely and fall back to the KA gather kernel — through a struct change touching every CSC consumer. (cuSPARSE's C API *does* take the two types separately; the Julia wrapper does not expose it.) |
| **I9 — widen `_operator_index_type` too** | **rejected — out of scope, different failure** | Its refusal guards a wrapping `cumsum!`, not a wrapping offset, and its wall is `nnodes` — near 1630³, five times further out. Its own issue if ever. |
| **Widen the preconditioner's internal `Int32` pore numbering** | **rejected — out of scope** | `two_level_preconditioner` builds its own `Int32` `idx`, which shares the *operator* wall (`nnodes` > 2³¹), far past anything this change reaches. |

## Verification protocol

The image that forces `Ti=Int64` on its own needs ~27 GB of device matrix before the solve allocates, so it cannot run on a 24 GiB card. Every line of the wide path is instead exercised at 24³–32³ by forcing the type, which is what I4 exists for. What that covers: the assembly kernels under `Int64`, entry-for-entry parity against the narrow build, the CUSPARSE wrapper staying 64-bit, the two-level preconditioner, and the end-to-end solve. What it does not cover: arithmetic *at* 2³¹ offsets, which is a pure range question that the type-level tests answer as well as anything can. The `build_steady_system` docstring says so rather than leaving a reader to discover it.

## Progress log

Format: `date — id(s) — status — finding — verification`.

2026-08-10 — investigation — **the `wontfix` alternative was weighed first, as the issue asked.** Measured `nnz/nnodes` (6.25–6.82), confirmed allocation is already exact, verified CUSPARSE `Int64` works on this stack, and computed the memory table. Posted to #86 as [issue comment 5242894791](https://github.com/ma-sadeghi/Tortuosity.jl/issues/86#issuecomment-5242894791) with the recommendation to close. Amin overruled on the grounds in Governing constraints — recorded here because the measurements stand even though the recommendation did not.

2026-08-10 — **I3 finding, and the reason this could not be a one-line change.** `_as_cusparse`'s non-`Int32` fallback converted `colptr` and `rowval` **down** to `Int32` before wrapping. Confirmed by construction: an `Int64`-indexed `PortableSparseCSC` on the GPU was handed a `CuSparseMatrixCSR{Float32,Int32}`. Below the wall this only wasted memory — it cached a second copy of both index arrays — but past the wall it is precisely the silent corruption the index bound exists to prevent, and it would have fired on *every* matrix the widened assembly path produces. **Deleting the `throw` without fixing this would have shipped a broken path.** Fixed by widening the native fast path to `Int32`/`Int64`; pinned by I6, which asserts `wrapper.rowPtr === wide.colptr` so a future reintroduction of the copy fails loudly.

2026-08-10 — I1, I2, I4 — done — type selection verified across both walls: `_assembled_index_type` returns `Int32` at the bound and `Int` past it; `_ordinal_index_type` still returns `Int32` one voxel past the *offset* wall, which is the decoupling working. Host parity bitwise: `colptr`, `rowval`, `nzval` and `b` all exactly equal between the narrow and wide builds across five fixtures × three axes, uniform and variable `D`.

2026-08-10 — I5 — **GPU τ delta is the preconditioner, not the index width.** `Ti=Int64` against the narrow build on GPU differs by 1.9e-5 in τ. Four *identical* narrow runs spread **2.15e-5** — larger than the delta — so the wide path sits inside the preconditioner's own atomics scatter, consistent with the known non-determinism. CPU is exactly 0.0.

2026-08-10 — I8 — done — **the wide path's price, measured rather than asserted.** At 400³ on the RTX PRO 5000 (31,906,673 pore voxels, 216,918,266 nonzeros, `Float32`, min of 20 reps after a discarded warm-up): one SpMV takes **3.45 ms with `Int32` against 4.03 ms with `Int64`, +16.9 %**, for **0.93 GiB against 1.85 GiB** of index storage. Quoted in the `_assembled_index_type` docstring in place of the adjective that stood there.

2026-08-10 — I7 — done — `simulations.jl`'s `matrixfree` docstring claimed the keyword "is what makes images past the assembled path's ~850³ ceiling solvable". The ceiling is gone; it now quotes the 600³ comparison and states that nothing switches paths on the caller's behalf. `docs/src/api.md` keeps its memory claim — which is true — and adds that it is a memory limit rather than a refusal.

2026-08-10 — **Suite gate: PASSED.** Foreground fresh-process `Pkg.test()`, GPU included, exit 0. Test-file runs during development: `test_assembly.jl` 9550/9550, `test_gpu_parity.jl` 388/388, `test_gpu_e2e.jl` 84/84.

## Final report

### What shipped

Implementation in `ab63e7f`.

| file | change |
| --- | --- |
| `src/assembly.jl` | `_assembled_index_type(nnodes)` without the refusal; `_ordinal_index_type`; `_resolve_index_type`; `Ti` keyword; `idx` typed by the ordinal bound |
| `ext/TortuosityCUDAExt.jl` | fast path covers `Int32` and `Int64`; range guard on the converting fallback |
| `src/simulations.jl` | `Ti` forwarded and refused alongside `matrixfree=true`; `matrixfree` docstring re-trued |
| `docs/src/api.md` | memory limit distinguished from refusal; no auto-routing stated |
| `test/test_assembly.jl` | 108 assertions — type selection, keyword validation, narrow/wide parity |
| `test/test_gpu_parity.jl` | 9 assertions — the narrowing regression |
| `test/test_gpu_e2e.jl` | 22 assertions — device parity, τ, preconditioner on a wide matrix |

**139 new assertions.**

### What this buys, honestly

On the 24 GiB card, **nothing new runs**. 800³ at ε ≥ 0.6 needs ≥ 27 GB of assembled matrix even as `Int32`; what changes is that the failure mode becomes an honest CUDA OOM rather than an `ArgumentError`. The sizes this genuinely unlocks need an 80 GB-class card, where the wide path reaches roughly 1170³ at ε = 0.5 before memory binds again.

On CPU it changes less than expected, because the host branch already widened — see the note in the status box. The `idx` narrowing does trim 4 B per grid voxel (2 GB at 800³) from the assembly peak there too.

The defect fixed as a prerequisite (I3) is worth more than the feature at present sizes: it removes a duplicated pair of index arrays from the cache for any non-`Int32` device matrix, and closes a silent-corruption path that the widened assembly would otherwise have walked into on its first run.

### Carried forward

- **No run past 306,783,378 pore voxels has ever happened.** Stated in the docstring. It needs hardware not present here.
- **Benchmark coverage is the follow-up, not code.** CPU sweeps stop at 400³ and GPU assembled at 600³, both for time rather than capability. A sweep that reaches 800³ CPU and low-ε 800³ GPU assembled would put real numbers against the regime this change opens. That belongs to a benchmark session.
- **`_operator_index_type` still refuses on GPU** past `nnodes` = 2³¹ (~1630³). Deliberate; see the rejection table.
