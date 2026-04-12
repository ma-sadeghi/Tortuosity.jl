# Tortuosity.jl — Design Decisions

Internal documentation for maintainers. Captures non-obvious design choices and
the rationale behind them, with quantitative data where it influenced the call.
This is **not** user-facing documentation — for that, see `docs/src/`.

Each entry follows the same shape: **Decision**, **Rationale**, **Data** (when
relevant), and **Alternatives considered**.

---

## 1. KernelAbstractions.jl over CUDA.jl as the GPU backend

**Decision.** All GPU kernels in `src/kernels/` are written using
`@kernel` from `KernelAbstractions.jl` and atomics from `Atomix.jl`, not
`@cuda` from `CUDA.jl`. CUDA.jl is a *weak dependency* loaded only via
`ext/TortuosityCUDAExt.jl`.

**Rationale.** We want the same source code to run on NVIDIA, Apple Silicon
(Metal), and AMD (ROCm) GPUs. CUDA.jl is NVIDIA-only. KernelAbstractions
provides a single kernel-writing interface that compiles to whichever GPU
backend the user has loaded — `using CUDA`, `using Metal`, or `using AMDGPU`
— via the corresponding extension. The core package never imports any
GPU-specific package.

**Data.** Beyond portability, the refactor turned out to be **3.2-3.9× faster
than the previous CUDA-specific code at all problem sizes** (see
`bench/baselines/`). Most of the win came from replacing CUSPARSE-mediated
sparse matrix construction with direct KA kernels — see decision #3.

**Alternatives considered.**

- *Multiple parallel implementations (one per GPU)*: rejected. Triples the
  surface area to maintain. Test matrix explodes.
- *Stick with CUDA.jl, give up portability*: rejected. The package's main
  use case is large 3D imaging volumes; users on Apple Silicon and AMD
  hardware are excluded for no good technical reason.
- *Rewrite to use GPUArrays.jl abstract types*: investigated and rejected.
  GPUArrays.jl's abstract sparse types aren't subtyped by `CuSparseMatrixCSC`
  in CUDA.jl, and Metal.jl has no sparse support at all. KA is the right
  abstraction layer.

---

## 2. Custom `PortableSparseCSC` type instead of `SparseMatrixCSC` or `CuSparseMatrixCSC`

**Decision.** Introduced `PortableSparseCSC{T,Ti,V,Vi}` in `src/sparse_type.jl`,
parameterised on the storage vector type `V`. Same struct works on CPU
(`V = Vector{T}`), CUDA (`V = CuVector{T}`), Metal (`V = MtlVector{T}`), and
AMDGPU (`V = ROCVector{T}`).

**Rationale.** We need a CSC sparse matrix type that:

1. Works with any GPU backend's storage vector
2. Is `<: AbstractMatrix{T}` so `LinearSolve.jl` and `Krylov.jl` accept it
3. Implements `mul!(y, A, x)` (the only operation Krylov needs)
4. Supports the in-place ops we use: `set_diag!`, `zero_rows_cols!`,
   `dropzeros!`, etc.

`SparseMatrixCSC` is CPU-only. `CuSparseMatrixCSC` is CUDA-only. There is
no existing sparse type that works across GPU backends. So we made one.

**Why not use Julia's `SparseMatrixCSC` for everything and live with CPU
performance on Metal/AMDGPU?** Because the package's whole point is GPU
acceleration on large 3D images. A CPU fallback for non-CUDA GPUs would be
30-100× slower than an actual GPU kernel.

**Data.** The custom `PortableSparseCSC` is verified mathematically equivalent
to `CuSparseMatrixCSC` on 374 GPU parity tests across 43 fuzz images
(`test/test_gpu_parity.jl`). Performance comparison vs CUSPARSE shows we're
**8-10× faster than CUSPARSE for matrix construction** (`laplacian`,
`adjacency_matrix`, `dropzeros!`, `zero_rows_cols!`) and **at parity for
SpMV** (via the fast-path in decision #3).

**Alternatives considered.**

- *Subtype `AbstractGPUSparseMatrixCSC` from GPUArrays*: doesn't exist /
  isn't subtyped by CUDA.jl. Dead end.
- *Wrap `SparseMatrixCSC` and dispatch internally*: same problem — no GPU
  storage.

---

## 3. CUSPARSE fast-path for SpMV in `TortuosityCUDAExt`

**Decision.** When `mul!(y, A, x)` is called on a `PortableSparseCSC` whose
storage is a `CuVector`, the CUDA extension wraps the matrix as a
`CuSparseMatrixCSC` and dispatches to CUSPARSE's hand-tuned SpMV. For other
backends (Metal, AMD, CPU), the fallback KA SpMV kernel is used.

**Rationale.** We tried writing our own SpMV kernel using `Atomix.@atomic`.
On GPU it was about **0.47× the speed of CUSPARSE** for the symmetric Laplacian
matrices we use. CUSPARSE's hand-tuned implementation uses warp-level reduction
tricks and tensor-core paths we don't replicate. Since CUDA is the only backend
with CUSPARSE, the cleanest fix was a vendor-specific fast-path *inside the
extension*, leaving the portable KA fallback for everything else.

**Data.** With the fast-path, SpMV is at **0.99-1.02× of the old CUDA-specific
code** at all sizes. Without it, we'd be at 0.47-0.56×.

**Wrapper reuse.** `PortableSparseCSC` carries an opaque `_cache::Ref{Any}`
slot that the CUDA extension uses to memoise the `CuSparseMatrixCSC`
wrapper. On each `mul!`, we validate the cache by comparing the
wrapper's `nzVal`/`rowVal`/`colPtr` pointers against `A`'s fields — a
mutator like `dropzeros!` that reassigns `A.nzval` automatically
invalidates the cache. Over a Krylov CG solve this eliminates the
per-iteration wrapper allocation entirely.

**Alternatives considered.**

- *Pure KA SpMV with column-based atomic accumulation*: rejected due to
  performance (see above).
- *Use a row-based SpMV exploiting the symmetry of our Laplacian*: not pursued
  because the CUSPARSE fast-path solved the problem with less code.

---

## 4. Don't sort CSC `rowval` after the atomic scatter

**Decision.** `create_adjacency_matrix` builds CSC via histogram + scan +
atomic scatter. The atomics produce non-deterministic row ordering within
each column. We do **not** sort the rowvals after the scatter.

**Rationale.** It would be more code, more memory traffic, more kernel
launches — and we measured that it doesn't help.

**Data.** Direct measurement on RTX 3060
(`bench/sort_csc_experiment.jl`, since deleted but reproducible from this
documentation):

| Size       | SpMV unsorted | SpMV sorted | Speedup    | Sort cost | Break-even calls |
|------------|--------------:|------------:|-----------:|----------:|-----------------:|
| 100K (47³) |       48.7 µs |     49.1 µs | **0.991×** *(slower)* |  17.72 ms | n/a              |
| 250K (63³) |       88.3 µs |     89.1 µs | **0.991×** *(slower)* |  52.73 ms | n/a              |
| 1M  (100³) |      278.1 µs |    275.7 µs | **1.009×** *(2.4 µs win)* | 297.44 ms | **124,921**      |

At 100K and 250K, sorted is *marginally slower* than unsorted (within noise).
At 1M, sorted is marginally faster, but the per-call savings are 2.4 µs while
the host-side sort costs 297 ms. Break-even is **~125,000 SpMV calls** — a
typical Krylov CG solve runs 50-500 iterations, so we'd never recover the
sort cost from a single solve. Even with a fast GPU radix sort instead of
CPU `sortperm`, the per-call savings are too small to amortise within a
realistic workload.

**Why is the difference so small?** CUSPARSE's modern `cusparseSpMV` uses a
row-block algorithm internally that doesn't depend on row order — it iterates
`colptr[j]:(colptr[j+1]-1)` regardless of how `rowval[k]` are ordered within
that range.

**Alternatives considered.**

- *Sort lazily on first SpMV*: same total cost, just deferred.
- *Sort during scatter using a different algorithm*: would require segmented
  sort kernel — significant new code for unmeasured benefit.

---

## 5. Float32 on GPU, Float64 on CPU

**Decision.** `TortuositySimulation` and `TransientProblem` use
`Float32` for all arrays when `gpu=true` and `Float64` when `gpu=false`.

**Rationale.** Inherited from the pre-refactor code intentionally:

- Float32 SpMV is 2× faster than Float64 on most consumer GPUs (memory
  bandwidth limited)
- Float32 atomics are universally supported; Float64 atomics require
  compute capability 6.0+ on NVIDIA and aren't supported at all on Apple
  Metal
- CG solves of the diffusion Laplacian converge to user-meaningful tolerance
  with Float32 — `reltol=1e-6` is achievable

**Trade-off.** On CPU, Float64 is the natural choice because there's no
bandwidth penalty and double-precision atomics aren't an issue. So we have
a backend-dependent default. **This is a real footgun**: if you compute
something on GPU and compare against CPU output, expect ~1e-6 differences
that aren't bugs.

**Alternatives considered.**

- *Float64 everywhere*: rejected due to GPU performance hit.
- *Float32 everywhere*: rejected because CPU users have no reason to give
  up precision and double-precision is the Julia ecosystem default.
- *User-controlled `T` parameter*: future work; currently the simulation
  constructors don't expose this knob.

---

## 6. `bench/old_baseline.jl` is intentionally retained as a frozen reference

**Decision.** `bench/old_baseline.jl` contains a complete copy of the
pre-refactor CUDA-specific code as `OldBaseline.create_connectivity_list_old`,
`OldBaseline.laplacian_old`, etc. It is loaded by `bench/gpu_bench.jl` and
`test/test_gpu_parity.jl` but never by the user-facing package.

**Rationale.** Two purposes:

1. **Bench oracle** — gives `bench/gpu_bench.jl` something to compare *against*
   so we can detect performance regressions vs the original implementation,
   not just vs an arbitrary saved baseline.
2. **Test oracle** — gives `test/test_gpu_parity.jl` a ground truth for 374
   fuzz tests that verify mathematical equivalence between old and new
   implementations.

**Why not delete it?** Without it:

- The `bench/gpu_bench.jl` "old vs new" columns become meaningless
- The 374 GPU parity tests can't run, leaving GPU correctness uncovered

**When to delete.** After this refactor branch has been merged to `main` and
we have a few weeks of confidence in the new code, we can delete
`bench/old_baseline.jl` and `test/test_gpu_parity.jl` together as a single
cleanup commit. By then, saved baselines from the post-merge state replace
the "vs old code" comparison, and the parity tests are no longer needed.

**Cost.** ~600 lines of frozen Julia. Compiles in ~12 seconds at first load,
cached afterwards. Zero maintenance burden — the file is never edited.

---

## 7. `bench/Project.toml` as a separate environment

**Decision.** Bench tooling (`CUDA`, `JSON`, `BenchmarkTools`, `LinearSolve`,
`OrdinaryDiffEq`) lives in `bench/Project.toml`, not the main `Project.toml`.
The bench is invoked as `julia --project=bench bench/gpu_bench.jl`.

**Rationale.** The main package's `[deps]` should contain only what users
actually need at runtime. Adding `BenchmarkTools` and `JSON` to runtime
dependencies for the sake of an internal benchmark would force every user to
download them. The separate environment isolates dev tooling from runtime
deps cleanly.

**Why not put the bench tools in `[extras]`?** `[extras]` is for test deps
(it's how `Pkg.test()` finds things). The bench is not a test — it's a
performance harness with its own lifecycle. Keeping it separate means the
test environment doesn't pull `BenchmarkTools` either.

**Convention.** This is the standard Julia pattern for benchmark suites
(e.g., `JuliaLang/Julia` and most SciML packages do the same).

---

## 8. `gpu=true` guards against silent CPU fallback

**Decision.** `TortuositySimulation(...; gpu=true)` and
`TransientProblem(...; gpu=true)` explicitly `error(...)` if no GPU backend
extension has been loaded.

**Rationale.** Without this check, `gpu=true` silently falls back to CPU
because `_gpu_adapt[]` defaults to `identity`. The user sees no warning, the
construction succeeds, and they get a CPU computation when they explicitly
asked for GPU. That's a footgun.

**Cost.** ~5 lines per constructor. Trivial.

---

## Open / known issues (not yet decisions)

These are tracked here to avoid losing them. Not actionable now; revisit
when relevant.

- **`dropzeros!` algorithm is counter-intuitive but benchmark-validated.**
  The compact-and-count kernel launches over nnz and calls
  `searchsortedlast(colPtr, k)` per entry to find the owning column.
  "Obvious" refactor: launch over `num_cols` and walk each column's nnz
  range inline — no binary search, no atomics, asymptotically better. We
  implemented and benchmarked it: on realistic porous Laplacians at
  100K/250K/1M voxels the per-column variant was **9–24% slower** because
  the uniform 7-point stencil gives each column only ≈7 entries, so the
  per-nnz launch has higher GPU occupancy and better coalescing than the
  per-column launch. If this workload ever shifts to matrices with
  long-tail column lengths (e.g. multi-physics couplings), revisit.

- **Matrix construction uses `Int32` indices throughout.** This is mandated
  by the CUSPARSE fast-path (CUSPARSE expects Int32). If someone ever
  constructs a `PortableSparseCSC` manually with Int64 indices the fallback
  `_as_cusparse` will quietly convert, allocating on every `mul!`. A warning
  on that path would be nice but hasn't been prioritised.
