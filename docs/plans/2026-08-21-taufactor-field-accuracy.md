---
title: Does taufactor's stopping rule cost concentration-field accuracy?
created: 2026-08-21
updated: 2026-08-21
status: complete
outcome: The mechanism is real — the field lags tau by a median 6x at declared convergence — but the claim's stated condition is inverted, so the claim was dropped rather than published.
branch: joss
supersedes: -
superseded-by: -
related: -
---

> **Status: complete.** An unverified assertion in a since-deleted internal doc held that taufactor's flux-matching convergence yields a less accurate concentration field even when tau is right, and that this is worst in strongly heterogeneous media near the REV. Measured on 17 cases against taufactor's own SOR fixed point: the first half is true and the second half is backwards — the field/tau error gap is *smallest* in the heterogeneous low-porosity images the doc named and largest in near-homogeneous ones. **Decision, 2026-08-21: the claim is dropped.** It is not in the paper and no issue was filed. This document exists so that nobody re-investigates it.

**Verdict: TRUE-BUT-CONDITIONAL — and the stated condition is backwards.** The field error at declared convergence really is larger than the tau error (median ~6x at taufactor's default `conv_crit`, up to ~100x), so "even when tau is correct the field may not be as precise" is defensible. But the doc pins that on *strongly heterogeneous media near the REV*, and that half is measurably false: the gap is at its **smallest** in exactly those cases (2.3–3.6x at porosity 0.17–0.18, tau 11–17) and at its **largest** in near-homogeneous high-porosity media (28–99x at porosity 0.80). The absolute field error at the stopping point is essentially constant across porosity, structure and image size, because the criterion adapts — the heterogeneous cases simply run 10x more iterations before it fires.

## What taufactor's convergence criterion actually is

Two tests, evaluated together every 100 iterations, in `check_convergence`.

**Upstream** (`tldr-group/taufactor@793717c`, `taufactor/taufactor.py`):

```python
143:  if not np.all(relative_error < conv_crit):
144:      self.old_tau = self.tau
145:      return False
146:
147:  tau_error = np.max(np.abs(self.tau - self.old_tau))
148:  if not tau_error < 2e-3:
```

with `relative_error` defined at `taufactor.py:297-301`:

```python
297:  fl_max = np.max(self.flux_1d, axis=1)
298:  fl_min = np.min(self.flux_1d, axis=1)
300:  relative_error = np.divide(fl_max - fl_min, fl_max, ...)
```

`flux_1d` is the yz-averaged x-flux per x-slice. So test one is a **flux-uniformity** test: the spread between the most and least conducting cross-section, relative to the largest, over *all* slices — not just inlet against outlet, as the doc says. Test two is a **tau-stability** test: the change in tau over the last 100 iterations, absolute, against a **hard-coded 2e-3** that silently overrides whatever the user passed as `conv_crit`. Neither is a residual. The premise of the claim is therefore right in substance: the stopping rule is a pair of global functionals of the field, not a bound on the field.

**The fork** (`benchmarks/vendor/taufactor/taufactor/taufactor.py`, `ma-sadeghi/taufactor@a4bc5f9`, branch `node-centered-bc`) changes exactly one character of this — line 148 becomes `if not tau_error < conv_crit:` — so the user's tolerance governs both tests. The structure of the criterion, and hence everything the claim is about, is identical in fork and upstream. The claim applies to both.

Note the consequence for upstream: because the tau gate is *absolute*, it is relatively strict where tau is large (2e-3 on tau=17 is 1.2e-4 relative) and relatively loose where tau ~ 1. In practice the flux test is the binding one at `conv_crit=1e-2` in 15 of the 17 cases measured.

## The experiment

The confounder to kill first is discretisation: comparing taufactor's field against Tortuosity.jl's would mix "stopped too early" with "solves a different discrete problem". So the reference is **taufactor's own SOR fixed point**. The SOR update is a linear fixed-point iteration, so its limit *is* the exact solution of the discrete system it defines, and the increment before relaxation is exactly the residual, giving an independent check on how converged the reference is.

- Images: the benchmark store's cached 100³ smoke grid, all 15 (3 blobiness x 5 porosity, already trimmed to the percolating pore space), plus two 200³ cases for a size axis.
- Solver: taufactor's own `Solver`, subclassed only to force `precision=torch.float64` (`Solver.__init__` does not pass `precision` through), on the GPU. No solver logic touched.
- Reference: run in blocks until `max|residual|` over interior pore voxels < 1e-13. Reached 6e-15 … 3.5e-14 in every case, at 4 000 – 30 000 iterations.
- Replay: a fresh solver stepped through the *real* `solve()` loop with `conv_crit=0` so it never exits, stopping every 100 iterations — i.e. at exactly the points where `check_convergence` fires — recording the flux spread, the tau it would report, the L2 and L-inf field error against the reference over pore voxels, and the residual. taufactor's two-part rule is then replayed over that trace to find the check at which it would have declared convergence.

**Validity checks.** Tortuosity.jl at `reltol=1e-12` on the same two images gives tau = 17.025815788 against taufactor's fixed point 17.025815790 (rel. diff **1.2e-8**) on `n100_b200_p020`, and 1.2290965070 vs 1.2290965069 on `n100_b050_p080`. The two codes solve the same discrete problem, so the reference is the right field and there is no discretisation confound hiding in the result. Separately, taufactor's default float32 path floors out at a field L2 error of ~3e-6 (`n100_b200_p020`, 10 000+ iterations) — two to three orders below the stopping-rule error, so precision is not what limits the field at any realistic tolerance.

## Numbers, at taufactor's default `conv_crit = 1e-2`

Errors are relative to the fixed point; `field_L2` is ‖dc‖2/‖c‖2 over pore voxels, `field_Linf` is max|dc| absolute (the field spans 0…1).

| case | porosity | tau_ref | iters at exit | tau err | field_L2 | field_Linf | L2/tau |
|---|---|---|---|---|---|---|---|
| n200_b100_p020 | 0.171 | 13.933 | 5200 | 8.8e-04 | 2.0e-03 | 6.6e-03 | 2.3 |
| n100_b050_p020 | 0.173 | 13.285 | 3800 | 9.4e-04 | 2.3e-03 | 3.4e-03 | 2.4 |
| n100_b200_p020 | 0.178 | 17.026 | 3100 | 7.6e-04 | 1.9e-03 | 3.6e-03 | 2.5 |
| n100_b100_p020 | 0.182 | 10.980 | 3600 | 1.2e-03 | 4.5e-03 | 1.5e-02 | 3.6 |
| n100_b200_p040 | 0.398 | 3.278 | 400 | 5.6e-04 | 1.1e-03 | 2.8e-03 | 1.9 |
| n100_b100_p040 | 0.398 | 2.776 | 600 | 1.5e-04 | 2.2e-04 | 1.0e-03 | 1.5 |
| n100_b050_p040 | 0.405 | 2.617 | 700 | 4.1e-04 | 1.9e-03 | 1.4e-02 | 4.7 |
| n100_b200_p060 | 0.601 | 1.795 | 300 | 1.4e-04 | 9.2e-04 | 4.8e-03 | 6.7 |
| n100_b100_p060 | 0.601 | 1.644 | 400 | 3.5e-05 | 8.0e-04 | 3.6e-03 | 23.2 |
| n100_b050_p060 | 0.601 | 1.637 | 400 | 5.4e-05 | 1.6e-03 | 5.6e-03 | 30.1 |
| n100_b050_p080 | 0.800 | 1.229 | 400 | 2.7e-04 | 6.7e-04 | 2.9e-03 | 2.5 |
| n100_b200_p080 | 0.801 | 1.277 | 300 | 5.0e-06 | 4.9e-04 | 2.3e-03 | 98.7 |
| n100_b100_p080 | 0.802 | 1.226 | 300 | 1.3e-05 | 9.7e-04 | 2.7e-03 | 75.1 |
| n200_b100_p080 | 0.804 | 1.194 | 500 | 4.8e-05 | 1.4e-03 | 6.9e-03 | 28.4 |
| n100_b100_p095 | 0.950 | 1.046 | 200 | 3.6e-05 | 8.7e-04 | 3.0e-03 | 24.1 |
| n100_b200_p095 | 0.951 | 1.057 | 200 | 3.1e-05 | 4.8e-04 | 1.7e-03 | 15.6 |
| n100_b050_p095 | 0.952 | 1.044 | 300 | 4.3e-05 | 8.5e-04 | 1.9e-03 | 19.5 |

Repeating the whole sweep at `conv_crit` = 1e-3 and 1e-4 (fork, both gates honoured) reproduces the same picture one and two decades down, so this is a property of the criterion and not of one tolerance:

| setting | field_L2 / conv_crit (median, range) | field_Linf / conv_crit (median, range) | L2 err / tau err (median, range) |
|---|---|---|---|
| upstream, conv_crit=1e-2 | 0.09 (0.02 – 0.45) | 0.30 (0.10 – 1.47) | 6.7 (1.5 – 98.7) |
| fork, conv_crit=1e-2 | 0.09 (0.05 – 0.45) | 0.30 (0.17 – 2.50) | 15.6 (1.9 – 98.7) |
| fork, conv_crit=1e-3 | 0.10 (0.06 – 0.58) | 0.32 (0.14 – 4.25) | 5.1 (2.0 – 65.2) |
| fork, conv_crit=1e-4 | 0.09 (0.04 – 0.53) | 0.34 (0.14 – 3.84) | 5.3 (1.9 – 75.0) |

And in the currency that matters to a user — how much more work a field-accurate stop would cost — the field reaches a given relative accuracy after a median **1.5x** the iterations tau needs, and in the heterogeneous low-porosity cases only **1.15–1.47x**.

## Is the gap material, and when?

Directionally the mechanism is confirmed: tau is a doubly-averaged functional of the field and converges ahead of it, by a median factor of ~6 in L2 at the default tolerance. The doc's sentence "even when the computed tortuosity factor is correct, the concentration field itself may not always be as precise" is a true statement about taufactor.

Three findings work against using it as written:

1. **The conditionality is inverted.** Ranked by porosity, L2/tau goes 2.3, 2.4, 2.5, 3.6 (porosity ~0.18) … 15.6, 19.5, 24.1 (porosity ~0.95). The heterogeneous, low-porosity, high-tau images — the ones the doc singles out — have the *tightest* coupling between tau error and field error, because there the slowest-decaying error mode is global and pollutes tau and the field alike. The large ratios come from near-homogeneous media where tau happens to be almost exactly right at 200–400 iterations while the field still carries 1e-3.
2. **The absolute field error at the stopping point barely moves.** Across a 5.5x range of porosity, three structures, 16x in tau and 8x in voxel count, `field_L2` at declared convergence stays inside 2.2e-4 … 4.5e-3, and tracks `conv_crit` at roughly 0.1x. The criterion adapts: `n100_b200_p020` runs 3100 iterations to satisfy it and `n100_b200_p095` runs 200. It is doing the job the doc implies it fails at.
3. **No image-size effect.** 200³ reproduces 100³ at both ends (2.3 vs 3.6 at porosity 0.17; 28 vs 75 at porosity 0.80), with the same absolute field error. The "image size comparable to the REV" framing has no support in the data.

The one genuinely defensible sharp edge is L-inf: the worst-voxel error can exceed `conv_crit` itself — up to 1.5x it upstream at 1e-2, and up to 4.2x on the fork at 1e-3. So `conv_crit` is not an upper bound on the pointwise field error, only a proxy good to within a small factor. A residual-based tolerance bounds the field directly. That is a real structural difference between the two stopping rules; it is not, on this evidence, an accuracy advantage in heterogeneous media.

## Recommendation

Discard the claim as written; it should not become a sentence in the paper and does not warrant a GitHub issue against taufactor. The premise is right about the mechanism but the doc's headline condition — worse in strongly heterogeneous media, near the REV, on small images — is the one thing the measurement contradicts, in all three of its variables at once. Publishing it would put an assertion in the JOSS paper that our own data refutes, on the axis a reviewer running taufactor is most likely to check. The only measured statement that would have been defensible is the narrow one: taufactor stops on global flux functionals rather than a residual, so at its default tolerance the pointwise concentration error is a few times larger than the tortuosity error and can exceed `conv_crit`, whereas a residual-based stop bounds the field — with the honest rider that closing that gap costs about 1.2–1.5x more SOR iterations and that the effect is weakest, not strongest, in the heterogeneous media the old doc named. That is a design-difference footnote, not a differentiator, and my recommendation is to say nothing rather than to say it thinly.

## What was measured, and what survives

The tables above are the result in full; every row quoted in them came from the traces described under "The experiment". The scripts and raw traces lived in a session scratch directory and were not kept, so redoing this means rebuilding the harness: subclass taufactor's `Solver` to force `precision=torch.float64`, run it to a residual floor to obtain the fixed point, then step a fresh solver through the real `solve()` loop with `conv_crit=0` and sample every 100 iterations, replaying the two-part rule over the trace. Nothing in the repository was modified by the investigation.
