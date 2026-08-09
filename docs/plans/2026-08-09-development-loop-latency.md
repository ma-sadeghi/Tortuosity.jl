---
title: Development loop latency
created: 2026-08-09
updated: 2026-08-09
status: draft
branch: "-"
supersedes: "-"
superseded-by: "-"
related: 2026-08-08-matrix-path-optimization.md
---

> **Status: draft.** Plan for removing the latency that dominates agent-driven development on this package: a coding agent that edits a source file and runs a script pays **~106 s before a single assertion executes**, of which **~92 s is precompilation and ~60 s of that is loading 212 dependencies into the precompile worker** — not compiling our code. The fix is not a package restructure; it is to stop spawning a fresh Julia process per edit. A warm session with Revise takes the same loop to **~2 s (measured, ~50×)** with no source change and no cost to end users. That tooling is delivered globally through the existing `~/.agents` + chezmoi fan-out rather than per-repository, so every agent (Claude, Codex, Copilot) and every Julia project gets it once. Two secondary items ship real improvements to users; two more are deliberately deferred. Every number in this document was measured on 2026-08-08/09 against an isolated copy of the package — none are estimates unless labelled as such.

# Development loop latency plan

This plan exists because the matrix-path optimization campaign (`2026-08-08-matrix-path-optimization.md`) spent most of its wall clock waiting rather than working. That campaign's own log records the symptom: a full `Pkg.test()` at **306.97 s wall against 170.6 s of testsets**, i.e. **136 s (44 %) before the first assertion**. This document addresses the cause.

## Governing constraints

Two constraints set by Amin on 2026-08-09. They are not preferences; they decide what is in and out of scope, and one of them removed an item that earlier analysis had ranked highly.

**1. The end-user experience is sacrosanct.** No change may make the package slower to load or slower on first call for people who install it. A change that makes development pleasant at the cost of user-visible warm-up is disqualified regardless of how much it saves us. Work items are therefore tagged **dev-only** (invisible to users) or **shipped** (users feel it), and every shipped item must be neutral-or-better for users.

**2. The problem is agent processes, not the package.** Local development by a human against a warm REPL was never slow, and still is not. The regression came from agent-driven development, where every check is a new `julia` invocation. The correct target is therefore the process model, not the package structure.

## The problem, measured

Precompiling cumulative prefixes of the include list decomposes the ~92 s per-edit precompile:

| component | cost | share |
| --- | --- | --- |
| fixed overhead before any package source compiles (loading 212 deps into the precompile worker) | ~59.5 s | 65 % |
| all 15 source files combined (`transient.jl` +7.4 s, `imgen.jl` +5.6 s, rest ≲2 s each) | ~16 s | 17 % |
| the CPU `@compile_workload` | ~16 s | 17 % |

The per-process floor on top of that, on a quiet machine:

| scenario | wall |
| --- | --- |
| bare `julia --startup-file=no -e exit()` | 0.72 s |
| bare julia **with** the user's `startup.jl` | 4.16 s |
| warm `using Tortuosity` | 6.1 s |
| warm `using Tortuosity` + `using CUDA` | 11.2 s |

`using Tortuosity` pulls **260 packages** at runtime (212 in the manifest closure). And the suite itself is compilation-bound, not compute-bound: the 15 CPU test files run **100.1 s cold and 29.9 s warm** in one process — **~70 % is first-call JIT**. `test_transient.jl` is the extreme: 20.2 s cold, 1.5 s warm.

**Conclusion.** Roughly 30 s of a ~307 s `Pkg.test()` is actual numerics. Everything else is process startup, dependency loading, precompilation and JIT — all of which a warm process pays once instead of every time.

## Work items

Statuses: `pending`, `done`, `rejected`, `blocked`, `reverted`, `deferred`.

### Phase 1 — the development loop (dev-only, no source changes)

> **Scope note.** Phase 1 changes no file in this repository. It lands in the chezmoi dotfiles repo (`~/.local/share/chezmoi`) and its generated targets under `~/.agents`, `~/.claude`, `~/.codex`. It is recorded here because this package is where the cost was measured and where the payoff is felt, but the work is global and benefits every Julia project. Phase 2 is the only part that touches Tortuosity's own source.

| id | item | kind | status | measured effect |
| --- | --- | --- | --- | --- |
| L1 | Adopt an MCP server giving agents a persistent Julia session | dev-only | pending | edit→green **106 s → ~2 s** |
| L2 | Pass `-O1` on all test runs | dev-only | pending | cold suite **90.7 s → 67.2 s** (−26 %) |
| L3 | Deliver the MCP server config from `~/.agents/` via a chezmoi sync script, to Claude / Codex / Copilot | dev-only | pending | enables L1 on every agent, not just one |
| L4 | Global `julia-workflow` skill in `~/.agents/skills/` carrying the guardrails | dev-only | pending | prevents the silent-stale failure mode |

**L1–L4 are one change delivered in four parts.** L1 picks the server, L3 installs it everywhere, L4 makes agents actually use it correctly. Shipping L1 without L4 is the dangerous combination: a persistent session that agents use naively returns stale results silently (see below).

**L1 — persistent session via MCP.** Three servers exist and none require any change to our source:

| server | tools | session model | Revise | maturity (2026-08-09) |
| --- | --- | --- | --- | --- |
| [julia-mcp](https://github.com/aplavin/julia-mcp) (aplavin) | 3 | per-project, Python server, stdio | automatic | 80★, 36 commits |
| [AgentREPL.jl](https://github.com/samtalki/AgentREPL.jl) | 8 | Malt.jl workers, Julia-native, 6 deps | auto-loads, plus a hook that nudges the agent to reload after `.jl` edits | 7★, 68 commits |
| [Kaimon.jl](https://github.com/kahliburke/Kaimon.jl) | 35–49 | HTTP + ZMQ, 30+ deps | manual | debugger, semantic search |

Recommendation: **start with julia-mcp**, primarily for its TestEnv integration — which matters here because the package uses `[extras]`/`[targets]` rather than a `test/Project.toml`, so `--project=.` cannot `using Test`. Kaimon is over-scoped for this problem.

*Durability was tested, because a daemon that agents hammer for hours is not the same thing as an interactive REPL.* Twelve consecutive edit → revise → test cycles:

- **Latency flat:** revise 0.06–0.08 s, tests 1.65–2.07 s, no drift across cycles.
- **Memory plateaus:** 978 → 1136 MiB by cycle 8, stable thereafter. Growth, not a leak.
- **Survives failures:** a failing testset throws, session stays usable (next run 1.64 s).
- **Struct redefinition worked** on Julia 1.12 — the classic Revise dealbreaker is largely gone. Verified only to the extent that `revise()` reported success and the session stayed healthy; stale-instance semantics were not probed.

**The one hazard, and it is a real one.** A missed `revise()` returns **stale results silently**. This was hit during the investigation: a measurement that looked like a fast edit cycle was a no-op reload, and an agent would have read it as a passing test on code it never compiled. **Acceptance criterion for L1: the workflow must verify that the revision applied rather than assume it** — e.g. a probe value bumped by the edit and asserted after reload. AgentREPL's reload hook is the mitigation if julia-mcp proves lossy in practice.

**L2 — `-O1` for test runs.** Cold suite 90.7 s → 67.2 s. Better than expected: pkgimages are **not** keyed by the process's `-O` level, so there is no rebuild cost and precompile time is unaffected (85.4 s at `-O2` vs 85.7 s at `-O1`). **Must not be used for `bench/`**, where runtime speed is the quantity being measured.

**L3 — deliver it through the existing chezmoi fan-out.** This work is not Tortuosity-specific and must not live in this repo. Amin's machine already has the right mechanism, and this follows it exactly rather than inventing a parallel one.

*How the existing mechanism works.* `~/.agents/` is the canonical, cross-agent source of truth (chezmoi source `dot_agents/`). `~/.agents/AGENTS.md` is the shared instruction set; `~/.claude/CLAUDE.md` is a generated stub that just does `@~/.agents/AGENTS.md`, and `~/.codex/AGENTS.md` is a generated `{{ include "dot_agents/AGENTS.md" }}`. Skills live in `~/.agents/skills/` and are fanned out by `.chezmoiscripts/run_onchange_link-agent-skills.{ps1,sh}.tmpl`, which links each skill into `~/.claude/skills/` — directory junctions on Windows (no admin rights needed), symlinks elsewhere — and prunes links whose skill has gone. The script re-runs because its rendered body embeds a `# skills: {{ range (glob …) }}` line, so the content changes whenever the skill set does.

*What to add, in the same shape:*

1. **Canonical definition:** `~/.agents/mcp/servers.json` (chezmoi source `dot_agents/mcp/servers.json`) — one neutral, declarative record per server (`command`, `args`, `env`), agent-agnostic.
2. **Fan-out script:** `.chezmoiscripts/run_onchange_sync-agent-mcp.{ps1,sh}.tmpl`, modelled on the skills linker, including its re-run trigger (embed a digest of `servers.json` in a comment line) and its pruning discipline (remove entries the script previously added once they disappear from the canonical file).
3. **Per-agent rendering**, because unlike skills there is no shared on-disk format — each agent needs its own:
   - **Claude Code** — `~/.claude.json`, key `mcpServers`. **Constraint discovered while reading the setup: `~/.claude.json` is NOT chezmoi-managed** (only `dot_claude/private_settings.json.tmpl`, `CLAUDE.md` and the statusline scripts are). It is 85 KB of Claude-managed state. The script must therefore **merge into `mcpServers` and write back**, never template the whole file. Preferred: shell out to `claude mcp add --scope user …`, which is the supported path and avoids hand-editing live state; JSON merge only as fallback.
   - **Codex** — `~/.codex/config.toml`, `[mcp_servers.<name>]` sections with `command` / `args` / `env`. Currently no such sections exist. Same rule: merge, do not overwrite — that file also carries hand-written content (e.g. the `oxygpt` profile in `dot_codex/oxygpt.config.toml`).
   - **Copilot** — `~/.copilot/settings.json`. Its MCP schema was **not verified** during this investigation; confirm before implementing rather than assuming it matches either of the above.
4. **Note in `~/.agents/AGENTS.md`** that MCP servers are defined once in `~/.agents/mcp/` and synced, mirroring the existing note about skills — so the next person edits the canonical file rather than a generated one.

*Acceptance:* after `chezmoi apply` on a clean machine, `julia_eval` (or the chosen server's equivalent) is callable from Claude Code **and** Codex without any manual per-agent configuration, and removing a server from `servers.json` removes it from both on the next apply.

**L4 — a global `julia-workflow` skill.** Lives at `~/.agents/skills/julia-workflow/SKILL.md`, so the existing linker fans it out with no extra work. Same frontmatter convention as the other skills (`name`, plus a `description` written with strong trigger phrasing — it must fire on *any* Julia work, not just on this package).

This skill is the guardrail layer, and it is what makes the persistent session safe rather than merely fast. It must cover:

- **The rule: never invoke `julia` through Bash/PowerShell** to evaluate code, run tests, or run scripts. Always go through the MCP session. This is the single behaviour change that produces the 106 s → ~2 s result, and agents will default to spawning processes unless told plainly not to.
- **The exceptions, stated explicitly**, so the rule survives contact with cases where it is genuinely wrong. A fresh process *is* correct for: benchmarks (`bench/`, where a warm session pollutes the measurement), verifying that precompilation itself works, the full `Pkg.test()` release gate, and reproducing a user-facing cold-start problem.
- **Staleness discipline.** After editing any `.jl`, reload and then **verify the reload applied** — do not assume. The measured failure mode is that a missed reload returns results for code that was never compiled, which reads exactly like a passing test. Concrete check: have the edit bump a probe value and assert it after reloading. Note the ~1 s filesystem-watcher latency, and that one edit in a 12-cycle run was never picked up at all.
- **When to reset the session.** Reset on: a struct or `const` redefinition that errors, world-age errors, results that contradict the source, switching git branches, and any change to `Project.toml` / `Manifest.toml`. Also reset when results simply stop making sense — a cheap reset beats debugging a poisoned session.
- **How to reset**, naming the chosen server's tool (`julia_restart` for julia-mcp, `reset` for AgentREPL), and stating plainly that reset clears **all** session state.
- **Do not use the session as storage.** Treat it as a cache of compiled code, not a place to keep variables between steps. Anything that must persist belongs in a file. This keeps resets cheap and stops agents building workflows that a necessary reset would destroy.
- **`-O1` for test runs, never for benchmarks** (L2), and the environment note that this package's `[extras]`/`[targets]` layout means `--project=.` cannot `using Test` — so the session needs TestEnv activation.

### Phase 2 — shipped changes (users feel these; both are neutral-or-better)

| id | item | kind | status | measured effect |
| --- | --- | --- | --- | --- |
| S1 | Move HDF5, LsqFit, ImageFiltering, ImageMorphology behind `[weakdeps]`/`[extensions]` | shipped | pending | **212 → 134 packages**, load **6.1 s → 4.1 s** |
| S2 | Make the CUDA extension's `@compile_workload` honour the parent package's preference | shipped | pending | unblocks a dev-time toggle worth **−75 s/edit**; no user-visible change |

**S1 — dependency reduction.** Measured against a matched control with workloads disabled on both sides. Attribution of the closure:

| direct dep | closure | packages unique to it | `using` cost |
| --- | --- | --- | --- |
| HDF5 | 62 | 26 | +0.35 s |
| LsqFit | 68 | 26 | +1.30 s |
| ImageFiltering | 80 | 9 | +1.18 s |
| ImageMorphology | 60 | 6 | +1.11 s |

The API surface to be moved is **12 call sites across 7 symbols in 3 files**: `curve_fit`/`stderror` (`transient_fitting.jl:131,234,235`), `h5open` (`utils.jl:35`), `imfilter`/`Kernel.gaussian`/`centered` (`imgen.jl:42-43`), `label_components` (`imgen.jl:146`). Note that `Imaginator` is a **submodule** with its own `using ImageFiltering` / `using ImageMorphology` and needs the same treatment in its own scope — a stub injected only into the parent module is not enough.

This is the only item users ever see, and it makes their load **faster**, not slower. The cost to them is an interface change: callers of `fit_effective_diffusivity` or the HDF5 writer must load the corresponding package first. That is the standard Julia extension idiom, but it is a breaking change to the shipped API and needs release notes and doc updates.

**S2 — extension workload guard.** `PrecompileTools.workload_enabled` resolves its preference against `@__MODULE__`. For `src/Tortuosity.jl` that is `Tortuosity`; for `ext/TortuosityCUDAExt.jl` it is the **extension** module, which has its own UUID (`54913a40-76ee-5b11-a213-d3fc310144ae`). Consequences, all verified:

- `set_preferences!(Tortuosity, "precompile_workload" => false)` works but reaches only the CPU workload (~16 s), leaving the extension's (~75 s) running.
- `set_preferences!` **rejects** the extension's UUID; the name+UUID tuple form fails; a hand-written `[TortuosityCUDAExt]` section in `LocalPreferences.toml` is **silently ignored**, because Preferences cannot map an extension name to a UUID.

Measured directly: with the CPU workload disabled, `Tortuosity` precompiles in **15.5 s** while `TortuosityCUDAExt` takes **71.5 s** — the extension is 82 % of the workload cost and is currently unreachable by the documented mechanism.

The fix is a one-line change to the guard at `ext/TortuosityCUDAExt.jl:93` so it also consults the parent package's preference. **Users are unaffected** — the workload stays on by default. This only makes the developer-side switch work as documented.

**Related trap, worth recording.** The *global* lever `set_preferences!(PrecompileTools, "precompile_workloads" => false; force=true)` does reach the extension, but writing that TOML by hand is **silently inert**: `PrecompileTools.enabled` stays `true` unless PrecompileTools is a **direct dependency** of the environment. Byte-identical file, opposite outcome. This produced two wrong measurements during the investigation before it was caught. If the global lever is ever used, assert `PrecompileTools.enabled == false` afterwards.

## Deferred

Both are genuinely worthwhile and both are measured. They are deferred because their gains largely overlap with Phase 1 once the inner loop is warm, and because their effort and risk are the highest in the set. Revisit if the full-suite gate becomes the bottleneck.

| id | item | measured effect | why deferred |
| --- | --- | --- | --- |
| X1 | Dependency sysimage (deps only, excluding Tortuosity so edits never invalidate it) | edit→precompile **83.0 s → 16.7 s**; load **5.74 s → 1.50 s**; cold suite 87.8 → 82.6 s | Helps only processes that *start*. Once L1 lands, the inner loop no longer starts processes, so this reduces to an optimisation of the occasional full-suite gate and CI. Costs 783 s to build and 710 MiB on disk, must be rebuilt whenever deps change, and a stale image disagrees with the manifest silently. |
| X2 | Parallel test execution (ReTestItems-style worker pool) | suite **101 s → 35 s** (6 workers); **28.8 s** with sysimage; **22.4 s** with sysimage + `-O1` | Highest effort in the set — restructuring ~15 test files into `@testitem` blocks — and it surfaces test-isolation bugs (shared state, RNG seeding, GPU contention across workers). Worth ~45 s on a gate that runs at phase boundaries only. |

Scaling data for X2, for whoever picks it up:

| workers | 1 | 2 | 4 | 6 | 8 |
| --- | --- | --- | --- | --- | --- |
| plain | 101.1 | 60.1 | 43.8 | **35.0** | 37.1 |
| + sysimage | 82.4 | 55.1 | 38.8 | **28.8** | 29.6 |
| + sysimage + `-O1` | 78.4 | 45.4 | 30.1 | 23.9 | **22.4** |

It plateaus at 4–6 workers; the floor is each worker's own load cost plus the slowest single file.

## Rejected

Each of these was measured, not reasoned about. They are recorded so they are not re-investigated.

| item | verdict | evidence |
| --- | --- | --- |
| **Ship with `@compile_workload` disabled** | **rejected — violates constraint 1** | Worth −75 s/edit, but the workloads exist to buy end users a fast first call. The measured proxy for removing them is **+27 s of first-call compilation** (cold suite 89.0 → 116.1 s, paired, both arms sanity-checked). This is exactly the user-visible warm-up cost the governing constraint forbids. Keep them on in the shipped package; disabling is a campaign-time switch only. |
| **Split the package into sub-packages so edits invalidate less** | **rejected — premise false** | Only ~16 s of the ~92 s precompile is our code; ~59.5 s is dependency loading. This is the intuitive move and the attribution kills it. |
| **Keep CUDA out of the iteration environment** | **rejected — no effect** | A CUDA-free env costs 91.6/100.4 s per edit, statistically identical to the CUDA-containing env's CPU-only path (81–97 s). The extension is built only when CUDA is actually *loaded*, not merely installed. An earlier claim that this saved ~60 s/edit was inferred from precompile-cache file counts and was **wrong**; direct measurement retracted it. |
| **`-O0` instead of `-O1`** | **rejected — not worth it** | 65.8 s vs 67.2 s cold suite, for a materially larger runtime penalty. |

## Execution

### Phases

- **Phase 0 — baseline and branch.** Commit this plan. Record in the Progress log, as measured now rather than quoted from this document: the current `Pkg.test()` assertion count and duration, and one edit→green timing for a single test file through the existing fresh-process workflow. Those two numbers are what Phases 1–2 are judged against; the figures in this file were measured against an isolated copy and are evidence for the design, not the run's baseline. Branch `perf/dev-loop` off `main` for the Phase 2 work (Phase 1 touches no file in this repository).
- **Phase 1 — the development loop (L1–L4).** Lands in the chezmoi dotfiles repo, not here. L1 then L3 then L4; L2 is independent and can go at any point. Exit: a persistent Julia session is reachable from Claude Code **and** its entry is present in `~/.codex/config.toml`; the `julia-workflow` skill is in `~/.agents/skills/` and linked into `~/.claude/skills/`; and a measured edit→green of ≤5 s is in the Progress log next to the Phase 0 baseline.
- **Phase 2 — shipped changes (S1, S2).** S2 first — it is one line and independent. Then S1, whose parts are separable and should be committed separately per dependency, with the suite green after each. Exit: suite green at or above the Phase 0 assertion count; a fresh measurement of `using Tortuosity` recorded; release notes and docs updated for the S1 interface change.
- **Phase 3 — verification and consolidation.** Independent adversarial review of the whole diff; foreground full `Pkg.test()`; refresh this file's numbers against what was actually observed; Final report written into this file.

### Orchestration

Per Amin's standing preference: the master stays context-light and directs; all reading, editing, testing and measuring is delegated; one write-agent at a time; every write is checked by an independent read-only reviewer who re-runs the tests rather than trusting the report; agents report in the compact format from campaign 1; the Progress log in this file — not anyone's context — is the state. The full protocol, including the `/goal` evaluator-visibility rules, is in `2026-08-08-matrix-path-optimization.md` §Orchestration protocol and **applies verbatim**.

One adaptation specific to this plan. Phase 1's acceptance is a *configuration* fact on the machine, not a test result, and the `/goal` evaluator can neither run commands nor see inside subagents. The master must therefore print the literal evidence in its own message text: the `claude mcp list` output showing the Julia server, the `[mcp_servers.*]` block from `~/.codex/config.toml`, the directory listing showing `julia-workflow` linked into `~/.claude/skills/`, and the two edit→green timings side by side. Paraphrase is not evidence.

A second adaptation: **Phase 1 changes the very tooling the agents are running under.** An agent cannot reconfigure its own MCP servers and then use them in the same session — the client loads them at startup. Treat "restart required, verify in a fresh session" as a normal step, not a failure, and have the master record the verification rather than assume it.

### Git discipline

This plan spans two repositories.

- **Dotfiles (Phase 1)** — work in the chezmoi source (`~/.local/share/chezmoi`), apply, and commit there. The `sync-chezmoi` skill is the sanctioned route; use it rather than hand-editing generated targets. Never edit `~/.claude/skills/*`, `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md` or any other generated file directly — edit the source and re-apply.
- **Tortuosity (Phase 2)** — branch `perf/dev-loop`; one conventional commit per accepted change referencing the item id; path-scope every commit to `src/`, `test/`, `ext/`, `docs/` plus this file; never `git add -A` or `git commit -a`; no pushes; no attribution trailers. **Never modify anything under `benchmarks/`** — that directory belongs to the JOSS effort.
- Commit authorization for unattended runs is per-campaign, granted when Amin starts it.

### Goal condition (paste to `/goal` when starting)

**Do not invoke as `/goal docs/plans/2026-08-09-development-loop-latency.md`.** The argument to `/goal` *is* the completion condition, not a file to execute — that form sets the literal filename as the condition and the evaluator judges nonsense. Paste the block below instead.

```
/goal Execute the plan in docs/plans/2026-08-09-development-loop-latency.md to completion. Read that file first, then resume from its Progress log. The condition is met when, and only when, your visible message text contains the exact line: LATENCY PLAN COMPLETE - all conditions met. Print that line only after you have personally verified all six: (1) every L- and S-series item is terminal (done, rejected, BLOCKED or REVERTED) in the Progress log; (2) a persistent Julia MCP session is reachable from Claude Code and its server entry is present in ~/.codex/config.toml, with both printed as literal evidence; (3) the julia-workflow skill exists in ~/.agents/skills/ and is linked into ~/.claude/skills/; (4) a measured edit-to-green latency of 5 s or less is recorded in the Progress log beside the Phase 0 baseline it replaces; (5) a full Pkg.test() run is green with assertions at or above the Phase 0 count recorded in the Progress log; (6) the Final report is written into this plan file. Constraints: never ship with @compile_workload disabled, never weaken or skip a test to make a change pass, never use git add -A or git commit -a, never leave the tree red, never modify anything under benchmarks/. Stop after 40 turns if not complete, print LATENCY PLAN HALTED and a status summary.
```

Roughly 1.4 k characters, inside the 4 000-character limit. It keys on one literal string for the same reason campaign 1's does: the evaluator is a small fast model reading only the transcript, so judgement belongs to the orchestrator, which can verify. `LATENCY PLAN HALTED` is a distinct marker so an exhausted turn budget cannot read as success, and both markers differ from the other plans' `CAMPAIGN COMPLETE` so concurrent runs cannot cross-trigger.

## Expected outcome

| loop | today | after Phase 1 | after Phase 1 + 2 |
| --- | --- | --- | --- |
| edit → one test file | ~106 s | **~2 s** | ~2 s |
| edit → full CPU suite | ~193 s | ~85 s | ~80 s |
| full `Pkg.test()` | 307 s | — | — |

Phase 1 alone captures essentially the whole inner-loop win, which is the loop that actually hurts. Phase 2 is about the shipped package, not our latency.

## Measurement appendix

**Method.** All measurements were taken against an **isolated copy** of the package in dedicated scratch environments, never the working tree, because the matrix-path campaign was running concurrently. Precompile-after-edit was induced by appending a comment to a source file in the copy (a real content change → new cache slug). Cold/warm test-suite figures come from running the 15 CPU test files in one process, twice.

**Reproducibility caveats, stated honestly:**

- Machine: Julia 1.12.6, Windows 11, 20 threads, 128 GB RAM. Windows Defender real-time protection was **on** and its exclusion list could not be read without administrator rights; each edit writes ~77 MB of fresh pkgimage into a 1.7 GB depot.
- The earliest figures (the 79.9 s / 153.9 s per-edit numbers) were taken while a 27 GB benchmark run was loading the machine and are **upper bounds**. Figures quoted in this document are the later, quiet-machine measurements.
- The cold-suite baseline moved from 100.1 s to 89.0 s during the investigation because the matrix-path campaign's optimisations landed in between. Only paired comparisons taken against a single snapshot of both package and tests are quoted for on/off decisions.
- One scratch environment resolved **CUDA 13.3.0 while CUDA.jl had been precompiled for 13.2.0**, warning on every load. Whether the real test environment carries the same mismatch was not checked and is worth a look, since that class of thing causes silent re-precompilation.

**Corrections made during the investigation**, recorded because each was a wrong number that survived one round of review: the CUDA-free-environment claim (retracted, see Rejected); a Revise timing that measured a no-op reload rather than an edit cycle; and two `PrecompileTools` global-preference measurements taken with a preference file that was silently inert.

The measurement harness lived in a session scratchpad and is **not** preserved — it was throwaway by design, since the working tree was in use by a concurrent campaign at the time. Anything a future run needs to re-verify must be re-measured, which is what Phase 0 is for.

## Progress log

**This log is the plan's state.** Read it before doing anything; append one line per terminal item, including rejections, blockers and reverts. An empty log means Phase 0 has not run.

Format: `date — id(s) — status — measured effect — commit sha (and repo) — reviewer verdict`.

### Phase 0 baseline (measured 2026-08-09 on the working tree, not the isolated copy)

| quantity | value | how it was taken |
| --- | --- | --- |
| full `Pkg.test()` assertions | **11576 pass / 11576 total** | `julia --startup-file=no --project=. -e "using Pkg; Pkg.test()"`, root testset `Tortuosity.jl` |
| full `Pkg.test()` wall | **265.41 s** (testsets 3m36.8s = 216.8 s) | same run; 48.6 s, 18 %, before the first assertion |
| **edit→green, one test file, fresh process** | **173.9 s** | edit `src/transient.jl` → `julia --startup-file=no --project=. edit_to_green.jl` (TestEnv activate → `using Tortuosity` → assert probe → `include test/test_transient.jl`, 175 assertions, 26.6 s of testset). Precompile inside that: `Tortuosity` 29.0 s + `TortuosityCUDAExt` 92.0 s = 127 s. |

This is the number Phase 1 is judged against. It is **higher than the ~106 s in the body of this plan** because the body's figure was a CPU-only path measured on an isolated copy, whereas the real test environment resolves CUDA, so a source edit also rebuilds `TortuosityCUDAExt` (92 s of the 174 s). The honest baseline for the loop an agent actually runs is 173.9 s.

Confirmed en route: the CUDA 13.3.0 / CUDA.jl-precompiled-for-13.2.0 mismatch flagged as unchecked in the Measurement appendix **is** present in the real test environment — it warns on every extension precompile.

### Item log
