---
title: Development loop latency
created: 2026-08-09
updated: 2026-08-09
status: complete
branch: perf/dev-loop
supersedes: "-"
superseded-by: "-"
related: 2026-08-08-matrix-path-optimization.md
---

> **Status: complete (executed 2026-08-09).** The latency that dominated agent-driven development on this package came from spawning a fresh Julia process per edit, not from the package's structure. Measured on the real working tree, a coding agent paid **173.9 s from source edit to a green test file**. A persistent Julia session reached over MCP takes the same loop to **1.13 s (measured, 154×)**, with no source change and no cost to end users. That tooling ships globally through the existing `~/.agents` + chezmoi fan-out, so every agent (Claude, Codex, Copilot) and every Julia project gets it once.
>
> **Outcome:** L1, L3, L4 done · L2 **rejected on execution** (`-O1` turns the suite red) · S2 done · S1 done for three of four dependencies, **blocked** for ImageMorphology · X1, X2 deferred as planned. Full suite green at **11576 / 11576, 222.93 s** against a 265.41 s baseline. Everything still open is gathered in [Open items and follow-ups](#open-items-and-follow-ups) at the end of this file — that section, not this paragraph, is where a future run starts.
>
> **How to read this document.** Sections above the Progress log are the plan **as designed on 2026-08-09**, kept so the reasoning survives; where execution contradicted a design-time number, the contradiction is marked inline rather than silently overwritten. Sections from the Progress log down are **what actually happened**. Design-time figures were measured against an isolated copy of the package; execution figures against the working tree. Where the two disagree, the execution figure wins.

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
| L1 | Adopt an MCP server giving agents a persistent Julia session | dev-only | **done** | edit→green **173.9 s → 1.13 s** (measured, ~154×) |
| L2 | Pass `-O1` on all test runs | dev-only | **rejected** | breaks a test — see below and Rejected |
| L3 | Deliver the MCP server config from `~/.agents/` via a chezmoi sync script, to Claude / Codex / Copilot | dev-only | **done** | one canonical file syncs Claude + Codex + Copilot |
| L4 | Global `julia-workflow` skill in `~/.agents/skills/` carrying the guardrails | dev-only | **done** | prevents the silent-stale failure mode |

**L1–L4 are one change delivered in four parts.** L1 picks the server, L3 installs it everywhere, L4 makes agents actually use it correctly. Shipping L1 without L4 is the dangerous combination: a persistent session that agents use naively returns stale results silently (see below).

**L1 — persistent session via MCP.** Three servers exist and none require any change to our source:

| server | tools | session model | Revise | maturity (2026-08-09) |
| --- | --- | --- | --- | --- |
| [julia-mcp](https://github.com/aplavin/julia-mcp) (aplavin) | 3 | per-project, Python server, stdio | automatic | 80★, 36 commits |
| [AgentREPL.jl](https://github.com/samtalki/AgentREPL.jl) | 8 | Malt.jl workers, Julia-native, 6 deps | auto-loads, plus a hook that nudges the agent to reload after `.jl` edits | 7★, 68 commits |
| [Kaimon.jl](https://github.com/kahliburke/Kaimon.jl) | 35–49 | HTTP + ZMQ, 30+ deps | manual | debugger, semantic search |

Recommendation: **start with julia-mcp**, primarily for its TestEnv integration — which matters here because the package uses `[extras]`/`[targets]` rather than a `test/Project.toml`, so `--project=.` cannot `using Test`. Kaimon is over-scoped for this problem.

> **Outcome — julia-mcp was adopted and the TestEnv reasoning held.** It ships pinned at commit `0000000045a2d345f15bd3ec3aa90ffb421dc14b`, launched as `uvx --from git+https://github.com/aplavin/julia-mcp@<sha> julia-mcp --threads=auto`. Two facts discovered during execution that the design table did not capture: the repo declares a `[project.scripts]` entry point, so **no clone is needed** and `uvx` builds and caches it (~25 s once, 2.8 s per launch thereafter); and this machine has **no system Python**, which `uvx` handles by provisioning its own — had that not been true, AgentREPL.jl (Julia-native) would have been the fallback. Passing `env_path = <pkg>/test/` makes the server run `using TestEnv; TestEnv.activate()` and treat the parent as project root, which is exactly the `[extras]`/`[targets]` workaround this row was chosen for. Anything after `julia-mcp` in `args` is forwarded verbatim as Julia's own flags — that is the hook L2 tried to use.

*Durability was tested, because a daemon that agents hammer for hours is not the same thing as an interactive REPL.* Twelve consecutive edit → revise → test cycles:

- **Latency flat:** revise 0.06–0.08 s, tests 1.65–2.07 s, no drift across cycles.
- **Memory plateaus:** 978 → 1136 MiB by cycle 8, stable thereafter. Growth, not a leak.
- **Survives failures:** a failing testset throws, session stays usable (next run 1.64 s).
- **Struct redefinition worked** on Julia 1.12 — the classic Revise dealbreaker is largely gone. Verified only to the extent that `revise()` reported success and the session stayed healthy; stale-instance semantics were not probed.

**The one hazard, and it is a real one.** A missed `revise()` returns **stale results silently**. This was hit during the investigation: a measurement that looked like a fast edit cycle was a no-op reload, and an agent would have read it as a passing test on code it never compiled. **Acceptance criterion for L1: the workflow must verify that the revision applied rather than assume it** — e.g. a probe value bumped by the edit and asserted after reload. AgentREPL's reload hook is the mitigation if julia-mcp proves lossy in practice.

**L2 — `-O1` for test runs. REJECTED on execution; the design-phase measurement was correct but incomplete.** The speed claim holds (cold suite 90.7 s → 67.2 s; pkgimages are not keyed by the process's `-O` level, so there is no rebuild cost). What the design phase never checked is whether the suite still *passes* at `-O1`. It does not.

`-O1` was shipped as the trailing argument of the MCP server, and the very first warm run came back **174 passed, 1 failed** on a file that passes 175/175 in a fresh `-O2` process. Attribution, four fresh runs differing only in flags:

| flags | result |
| --- | --- |
| `-O2 --threads=1` | 175 / 175 pass |
| `-O2 --threads=auto` (20 threads) | 175 / 175 pass |
| `-O1 --threads=1` | 174 pass, **1 fail** |
| `-O1 --threads=auto` | 174 pass, **1 fail** |

Thread count is irrelevant; `-O1` is the whole effect. The failure is `test/test_transient.jl:462`, `@test m_override == m_prob` — an **exact bitwise float equality** between two mathematically equivalent computations. At `-O2` both paths happen to generate identical code; at `-O1` they diverge in the last 1–2 ULP (`0.12524001601324483` vs `0.12524001601324478`) and the assertion fails. Making the suite green at `-O1` would mean loosening that assertion to `≈`, i.e. weakening a test to accommodate a dev-time speed lever. That trade is not available.

`-O1` was therefore removed from the server configuration. The session runs at the default `-O2`, and the `julia-workflow` skill carries the durable form of the lesson: lowering Julia's optimisation level changes floating-point codegen, so it is never a free speed lever — re-run the full suite at that level before adopting it.

**Separately worth Amin's attention, and deliberately not fixed here:** `test_transient.jl:462` asserting bitwise float equality across two code paths is latent fragility independent of this plan. It survives today only because `-O2` codegen happens to match. Any change to Julia's optimiser, a dependency's inlining, or the surrounding code can break it with no bug in the package. Loosening it to a tolerance-based comparison is a reasonable separate change, but it is out of scope for a latency campaign and was not made.

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

> **Outcome — built as designed, with three corrections found in execution.**
>
> 1. **Copilot's path was wrong in this plan.** `~/.copilot/settings.json` is *not* where MCP lives; it is `~/.copilot/mcp-config.json`, with a `type: "stdio"` field the other two agents do not use. The plan flagged this as unverified and it was indeed wrong — the verification was worth doing.
> 2. **Claude's `mcpServers` cannot live in `settings.json` at all.** It is only ever `~/.claude.json`, which is unmanaged 85 KB machine state — so the script shells out to `claude mcp add --scope user`, the supported path, exactly as the plan preferred. `claude mcp add` is **not idempotent** (it errors if the name exists), so the script removes first and ignores the failure.
> 3. **Codex is handled by a marker-delimited generated block appended at the end of `~/.codex/config.toml`**, since TOML sections are order-sensitive and anything written after the block would be swallowed by its last section. Hand-written content above it is preserved byte-for-byte.
>
> Pruning was proven, not assumed: removing the server from `servers.json` and applying made it vanish from all three agents while the four claude.ai connectors and the chrome-devtools plugin survived untouched; restoring it brought it back. That test exposed a real bug — an empty JSON object yields a one-element array containing `$null` in PowerShell, which wrote a `[mcp_servers.]` section with a blank name — since fixed. Files: `dot_agents/mcp/servers.json`, `.chezmoiscripts/run_onchange_sync-agent-mcp.{ps1,sh}.tmpl`, `.chezmoiignore` platform gate, and a note in `dot_agents/AGENTS.md`.

**L4 — a global `julia-workflow` skill.** Lives at `~/.agents/skills/julia-workflow/SKILL.md`, so the existing linker fans it out with no extra work. Same frontmatter convention as the other skills (`name`, plus a `description` written with strong trigger phrasing — it must fire on *any* Julia work, not just on this package).

This skill is the guardrail layer, and it is what makes the persistent session safe rather than merely fast. It must cover:

- **The rule: never invoke `julia` through Bash/PowerShell** to evaluate code, run tests, or run scripts. Always go through the MCP session. This is the single behaviour change that produces the 106 s → ~2 s result, and agents will default to spawning processes unless told plainly not to.
- **The exceptions, stated explicitly**, so the rule survives contact with cases where it is genuinely wrong. A fresh process *is* correct for: benchmarks (`bench/`, where a warm session pollutes the measurement), verifying that precompilation itself works, the full `Pkg.test()` release gate, and reproducing a user-facing cold-start problem.
- **Staleness discipline.** After editing any `.jl`, reload and then **verify the reload applied** — do not assume. The measured failure mode is that a missed reload returns results for code that was never compiled, which reads exactly like a passing test. Concrete check: have the edit bump a probe value and assert it after reloading. Note the ~1 s filesystem-watcher latency, and that one edit in a 12-cycle run was never picked up at all.
- **When to reset the session.** Reset on: a struct or `const` redefinition that errors, world-age errors, results that contradict the source, switching git branches, and any change to `Project.toml` / `Manifest.toml`. Also reset when results simply stop making sense — a cheap reset beats debugging a poisoned session.
- **How to reset**, naming the chosen server's tool (`julia_restart` for julia-mcp, `reset` for AgentREPL), and stating plainly that reset clears **all** session state.
- **Do not use the session as storage.** Treat it as a cache of compiled code, not a place to keep variables between steps. Anything that must persist belongs in a file. This keeps resets cheap and stops agents building workflows that a necessary reset would destroy.
- ~~**`-O1` for test runs, never for benchmarks** (L2)~~ — **superseded**: L2 was rejected, so the shipped skill instead warns that lowering Julia's optimisation level changes floating-point codegen and is never a free speed lever. The "never benchmark in the warm session" rule survives on its own merit: a warm session has already paid the JIT cost the measurement is supposed to include.
- The environment note that this package's `[extras]`/`[targets]` layout means `--project=.` cannot `using Test` — so the session needs TestEnv activation.

> **Outcome — shipped at `~/.agents/skills/julia-workflow/SKILL.md`** and linked into `~/.claude/skills/` by the existing linker with no extra work, as predicted. It covers every bullet above. Worth noting for anyone editing it: the skill is what makes the session *safe*, not merely fast — the staleness bullet is the one that earns its keep, and it is written as a concrete probe-and-assert procedure rather than an exhortation to be careful.

### Phase 2 — shipped changes (users feel these; both are neutral-or-better)

| id | item | kind | status | measured effect |
| --- | --- | --- | --- | --- |
| S1 | Move HDF5, LsqFit, ImageFiltering, ImageMorphology behind `[weakdeps]`/`[extensions]` | shipped | **done (3 of 4)** — ImageMorphology **BLOCKED** | **213 → 152 manifest stanzas**, load **4.6 s → 3.5 s** |
| S2 | Make the CUDA extension's `@compile_workload` honour the parent package's preference | shipped | **done** | extension precompile **55.0 s → 8.3 s** when the parent preference is off; default unchanged |

**S1 — dependency reduction.** Measured against a matched control with workloads disabled on both sides. Attribution of the closure:

| direct dep | closure | packages unique to it | `using` cost |
| --- | --- | --- | --- |
| HDF5 | 62 | 26 | +0.35 s |
| LsqFit | 68 | 26 | +1.30 s |
| ImageFiltering | 80 | 9 | +1.18 s |
| ImageMorphology | 60 | 6 | +1.11 s |

The API surface to be moved is **12 call sites across 7 symbols in 3 files**: `curve_fit`/`stderror` (`transient_fitting.jl:131,234,235`), `h5open` (`utils.jl:35`), `imfilter`/`Kernel.gaussian`/`centered` (`imgen.jl:42-43`), `label_components` (`imgen.jl:146`). Note that `Imaginator` is a **submodule** with its own `using ImageFiltering` / `using ImageMorphology` and needs the same treatment in its own scope — a stub injected only into the parent module is not enough.

This is the only item users ever see, and it makes their load **faster**, not slower. The cost to them is an interface change: callers of `fit_effective_diffusivity` or the HDF5 writer must load the corresponding package first. That is the standard Julia extension idiom, but it is a breaking change to the shipped API and needs release notes and doc updates.

> **Outcome — three of four moved; ImageMorphology blocked.** HDF5, LsqFit and ImageFiltering are now `[weakdeps]` with one extension each (`ext/TortuosityHDF5Ext.jl`, `TortuosityLsqFitExt.jl`, `TortuosityImageFilteringExt.jl`), so loading one does not drag in the others. `src/weakdeps.jl` holds one stub hook per optional symbol plus `_weakdep_error`; public entry points stay defined in the parent and route through the hooks, so names still resolve and the failure is an actionable message rather than an `UndefVarError`. The submodule trap the plan flagged was real and was handled: `Imaginator` imports the hooks in its own scope via `using ..Tortuosity: _gaussian_blur`.
>
> The two `@compile_workload` lines that needed a moved package went **into the extension that owns them**, so a user who loads LsqFit still gets a compiled first call and a user who does not never pays for it — constraint 1 holds in both directions.
>
> **ImageMorphology could not move**, and the reason is a correction to this plan rather than a shortfall: `label_components` is reached from `_warn_nonpercolating`, which runs by default from every `SteadyDiffusionProblem` under 50 M voxels, and the CPU workload builds one — so the parent package will not precompile without it. Full detail and the failing output are in the Progress log. **The 212 → 134 / 6.1 s → 4.1 s headline in the table above is therefore not achievable as written**; the delivered figure is 213 → 152 manifest stanzas and 4.6 s → 3.5 s. Finishing the fourth dependency is a user-visible behaviour decision, tracked as item **F** in Open items and follow-ups.
>
> The line numbers in the call-site list above had already drifted by execution time; trust the symbol names, not the line numbers.

**S2 — extension workload guard.** `PrecompileTools.workload_enabled` resolves its preference against `@__MODULE__`. For `src/Tortuosity.jl` that is `Tortuosity`; for `ext/TortuosityCUDAExt.jl` it is the **extension** module, which has its own UUID (`54913a40-76ee-5b11-a213-d3fc310144ae`). Consequences, all verified:

- `set_preferences!(Tortuosity, "precompile_workload" => false)` works but reaches only the CPU workload (~16 s), leaving the extension's (~75 s) running.
- `set_preferences!` **rejects** the extension's UUID; the name+UUID tuple form fails; a hand-written `[TortuosityCUDAExt]` section in `LocalPreferences.toml` is **silently ignored**, because Preferences cannot map an extension name to a UUID.

Measured directly: with the CPU workload disabled, `Tortuosity` precompiles in **15.5 s** while `TortuosityCUDAExt` takes **71.5 s** — the extension is 82 % of the workload cost and is currently unreachable by the documented mechanism.

The fix is a one-line change to the guard at `ext/TortuosityCUDAExt.jl:93` so it also consults the parent package's preference. **Users are unaffected** — the workload stays on by default. This only makes the developer-side switch work as documented.

> **Outcome — done, and it was three extensions, not one.** The `:93` line reference was already stale; the workload block is around line 116. `TortuosityMetalExt.jl` and `TortuosityAMDGPUExt.jl` carried the identical bug and got the identical guard. Measured: `TortuosityCUDAExt` precompile **55.0 s → 8.3 s** once `set_preferences!(Tortuosity, "precompile_workload" => false)` is set, and **unchanged with the preference unset**, which is every user's state. That the extension recompiled *at all* when only the parent's preference changed is itself the proof the guard works — the extension now records a compile-time preference against Tortuosity's UUID, so the toggle genuinely invalidates rather than reading a stale cache.
>
> The guard was later reworked in review: importing `PrecompileTools.workload_enabled` directly made five extensions depend on an **unexported** internal under a loose `1.2` pin. It now goes through `Tortuosity._workload_enabled()`, which guards on `isdefined` and falls back to `true`. See Phase 3 in the Progress log.

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
| **`-O1` for test runs (L2)** | **rejected on execution — turns the suite red** | Saves 23 s of cold suite but fails `test/test_transient.jl:462`, which asserts exact bitwise float equality; `-O1` codegen diverges by 1–2 ULP. Confirmed by a 2×2 flag matrix: both `-O2` arms 175/175, both `-O1` arms 174/175, independent of thread count. Full detail in the L2 section. |

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

## Expected outcome versus achieved

Predicted at design time, against what execution actually measured:

| loop | predicted before | predicted after | **measured before** | **measured after** |
| --- | --- | --- | --- | --- |
| edit → one test file | ~106 s | ~2 s | **173.9 s** | **1.13 s** |
| full `Pkg.test()` | 307 s | — | **265.4 s** | **222.9 s** |
| `using Tortuosity` | 6.1 s | 4.1 s (S1) | **4.6 s** | **3.5 s** |
| manifest closure | 212 | 134 (S1) | **213** | **152** |

The two prediction misses both matter and both are recorded in full below: the **baseline was 68 s worse than predicted** (the design figure was a CPU-only path on an isolated copy; the environment an agent actually runs in resolves CUDA, so an edit also rebuilds the 92 s extension), and the **S1 target was unreachable** (ImageMorphology is load-bearing at precompile time). The inner-loop prediction was, if anything, conservative.

Phase 1 alone captured essentially the whole inner-loop win, exactly as designed. Phase 2 was about the shipped package, not our latency.

## Measurement appendix

**Method.** All measurements were taken against an **isolated copy** of the package in dedicated scratch environments, never the working tree, because the matrix-path campaign was running concurrently. Precompile-after-edit was induced by appending a comment to a source file in the copy (a real content change → new cache slug). Cold/warm test-suite figures come from running the 15 CPU test files in one process, twice.

**Reproducibility caveats, stated honestly:**

- Machine: Julia 1.12.6, Windows 11, 20 threads, 128 GB RAM. Windows Defender real-time protection was **on** and its exclusion list could not be read without administrator rights; each edit writes ~77 MB of fresh pkgimage into a 1.7 GB depot.
- The earliest figures (the 79.9 s / 153.9 s per-edit numbers) were taken while a 27 GB benchmark run was loading the machine and are **upper bounds**. Figures quoted in this document are the later, quiet-machine measurements.
- The cold-suite baseline moved from 100.1 s to 89.0 s during the investigation because the matrix-path campaign's optimisations landed in between. Only paired comparisons taken against a single snapshot of both package and tests are quoted for on/off decisions.
- One scratch environment resolved **CUDA 13.3.0 while CUDA.jl had been precompiled for 13.2.0**, warning on every load. Whether the real test environment carries the same mismatch was not checked and is worth a look, since that class of thing causes silent re-precompilation. — **Checked during execution: it does.** The real test environment warns on every extension precompile. Tracked as item **I** in Open items and follow-ups.

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

- 2026-08-09 — **Phase 0** — done — baseline above (11576 assertions / 265.41 s; edit→green 173.9 s) — `569c0e7` (Tortuosity) — branch `perf/dev-loop` cut from `main`.
- 2026-08-09 — **L1, L3, L4** — done — **edit→green 173.9 s → 1.13 s (~154×)** — `e36e53c` (chezmoi dotfiles) — verified by the master against the live machine, not by agent report: `claude mcp list` shows `julia … ✔ Connected`, `~/.codex/config.toml` carries `[mcp_servers.julia]`, `~/.claude/skills/julia-workflow` is a Junction to `~/.agents/skills/julia-workflow`. Server is [julia-mcp](https://github.com/aplavin/julia-mcp) pinned at `0000000045a2d345`, launched via `uvx --from git+…` (no clone, no system Python needed), delivered from the canonical `~/.agents/mcp/servers.json` by `run_onchange_sync-agent-mcp.{ps1,sh}.tmpl`. Pruning was proven by removing the server, applying, confirming it vanished from all three agents while the four claude.ai connectors and the chrome-devtools plugin survived, then restoring it.
- 2026-08-09 — **L2** — **rejected** — would save ~23 s of cold suite but fails `test_transient.jl:462` — `019041a` (chezmoi dotfiles) removes it — see the L2 section for the 2×2 attribution matrix.

- 2026-08-09 — **S2** — done — extension precompile **55.0 s → 8.3 s** with `set_preferences!(Tortuosity, "precompile_workload" => false)`; **default unchanged, workload still ON** — `95cc2b5` — reviewer CONFIRMED the default path: `workload_enabled` defaults `true` on both the global and per-package preference and on its `catch` branch, so users are unaffected. The plan's `ext/TortuosityCUDAExt.jl:93` was stale; the real block is ~line 116. The same guard was applied to the Metal and AMDGPU extensions, which had the identical bug.
- 2026-08-09 — **S1** — **done for HDF5, LsqFit, ImageFiltering; BLOCKED for ImageMorphology** — **213 → 152 manifest stanzas, `using Tortuosity` 4.6 s → 3.5 s, 277 → 201 packages loaded** — `139bb80`, `e94504c`, `5e04cea`, `667f614`, `df77df6` — full suite green at **11576/11576** after each.

**Why ImageMorphology could not move — this is a correction to the plan, not a shortfall in execution.** It was implemented fully and the package then failed to precompile at all:

```
ERROR: LoadError: Imaginator.trim_nonpercolating_paths requires ImageMorphology, ...
in expression starting at src\Tortuosity.jl:1
```

`label_components` is reached from `_warn_nonpercolating` (`src/simulations.jl:63-68`), which runs **by default** from every `SteadyDiffusionProblem` on an image under 50 M voxels; `src/caverns.jl` uses it too; and the CPU `@compile_workload` builds a `SteadyDiffusionProblem`. So ImageMorphology is load-bearing at precompile time. Making it optional would mean either dropping the percolation check from the default path or silently degrading it when the package is absent — both user-visible behaviour changes well beyond "load a package first", and both barred by constraint 1. The work was reverted cleanly.

**Consequence: the plan's headline S1 figure of 212 → 134 packages / 6.1 s → 4.1 s is not reachable** without changing when that warning fires. The achieved figure is 213 → 152 stanzas and 4.6 s → 3.5 s. Anyone quoting the original number should stop.

- 2026-08-09 — **Phase 3** — done — independent review returned **DO NOT SHIP** on four findings; all four fixed — `883a41e`, `1a3b04a`, `fc19b60` — orchestrator then re-ran the full suite in the foreground: `EXIT=0`, **11576 / 11576, 222.93 s** (Phase 0: 11576 / 265.41 s). The reviewer was given the diff and forbidden from running Julia, so its findings are independent of the writer's own test claims; the green run is the orchestrator's, not an agent's report.

**The L1 measurement, in full.** Warm session on `env_path = …/Tortuosity/test/` (which makes julia-mcp run `using TestEnv; TestEnv.activate()` and take the parent as project root — this is how the `[extras]`/`[targets]` layout is handled). Session warm-up 44.0 s, paid once. Then three consecutive edit → reload → test cycles on `test/test_transient.jl`:

| cycle | edit→green | assertions | probe verified |
| --- | --- | --- | --- |
| 1 | 4.58 s | 175 / 175 | yes (probe = 2) |
| 2 | 1.14 s | 175 / 175 | yes (probe = 3) |
| 3 | 1.13 s | 175 / 175 | yes (probe = 4) |

Cycle 1 is slower because the reloaded methods are re-JITted once. **Steady state is 1.13 s against a 173.9 s fresh-process baseline.**

## Final report

**Status: complete.** Phase 1 delivered and verified against the live machine. Phase 2 delivered except ImageMorphology, which is blocked on evidence recorded above. Phase 3 done: an independent adversarial review returned DO NOT SHIP, its four blocking findings were fixed, and the full suite was then re-run in the foreground by the orchestrator — `EXIT=0`, **11576 / 11576 pass, 222.93 s wall** against the Phase 0 baseline of 11576 / 265.41 s. Branch `perf/dev-loop` is green and ready to merge; nothing was pushed.

**What was achieved.**

| loop | before | after | factor |
| --- | --- | --- | --- |
| edit → one test file | 173.9 s | **1.13 s** | **154×** |
| `using Tortuosity` | 4.6 s | **3.5 s** | 1.3× |
| manifest stanzas | 213 | **152** | −61 |
| full `Pkg.test()` | 265.4 s / 11576 assertions | 229 s / 11576 assertions | — |

The inner loop — the one that actually hurt — is the whole story, and it came from the process model, exactly as the plan predicted. No source change was needed for it.

**The three things this run learned that the plan had wrong.**

1. **L2 (`-O1`) is not free.** It saves 23 s of cold suite and fails `test_transient.jl:462`, which asserts exact bitwise float equality. The design phase measured speed and never checked correctness. Rejected and removed.
2. **S1 is only 3/4 achievable.** ImageMorphology is load-bearing at precompile time; the 212 → 134 target was never reachable.
3. **The real baseline was 173.9 s, not ~106 s.** The plan's figure came from a CPU-only path on an isolated copy; the environment an agent actually runs in resolves CUDA, so an edit also rebuilds the 92 s extension.

**Review findings, fixed** — `883a41e`, `1a3b04a`, `fc19b60`, suite re-verified green after all three:

1. **The docs claimed weak dependencies install with the package.** They do not — `Pkg.add` resolves `[deps]` only, so both the README and `docs/src/index.md` quick-starts died on their first line. Fixed, along with the same sentence in `CHANGELOG.md`, the GPU-backend snippets, and four docstrings, all now matching the stub error's wording.
2. **Five extensions imported `PrecompileTools.workload_enabled`**, which is *not exported* and pinned loosely at `1.2`. A 1.x release dropping it would be a **load-time** failure in all five — and for the GPU extensions that means the backend registration never runs and `using Tortuosity; using CUDA` silently falls back to CPU. Replaced with `Tortuosity._workload_enabled()`, which guards on `isdefined` and returns `true` when absent; default-ON confirmed.
3. **`docs/src/index.md` advertised `export_to_hdf5`, which is not exported.** Qualified as `Tortuosity.export_to_hdf5`, consistent with how `api.md` presents every other unexported name, rather than widening the public API on an already-breaking branch.
4. **The stubs' `kwargs...` catch-all would eventually tell a user to load a package they already have.** Removed; that case is now an honest `MethodError`.

**Nothing was left half-done.** Every item is terminal, the tree is green, and everything not finished is written up with its rationale in [Open items and follow-ups](#open-items-and-follow-ups) below — twelve items, each with what it is, why it was not done, and what doing it would take. Three of them (**A**, **C**, **F**) are the ones worth a decision; the rest are hygiene.

The staleness hazard the plan flags was handled the way the plan demanded, not assumed away: each edit bumped a probe function's return value and the session **asserted the new value before running the tests**, so a missed reload would have errored rather than silently reporting a pass. All three cycles verified clean. The harness driving this is `warm_loop.mjs`, a small stdio MCP client; it is measurement scaffolding and is not preserved in the repo.

## Open items and follow-ups

Everything this campaign knowingly left undone, in one place, so the next run does not have to reconstruct it from the prose above. Each entry says what it is, **why it was not done**, and what closing it would take. Nothing here blocks the merge; **A**, **C** and **F** are the three that want a decision rather than just effort.

**Priority order is A → C → F first, then the rest in any order.**

### A. No test reaches the optional-dependency error paths — *the one real coverage gap this campaign created*

The stubs in `src/weakdeps.jl` are the only user-visible behaviour change in S1, and **nothing asserts they work**. `test/runtests.jl` loads ImageFiltering and LsqFit for the whole session and `test_utils.jl` loads HDF5, so from the first line of the suite the extensions are active and the stubs are permanently shadowed. Julia cannot unload an extension, so no in-process test can ever reach them.

*Why not done:* it needs a subprocess-based test, which is a new testing pattern for this suite, and it landed as a review finding at the end of the run rather than during S1. Adding an untested new test harness under time pressure was the worse risk.

*What it takes:* one test that shells out to a bare environment and asserts the process fails with the expected message —

```julia
cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -e 'using Tortuosity; Tortuosity.Imaginator.blobs(shape=(4,4,4), porosity=0.5, blobiness=1)'`
```

— repeated for `export_to_hdf5` and `fit_effective_diffusivity`. Without it, a typo in a package name inside `_weakdep_error`, or a stub whose signature stops covering a call site, ships silently.

### B. `test/test_transient.jl:462` asserts exact bitwise float equality

`@test m_override == m_prob` compares two mathematically equivalent computations for **bit-identical** results. It passes only because `-O2` happens to generate identical code for both paths; at `-O1` they diverge by 1–2 ULP. Any change to Julia's optimiser, a dependency's inlining, or the surrounding code can break it with no bug in the package.

*Why not done:* changing it during this campaign would have been weakening a test to unlock a dev-time speed lever (L2), which was explicitly barred. Changed **on its own merits**, outside a campaign that benefits from it, it is a legitimate and probably correct fix.

*What it takes:* replace `==` with a tolerance-based comparison, then re-run at both `-O1` and `-O2`. Note that doing so would also unblock L2, worth ~23 s of cold suite — but L2 should be re-measured, not assumed, if that happens.

### C. Version not bumped and no release cut — *S1 is a breaking API change that is not yet released*

`Project.toml` is still `0.0.7`. S1 changed the shipped API: callers of `fit_effective_diffusivity`, `fit_voxel_diffusivity`, `Tortuosity.export_to_hdf5`, `Imaginator.blobs` and `Imaginator.apply_gaussian_blur` must now load the backing package first.

*Why not done:* cutting a release is a judgement call, not an engineering step, and was outside what this campaign was authorised to decide.

*What it takes:* bump to `0.0.8`, move the existing `## Unreleased` section of `CHANGELOG.md` under a `## 0.0.8` heading, and paste those notes into the `@JuliaRegistrator register` comment. Per `CLAUDE.md`, `0.0.7 → 0.0.8` changes the leftmost non-zero component, so AutoMerge treats it as **breaking** and will refuse to merge without the literal substring `breaking` or `changelog` in the notes. The `## Unreleased` section already contains `breaking`, so the notes are ready to use as written.

### D. AMDGPU and Metal extension edits are parse-checked only

The S2 guard and the later `_workload_enabled` rework were applied identically to `ext/TortuosityAMDGPUExt.jl` and `ext/TortuosityMetalExt.jl`, but neither package is installed and neither device exists on this machine, so neither extension was ever loaded.

*Why not done:* not possible here. The edits are character-identical to the CUDA one, which does precompile and whose GPU testsets pass, so the residual risk is confined to something the parser would already have caught.

*What it takes:* load them once on a Mac (Metal) and an AMD box (AMDGPU).

### E. The scratch-env dev-loop recipe is now broken

`2026-08-08-matrix-path-optimization.md:134` documents building a fast test environment with `Pkg.add(["Test","JLD2","HDF5","SparseArrays","LinearAlgebra","Statistics","Random"])`. Since S1, `test/runtests.jl` also needs **ImageFiltering** and **LsqFit**, so that env now fails with `ArgumentError: Package ImageFiltering not found in current path` before a single test runs. `test/Project.toml.bak` (tracked, and already unusable) is likewise more stale than before.

*Why not done:* it is in another plan's file, and this campaign's own deliverable supersedes it — the MCP session is strictly better than the scratch-env trick.

*What it takes:* either add the two packages to that recipe, or delete the recipe and point at the `julia-workflow` skill instead. The second is probably right.

### F. ImageMorphology is still a hard dependency — the unfinished quarter of S1

Worth ~60 packages in the closure and ~1.11 s of load. Blocked because `label_components` is reached from `_warn_nonpercolating` (`src/simulations.jl:63-68`), which runs **by default** from every `SteadyDiffusionProblem` on an image under 50 M voxels; `src/caverns.jl` uses it too; and the CPU `@compile_workload` builds a `SteadyDiffusionProblem`. The parent package therefore will not precompile without it.

*Why not done:* the only ways through are to drop the percolation check from the default path, or to degrade it silently when the package is absent. Both are user-visible behaviour changes well beyond "load a package first", and constraint 1 bars them without an explicit decision.

*What it takes:* **Amin's call on when that warning should fire.** If the check becomes opt-in — or emits a "install ImageMorphology for percolation checking" note instead of running — the dependency moves cleanly and S1's original target comes back into reach.

### G. `get_taufactor_conc` cannot run — pre-existing, unrelated

`src/utils.jl` defines `get_taufactor_conc`, which references `pyconvert`, `Py` and `pad`. None of the three is defined anywhere in the package.

*Why not done:* it predates this work entirely and touching it was out of scope. Recorded because someone should decide whether it is dead code to delete or a broken feature to repair.

### H. The LsqFit extension duplicates a solve already in the parent workload

`ext/TortuosityLsqFitExt.jl` builds a `TransientDiffusionProblem` and runs a ROCK4 solve that the parent workload in `src/Tortuosity.jl` already performs with identical arguments. Flagged by the reviewer as *suspected*, never measured.

*Why not done:* correctness is unaffected; it can only cost LsqFit users some extension-precompile time. Settling it needs a measurement that did not fit the run.

*What it takes:* time `TortuosityLsqFitExt` precompile with and without the duplicated block; drop it if the saving is real.

### I. CUDA version mismatch in the real test environment

The environment resolves **CUDA 13.3.0 while CUDA.jl was precompiled for 13.2.0**, and warns on every extension precompile. Confirmed present during Phase 0 — the Measurement appendix had flagged it as unverified.

*Why not done:* it is environment drift, not a package defect, and fixing it mid-campaign would have invalidated the baseline every other number is compared against.

*What it takes:* re-resolve or pin CUDA.jl so the two agree. Worth doing — this class of mismatch is what causes silent re-precompilation, and re-precompilation is the entire subject of this plan.

### J. The POSIX half of the MCP sync script has never executed

`.chezmoiscripts/run_onchange_sync-agent-mcp.sh.tmpl` is written and gated out by `.chezmoiignore` on Windows. It is honest code — `jq`-based, same marker and prune semantics, exits 0 with a message if `jq` is absent — but unverified code.

*Why not done:* no POSIX machine here.

*What it takes:* the first `chezmoi apply` on Linux or macOS is the test. Expect to fix something.

### K. julia-mcp is pinned to a commit, with no update path

`~/.agents/mcp/servers.json` pins `git+https://github.com/aplavin/julia-mcp@0000000045a2d345…`. This is deliberate — it makes `uvx` cache-hit and work offline, and stops an upstream change silently altering agent behaviour — but nothing will ever tell you a newer version exists.

*What it takes:* bump the sha in that one file and `chezmoi apply`; it fans out to all three agents. Re-verify with `claude mcp get julia` afterwards.

### L. `~/.agents/mcp/.synced.json` is unmanaged state inside a managed directory

The sync script's pruning ledger. `dot_agents/mcp/` is not `exact_`, so chezmoi leaves it alone and `chezmoi status` stays clean. Recorded so nobody "tidies" it into the chezmoi source, which would break pruning on other machines.

### Still deferred by design — X1 and X2

Not leftovers, decisions. Both are measured and documented in [Deferred](#deferred). Phase 1 removed the loop they would have optimised, which is exactly why they were deferred; revisit only if the full-suite gate becomes the bottleneck. X2's worker-scaling table is preserved there for whoever picks it up.
