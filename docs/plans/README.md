# Plans

Design and execution plans for substantial pieces of work on Tortuosity.jl. Maintainer documentation, not user-facing — the published site is built from `docs/src/` only, so nothing here appears in it.

Plans are kept after they finish. A completed plan is the record of what was tried, what was measured, what was rejected and why — which is what stops the next person re-investigating a dead end. **Do not delete a plan when it completes; set its status to `complete` and leave it.**

## Naming

`YYYY-MM-DD-short-descriptive-name.md`

The date is when the plan was **created**, not when it finished — it stays fixed for the life of the document so links and references do not rot. The name is kebab-case, three or four words, naming the *goal* rather than the mechanism (`matrix-path-optimization`, not `fix-oom-bug`).

## Preamble

Every plan starts with the block below, before anything else. Keep the field order; it is what makes a folder of plans skimmable.

```markdown
---
title: <human-readable title>
created: YYYY-MM-DD
updated: YYYY-MM-DD
status: draft | active | complete | superseded | abandoned
outcome: <one line — only once status is terminal. Omit while draft/active.>
branch: <git branch the work lives on, or "-" if none>
supersedes: <plan filename, or "-">
superseded-by: <plan filename, or "-">
related: <plan filenames, or "-">
---

> **Status: <status>.** <Two or three sentences: what this plan is for, and — if it is
> finished — what actually happened, with the headline numbers. Someone who reads only
> this box should know whether to keep reading.>
```

### `status` values

| value | meaning |
| --- | --- |
| `draft` | being written; not yet agreed or started |
| `active` | work is in progress against it |
| `complete` | the work finished; the document records the outcome |
| `superseded` | replaced by a later plan — set `superseded-by` |
| `abandoned` | deliberately stopped; say why in `outcome` |

### Rules

- **`updated` changes whenever the body changes.** A plan whose numbers are stale is worse than no plan, because it is quoted with confidence.
- **Record measurements as measured and estimates as estimates.** If a plan carries an estimate that was later measured, replace it and mark it measured. If the estimate was badly wrong, say so — that is information about where the estimating was weak.
- **Log rejections with their reasoning.** A rejected item with a reason is as valuable as a completed one; a silently dropped item costs the next person a re-investigation.
- **Terminal states are terminal.** Every work item in a completed plan should be `done`, `rejected`, `blocked` or `reverted` — never left `pending`.
