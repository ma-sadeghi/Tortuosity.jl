# CLAUDE.md

## Releasing

Release flow: bump `version` in `Project.toml` on `main` (commit message: `chore: release vX.Y.Z`), then post a `@JuliaRegistrator register` comment on that commit. Registrator opens a PR against `JuliaRegistries/General`; the AutoMerge bot then decides whether to merge without human review.

**Release-notes keyword requirement.** AutoMerge classifies any version bump that changes the leftmost non-zero component as breaking, and refuses to auto-merge a breaking release whose notes don't contain the literal substring `breaking` or `changelog`. For this package (pre-1.0) every patch bump triggers the rule:

- `0.0.6 → 0.0.7` — `y` is the leftmost non-zero, so this **is** breaking by Julia SemVer.
- `0.5.2 → 0.5.3` (hypothetical, post-`0.1`) — patch under a non-zero `x`, **not** breaking.
- `1.2.3 → 1.2.4` (hypothetical, post-`1.0`) — patch, **not** breaking.

So the Registrator trigger comment must always include release notes with one of those keywords — even just `No breaking API changes.` Format:

```markdown
@JuliaRegistrator register

Release notes:

## Bug fixes
- ...

No breaking API changes.
```

If the General PR gets blocked because notes were missing, **re-post the same trigger comment with notes added on the same release commit**. Registrator updates the existing General PR in place; do not bump the version again.

After AutoMerge passes, the General PR enters a 3-day registry waiting period and then merges. TagBot then tags `vX.Y.Z` and creates the GitHub release without further action.
