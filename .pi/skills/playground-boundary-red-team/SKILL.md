---
name: playground-boundary-red-team
description: Explicitly runs a bounded, read-only localhost red-team of the browser Playground compiler and engine boundaries. Never invoke automatically.
disable-model-invocation: true
---

# Playground boundary red-team

This skill is an **explicit review gate**, not a general penetration-testing
capability. The reviewing child reports only and never modifies the repository.

## Authority contract

1. Work only in the current Bayescycle repository and read
   `playground/invariants.md` as the normative authority.
2. Use only synthetic canaries and temporary localhost origins. Never use real
   data, credentials, secrets, deployed Pages, third-party targets, destructive
   resource exhaustion, host escape, or browser/Pyodide vulnerability research.
3. Repository writes are prohibited. Browser profiles, screenshots, logs,
   payloads, transcripts, and reports must stay beneath `/tmp`.
4. Browser automation and bounded temporary local processes may use `bash`.
5. Always stop Rodney and every server/sink, remove disposable browser state,
   and report cleanup evidence.
6. Classify each probe as `RESISTED`, `ESCAPED`, `EXPECTED CAPABILITY`, or
   `UNTESTED`, citing the exact invariant clause. Current-compile mutation,
   public same-origin static fetches, and explicitly excluded persistence are
   not escapes by themselves.
7. Do not fix code. An alleged escape needs reproducible evidence for the
   parent agent to verify.

## Explicit invocation

From the repository root, after committing the skill and ensuring a clean
worktree:

```bash
.pi/skills/playground-boundary-red-team/scripts/run.sh \
  /tmp/playground-boundary-red-team-report.md
```

The runner starts a fresh `openai-codex/gpt-5.6-sol` Pi session at `xhigh`,
disables child skill discovery to prevent recursion, and grants only `read` and
`bash`. It rejects a dirty worktree and verifies that the child left the
worktree unchanged.

The child must execute every safe probe in
[`attack-catalog.md`](attack-catalog.md) and use
[`report-template.md`](report-template.md). A parent agent independently
verifies every alleged escape, freezes a RED regression, makes the narrow GREEN
fix, and reruns this skill in a fresh session until no in-scope escape remains.
