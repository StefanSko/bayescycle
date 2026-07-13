# Playground v0 — red→green execution plan

Companion to [`playground-plan.md`](playground-plan.md). Each work order has a
RED commit containing reviewed tests and importable stubs, followed by a GREEN
commit that does not modify the frozen tests. A discovered specification defect
stops the green pass and is corrected in a separate test-review commit.

All browser behavior is tested in Chromium through pytest + Playwright. The
single validation command is:

```bash
uv run --no-sync pytest playground/tests -q
```

Root guards remain:

```bash
uv run pytest tests -q
```

## Work orders

1. **R0 scaffold and executable guards.** No npm, no workspace membership, no
   CodeMirror, staged asset hashes, native document surfaces, no raw-IR tag
   inference in application code.
2. **R1 isolated compiler worker.** Ten corpus hashes, useful errors, explicit
   execution, timeout, stale-id rejection, source-only worker capability.
3. **R2 generic JSON codec.** Plain/canonical JSON parity with native
   `bayescycle.data`; no model dependency.
4. **R3 artifact runtime.** Wasm hash, typed errors, progress, valid multi-chain
   NDJSON, golden diagnose, named artifact results, independent follow-ups.
5. **R4 explicit state and observed vertical slice.** Revision-keyed immutable
   transitions; edit invalidation; native UI; compile, sample, diagnose,
   downloads.
6. **R5 pure plots.** Artifact fixtures to deterministic SVG; unavailable
   predictive operations do not invalidate posterior results.
7. **R6 explicit simulation.** Design/truth JSON, prior predictive, simulate,
   sample generated data, engine-owned recovery check. Ordered/vector/matrix
   examples are explicit documents, never inferred controls.
8. **R7 examples, shares, and hostile-source behavior.** No auto-compile,
   reviewed source, source-only compiler, timeout recovery, same-origin app
   requests.
9. **R8 publish.** Path-prefixed assets, version display, Pages workflow,
   releasing edge, clean fresh-browser dogfood walk.

At every green gate the running app is inspected with Rodney in addition to the
automated browser suite. Findings become a new RED test before a fix.
