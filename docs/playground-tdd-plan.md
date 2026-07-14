# Playground v0 — red→green execution plan

Companion to [`playground-plan.md`](playground-plan.md) and the normative
[`playground/invariants.md`](../playground/invariants.md). Each work order has a
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

## Functional generation and conditioning

The next work orders migrate the source-specific controls onto the contract in
[`generation-plan-v0.md`](generation-plan-v0.md):

18. **B0 contract reconciliation.** Freeze the source variants, redraw law,
    conditioning boundary, capability failures, paired artifact, provenance,
    scoped invalidation, compatibility behavior, and non-goals before behavior
    code. Obtain an independent invariant review.
19. **B1 paired artifact.** RED shared Python/JavaScript golden fixture,
    malformed/oversized streams, exact order/count, finite values, selection,
    and byte-preserving downloads. GREEN pure bounded codecs.
20. **B2 generation plans.** RED exact immutable variants, bounds, unknown-key
    rejection, Python/JavaScript parity, convenience equivalence, identities,
    and invalidation keys. GREEN typed Python values and pure browser values.
21. **B3 runtime lowering.** RED one recording-executor `generate` operation,
    byte preservation, visible capability failures, stale/failed preservation,
    and no private generation verbs in application code. GREEN private adapter
    lowering and paired-result normalization.
22. **B4 native generation.** In a fresh Bayesite branch, RED one bounded pure
    request with deterministic per-dataset redraw, fixed/posterior validation,
    paired output, capability rejection, and CLI/Wasm parity. GREEN one core
    operation; stage the reviewed Wasm release rather than worker fan-out.
23. **B5 scoped reducer migration.** RED independent revisions and dependency
    keys for model, documents, source, generation, selection, inference, and
    fits. GREEN immutable reducer transitions and stale completion rejection.
24. **B6 shared generation UI.** RED accessible source radios, conditional fixed
    values, compatible-posterior availability, common count/seed, paired
    download, bounded errors, and no automatic execution. GREEN native controls
    that call the generation convenience operation only.
25. **B7 selection and conditioning.** RED a `1..N` selector, canonical dataset
    and paired-parameter previews, fit-selected action, generating/fit-model
    lineage, recovery, and selection-scoped invalidation. GREEN immutable pair
    selection and runtime-boundary dataset materialization.
26. **B8 compatibility/publication.** Preserve legacy artifact meanings and
    shared projects; update examples, version/deployment checks, release docs,
    changelog, and fresh-browser publication smoke.

Each behavior work order is a reviewed RED commit followed by a minimum GREEN
commit that does not alter the frozen RED tests. Specification defects receive
their own correction commit.

## Disposable-compiler consolidation

The initial vertical slice is complete. The following work orders consolidate
its security boundary without claiming source-to-IR attestation. They adapt the
existing implementation; they do not rebuild the playground.

10. **R9 freeze the boundary.** Add `playground/invariants.md`, reconcile the
    runtime protocol and PR wording with its trust model, and stop automatic
    review fixes that would expand the security claim without first changing
    the invariant document.
11. **R10 runtime-owned compilation.** RED architecture and lifecycle tests
    require the UI to call `runtime.compile(source)`, prohibit direct compiler
    and engine imports from application code, require source-only requests, and
    cover worker termination after startup failure, success, compile failure,
    malformed response, worker error, timeout, and cancellation. GREEN moves
    compilation behind `BrowserRuntime`, injects test doubles at its narrow
    boundary, and removes production worker reuse.
12. **R11 untrusted compile artifact.** RED tests require a later golden corpus
    compile to survive earlier module poisoning, require the client to ignore a
    false worker digest and hash exact received bytes, and bound malformed and
    oversized output. GREEN replaces serializer cloning and the custom JSON
    encoder with Bayeswire's ordinary canonical serializer, computes SHA-256 in
    the trusted client, and treats returned IR as user-controlled bytes.
13. **R12 engine handoff.** RED browser tests prove that no project document is
    sent to the compiler, compilation terminates before engine work begins,
    external-origin requests remain blocked, and malformed IR cannot become a
    successful inference operation. GREEN makes only the minimum protocol and
    error-surface changes needed at those boundaries; it does not introduce an
    import allowlist or new sandbox claim.
14. **R13 execute each test once.** RED harness accounting proves each named
    JavaScript case runs once per validation invocation. GREEN removes the
    pytest pattern that reruns a complete JavaScript suite for every case and
    replaces the production `reuseWorker` test shortcut with test isolation
    that does not weaken lifecycle invariants.
15. **R14 reconcile and dogfood.** Update `playground-plan.md`,
    `playground-runtime-v0.md`, UI security copy, and the PR description. Run
    format/lint, all 60+ browser checks, root guards, fresh-browser observed and
    simulation Rodney walks, and publication-path checks. Add a RED regression
    for the independently observed missing compile-success announcement before
    fixing it. Any other behavioral defect gets its own RED/GREEN pair.
16. **R15 safe boundary red-team.** Add an explicitly invoked project skill at
    `.pi/skills/playground-boundary-red-team/` with a reviewed attack catalog,
    report contract, and runner for a fresh `gpt-5.6-sol` `xhigh` Pi session.
    It may use only synthetic canaries, temporary localhost origins, disposable
    browser state, bounded resource probes, and `/tmp` evidence; it never edits
    the repository, targets a deployed or third-party service, uses secrets, or
    attempts browser/Pyodide vulnerabilities. Exercise source-only messaging,
    external-origin denial, stale/spoofed responses, worker poisoning and
    recovery, every lifecycle exit, client-side hashing, output bounds,
    malformed IR, and compiler/engine separation. Classify each probe as
    `RESISTED`, `ESCAPED`, `EXPECTED CAPABILITY`, or `UNTESTED` against explicit
    invariant clauses. Verify every claimed escape, freeze it with a RED test,
    make the narrow GREEN fix, and rerun a fresh audit until no in-scope escape
    remains. The skill is hidden from automatic model invocation.
17. **R16 independent Pi review.** Invoke the global `pi-review` skill at
    `xhigh` with `origin/main...HEAD`, `playground/invariants.md`, the runtime
    protocol, and the observed/simulation/share journeys as its review angle.
    The fresh read-only reviewer must perform both an invariant ledger and a
    screenshot/accessibility-backed Rodney walkthrough. Verify its evidence;
    address confirmed in-scope defects with RED/GREEN pairs, then rerun a fresh
    independent review until it is clean or reports only explicit residual
    risks accepted by the invariant contract.
18. **R17 bounded Codex review.** Request one final Codex review against the
    explicit invariant document. Fix in-scope correctness defects with
    RED/GREEN pairs; document or decline suggestions that assume a stronger
    non-goal. Merge only after the final head has complete CI and no unresolved
    in-scope finding.
