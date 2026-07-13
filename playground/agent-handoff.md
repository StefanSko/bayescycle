# Agent handoff: browser Playground consolidation

Continue the browser Playground consolidation on the existing worktree and PR.

## Working state

Repository/worktree:

```text
/private/tmp/bayescycle-playground-runtime
```

Branch:

```text
playground-runtime-v0
```

PR:

```text
#65 — https://github.com/StefanSko/bayescycle/pull/65
```

The starting history must include:

```text
61e37a3 docs(playground): freeze browser runtime invariants
```

Do not start over from `main`, reset the branch, or reimplement the stable
Playground subsystems. Adapt PR #65 in place. First verify that the worktree is
clean, fetch current remote state, and confirm the branch/head. Mark PR #65 as
draft while consolidation is underway.

## Required reading

Before editing, read completely:

1. `AGENTS.md`
2. `playground/invariants.md`
3. `docs/playground-tdd-plan.md`
4. `docs/playground-plan.md`
5. `docs/playground-runtime-v0.md`
6. The current compiler, runtime, application, and browser-test harness code
7. Existing PR #65 review feedback, distinguishing comments on obsolete heads
   from comments on the current head

Treat `playground/invariants.md` as normative. Do not weaken or silently expand
its security contract. If implementation appears to require changing an
invariant or choosing an unspecified product policy, stop and ask before
freezing behavioral tests.

## Consolidation context

- The Playground feature set is already substantially complete.
- Stable engine, artifact, state, rendering, simulation, sharing, examples, and
  publication work should remain intact unless a frozen test proves a change is
  necessary.
- Current localized problems are:
  1. `main.mjs` imports `compile()` directly instead of using
     `runtime.compile()`.
  2. `compile/index.mjs` owns a reusable singleton worker and exposes a
     `reuseWorker` test shortcut.
  3. `compiler-worker.mjs` clones Bayeswire serializer internals and maintains a
     custom JSON encoder in an attempt to establish source-to-IR trust.
  4. pytest parameterization reruns complete JavaScript harness suites.
- Browser v0 does not provide source-to-IR attestation or a general Python
  sandbox.
- Compiler output is user-controlled.
- A smoke run of the independent `pi-review` skill already found one concrete
  UX defect outside the expected R10 boundary failures: after compilation
  succeeds, the live status becomes blank and users must infer success from a
  truncated hash and enabled button. Freeze that behavior with a RED regression
  before fixing it during R14.
- The meaningful guarantees are source-only compilation, one disposable worker
  per production compile, unconditional termination, external-origin isolation,
  trusted-client hashing of exact bytes, strict message boundaries, and
  separate engine workers.

## Execution discipline

Execute work orders R10 through R17 in `docs/playground-tdd-plan.md` using
strict RED → GREEN discipline:

- Each behavioral work order begins with a RED commit containing reviewed tests
  that fail for the intended reason.
- Then make a separate GREEN implementation commit without weakening or editing
  those frozen tests.
- If a test reveals a specification defect, stop the green pass and correct the
  specification/test in a separate reviewed commit.
- Keep edits narrow and preserve stable modules.
- Do not add Node/npm, TypeScript, CodeMirror, raw-IR semantic inference, or a
  generated design/truth form system.

## Required end state

1. The UI uses one runtime interface:

   ```text
   runtime.compile(source)
   runtime.run(request, onProgress)
   ```

2. `BrowserRuntime` owns compilation and execution adapters.

3. Every production compile creates one fresh worker and terminates it before
   its promise settles on success, declaration failure, malformed response,
   worker error, startup failure, timeout, or cancellation.

4. Compiler requests contain source and protocol metadata only. They never
   contain observed data, design, truth, settings, fits, posterior draws,
   diagnostics, or artifacts.

5. `compiler-worker.mjs` uses Bayeswire's ordinary canonical serializer. Remove
   cloned private function graphs, frozen Python globals, and the custom JSON
   serializer.

6. The worker returns exact IR bytes. The trusted browser client computes
   SHA-256 with Web Crypto and does not trust a worker-provided digest.

7. Compiler output is bounded and validated as untrusted input. Malformed IR
   produces a visible bounded error and cannot become a successful inference
   operation.

8. Poisoning one compiler worker may affect that compile, but cannot affect the
   next known corpus compile.

9. Remove production worker reuse. Rework tests so each named JavaScript case is
   executed once per validation invocation rather than rerunning entire suites.

10. Reconcile docs, UI security wording, runtime protocol, and PR description
    with `playground/invariants.md`.

## Rodney inspection

Periodically inspect the real app with Simon Willison's Rodney, especially
after behavior-affecting GREEN commits. Cover at least:

- observed example: load → compile → sample → diagnostics/plots;
- simulation example: compile → simulate → sample simulated data → recovery;
- shared project: display without automatic compilation;
- compile failure and malformed document UX;
- sampler-setting invalidation;
- a fresh compile after a poisoned or timed-out compile.

Turn any Rodney-discovered defect into a RED test before fixing it.

## Validation

Run targeted tests during development and these complete gates before final
review:

```bash
uv run --no-sync ruff format --check playground
uv run --no-sync ruff check playground
uv run --no-sync pytest playground/tests -q
uv run pytest tests -q
```

Also follow all repository/package validation instructions relevant to changed
files. Keep the branch pushed at meaningful GREEN checkpoints.

## Safe boundary red-team

At R14 completion, implement and explicitly invoke the project-local skill:

```text
.pi/skills/playground-boundary-red-team/SKILL.md
```

Give it `disable-model-invocation: true`; it must never run implicitly. The skill
must start a fresh `openai-codex/gpt-5.6-sol` Pi session at `xhigh`, disable
child skill loading to prevent recursion, and grant only `read` and `bash`.
Repository modification remains prohibited by its authority contract even
though browser automation and temporary local processes require `bash`.

Commit a reviewable attack catalog and report template beside the skill. Run
only against a temporary localhost Playground with synthetic canaries and a
second localhost origin as the external-request sink. Keep browser profiles,
screenshots, logs, payloads, and reports beneath `/tmp`. Never use real data,
secrets, credentials, deployed Pages, third-party targets, destructive resource
exhaustion, host escape techniques, or browser/Pyodide vulnerability research.
Always clean up Rodney, browser profiles, sinks, and servers, then prove the
worktree is unchanged.

The catalog must safely probe:

- source-only compiler messaging with a synthetic data canary;
- `fetch`, XHR, WebSocket, and EventSource toward the local external-origin
  sink;
- spoofed, malformed, unknown, and stale worker messages;
- module poisoning followed by a fresh golden corpus compile;
- declaration failure, throw, worker close/error, startup failure, timeout, and
  bounded non-termination recovery;
- ignored worker digests and trusted-client hashing of exact bytes;
- malformed and oversized compiler output;
- compiler termination before engine handoff and rejection of malformed IR.

Classify every result as `RESISTED`, `ESCAPED`, `EXPECTED CAPABILITY`, or
`UNTESTED`, citing the exact clause in `playground/invariants.md`. In particular,
current-compile mutation, public same-origin static fetches, and explicitly
excluded persistence behavior are not escapes by themselves. The child reports
only; it never fixes code. Verify each alleged escape, add a RED regression,
make the narrow GREEN fix, and rerun a fresh child audit until no in-scope
escape remains.

## Independent Pi review

At R15 completion, invoke the global `pi-review` skill:

```text
/Users/stefansko/.pi/agent/skills/pi-review/SKILL.md
```

Use `xhigh` because this is a cross-cutting invariant audit plus Rodney UI/UX
walkthrough. The review angle must include:

- `origin/main...HEAD` plus any uncommitted state;
- `playground/invariants.md` and `docs/playground-runtime-v0.md` as authority;
- compiler lifecycle, untrusted-artifact, state, and runtime boundaries;
- observed, simulation/recovery, and shared-project journeys;
- desktop and narrow/mobile screenshots plus accessibility evidence;
- no source-to-IR attestation or general Python sandbox as explicit non-goals.

The child reviewer is read-only. Verify every finding yourself. Address each
confirmed in-scope behavior with a RED/GREEN pair, then run a fresh independent
review rather than reusing the reviewer session. Continue until its verdict is
clean or clean with only explicit residual risks allowed by the invariants.

## Final PR preparation

At R16 completion:

- ensure the worktree is clean;
- push the final implementation;
- wait for complete CI;
- update the PR description with the explicit threat model and validation;
- mark PR #65 ready for review.

## Codex review loop

Then execute R17 using the `codex-pr-review-loop` skill:

```text
/Users/stefansko/.pi/agent/skills/codex-pr-review-loop/SKILL.md
```

Read that skill completely before using it. Explicitly run the Codex review
loop on PR #65 until the latest current-head review is clean.

For every actionable in-scope Codex finding:

1. add a RED regression test;
2. commit the RED test;
3. make the GREEN fix without weakening the test;
4. run targeted and required validation;
5. commit, push, and request another review.

Evaluate security suggestions against `playground/invariants.md`. Do not resume
the old serializer-freezing arms race. If a finding assumes source-to-IR
attestation or a general Python sandbox, explain why it is outside the declared
v0 contract and update wording if genuinely ambiguous rather than adding
speculative hardening. If Codex identifies an actual violation of an invariant,
address it.

Ignore stale feedback tied only to superseded commits. Continue until the
latest review of the latest pushed head has no actionable feedback and CI is
green.

## Final report

Report:

- the final commit;
- validation results;
- Rodney flows completed;
- safe boundary red-team verdict and report path;
- independent Pi review verdict and report path;
- PR URL;
- clean current-head Codex review evidence;
- any explicitly declined out-of-scope suggestions and their invariant-based
  rationale.
