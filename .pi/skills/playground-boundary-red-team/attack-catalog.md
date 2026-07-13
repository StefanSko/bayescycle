# Safe browser Playground attack catalog

Read `playground/invariants.md` completely before probing. Use a temporary
Playground origin and a second temporary localhost origin as the request sink.
Use a unique synthetic canary such as `PLAYGROUND_CANARY_<random>`. Keep every
artifact under `/tmp` and every resource/time probe bounded.

## Required probes

| ID | Probe | Required evidence | Authority |
|---|---|---|---|
| C1 | Observe compiler `postMessage` traffic while project documents contain the canary. | Compiler request key inventory and proof the canary/data/design/truth/settings/artifacts are absent. | Disposable compiler 3; runtime 3 |
| N1 | From model source attempt `fetch`, XHR, WebSocket, and EventSource to the second localhost origin. | Sink log and browser request evidence for each API. | Disposable compiler 5; trust model |
| M1 | Inject or emulate spoofed, malformed, unknown-id, unknown-type, and stale worker responses at the client boundary. | Bounded disposition and no stale state mutation. | Disposable compiler 10; runtime 7 |
| P1 | Mutate `bayeswire.ir`/serializer behavior in the current worker, then compile a golden corpus model in a fresh worker. | Current result classification plus exact later golden hash. | Disposable compiler 1, 4, 8 |
| L1 | Exercise declaration failure and source throw. | Visible bounded error and termination-before-settlement evidence. | Disposable compiler 2, 6 |
| L2 | Exercise worker close/error and startup failure using existing lifecycle harnesses or safe browser interception. | Termination and bounded recovery evidence. | Disposable compiler 2, 10 |
| L3 | Exercise timeout with bounded non-termination, then compile a golden model. | Timeout error and later exact golden hash. | Disposable compiler 2, 6 |
| L4 | Exercise cancellation through the compiler client harness. | Rejection after termination. | Disposable compiler 2 |
| H1 | Return exact known bytes with a false worker digest. | Browser-computed SHA-256 equals independent SHA-256 of bytes, not false digest. | Disposable compiler 9 |
| O1 | Return malformed byte type, malformed response shape, and oversized compiler output. | Bounded errors, worker disposal, no successful compile state. | Disposable compiler 6, 10 |
| E1 | Produce malformed JSON IR from a poisoned current compile and attempt inference. | Compiler terminates before engine handoff; authoritative engine error is visible; no posterior artifact. | Disposable compiler 11; runtime 3 |
| E2 | Record compiler-worker and engine-worker identities/lifetimes. | Separate workers and compiler termination before first engine request containing a project document. | Runtime 2, 3 |
| S1 | Load a shared project. | Source displayed with no automatic compilation or worker execution. | Trust model; browser constraints |
| R1 | Edit source/documents/settings during or after operations and replay stale completions where safely supported. | Descendant invalidation and no stale state mutation. | Runtime 7, 8 |

## Mandatory controls

- Also record public same-origin static fetch behavior and classify it against
  disposable compiler clause 5; it is not automatically an escape.
- Do not attempt persistent-storage erasure claims. If safely observed, classify
  behavior against the explicit non-guarantees rather than escalating it.
- Do not increase the compile timeout, allocate intentionally huge memory, fork
  uncontrolled processes, probe host files, or research browser/Pyodide flaws.
- Prefer existing browser tests and lifecycle fakes for malformed worker events;
  do not patch repository files.
- Capture desktop and narrow viewport screenshots only when useful for visible
  errors; screenshots stay in `/tmp`.

Every `ESCAPED` classification must include exact reproduction steps, expected
versus actual behavior, affected invariant clause, and evidence paths.
