# Browser playground invariants

This file is the normative engineering boundary for the browser playground.
The browser is the first runtime; a future local `bayescycle ui` must implement
the same UI-facing runtime and artifact contracts. If another playground
document conflicts with this file, resolve the conflict before changing code.

## Trust model

- Model source is arbitrary, user-controlled Python. Shared source is displayed
  without execution and runs only after an explicit compile action.
- Compiler output is user-controlled input. It is not an attestation that a
  human correctly understood the displayed source.
- The checked-in application, pinned Pyodide and Bayesite assets, main-page
  control plane, and runtime adapters are trusted release artifacts.
- The browser playground has no backend and holds no server-side secrets. Its
  same-origin files are public static assets.
- The playground does not describe an import policy, API denylist, Pyodide, or
  a Web Worker as a general Python sandbox.

## Disposable compiler boundary

1. One production compile attempt owns one fresh compiler worker and one fresh
   Pyodide runtime. Production code has no worker pooling or reuse option.
2. The worker is terminated before the compile promise settles on every exit:
   success, declaration failure, malformed response, worker error, startup
   failure, timeout, or cancellation.
3. A compile request contains only protocol metadata and model source. The
   compiler worker never receives observed data, design data, parameter truth,
   sampler settings, fits, posterior draws, diagnostics, or run artifacts.
4. Pyodide and Bayeswire are loaded from pinned same-origin release assets into
   worker-local memory. Later compiles do not load Python modules or files from
   state produced by an earlier compiler worker.
5. External-origin requests from compiler source are blocked by browser policy;
   worker API restrictions are defense in depth. Same-origin compiler fetches
   can reach only the playground's public static deployment.
6. Compilation has explicit time and output-size bounds. Exceeding either
   bound terminates the worker and produces a visible error.
7. The compiler uses Bayeswire's ordinary canonical serializer. Application
   code does not clone Bayeswire private functions, freeze Python module
   graphs, or maintain a second canonical JSON serializer.
8. Model source may alter its current Python interpreter and may therefore
   cause its own compile to fail or emit unexpected bytes. Such mutation must
   not affect a later compile.
9. The trusted browser client computes SHA-256 over the exact bytes received
   from the worker. A worker-supplied digest is neither required nor trusted.
10. Compiler responses are validated for request identity, message shape,
    byte type, and size before use. Malformed or stale responses are ignored or
    surfaced as bounded errors.
11. Exact IR bytes are never silently reserialized. The authoritative engine
    decoder validates model IR before inference; malformed IR becomes a visible
    operation error.

Terminating a worker guarantees disposal of its JavaScript realm, Pyodide WASM
memory, Python heap, `sys.modules`, and in-memory filesystem. It does not claim
to erase every browser-origin side effect an arbitrary program could attempt.
The compiler implementation itself must not use origin-persistent storage as an
input to a later compilation.

## Runtime and artifact boundary

1. The UI talks to one runtime interface with both `compile(source)` and
   `run(request, onProgress)` operations. Application code does not import the
   compiler or Bayesite executor directly.
2. `BrowserRuntime` implements that interface with the disposable compiler and
   separate Bayesite WASM workers. A future `LocalRuntime` may implement it over
   a loopback Bayescycle service without changing UI workflow semantics.
3. Compilation finishes and its worker is terminated before any project
   document is sent to an engine worker.
4. Runtime results are named, immutable artifacts using the Bayescycle
   run-directory vocabulary, including `model.ir.json`, `data.json`,
   `posterior.ndjson`, and `diagnostics.json`.
5. The UI may parse artifacts through pure renderers. It does not inspect raw
   IR node tags to infer required data, defaults, dimensions, parameter truth,
   partial-observation behavior, or engine capabilities.
6. Observed data, simulation design, and parameter truth remain explicit JSON
   documents. The browser does not synthesize semantic forms from model IR.
7. Source, document, and settings revisions invalidate descendant artifacts.
   Unknown or stale asynchronous completions cannot mutate current state.
8. A failed follow-up operation does not erase artifacts from an earlier
   successful run.

## Browser application constraints

- The playground remains a static site with no Node/npm build toolchain.
- Native HTML controls are preferred; JavaScript exists only for browser APIs,
  workers, artifact handling, state transitions, and rendering.
- Vendored runtime assets are pinned and hash-checked at staging or test time.
- Shared projects never compile automatically.
- User-visible failures are bounded and actionable; malformed JSON, invalid
  settings, compile failures, and engine failures do not disappear into the
  developer console.

## Explicit non-guarantees

Browser v0 does not guarantee that:

- arbitrary Python is safe or side-effect free;
- compiler output semantically matches a reader's interpretation of source;
- model source cannot alter its own compile result;
- worker termination erases browser storage explicitly written by hostile code;
- a tab survives deliberate memory exhaustion or a browser/Pyodide exploit.

A future requirement for source-to-IR attestation or stronger hostile-code
containment requires a different architecture, such as a restricted declaration
language or an isolated producer that emits a narrow validated document for a
separate trusted serializer.

## Executable evidence

Tests must freeze these observable claims before implementation changes:

- the UI reaches compilation only through the runtime interface;
- production compilation creates and unconditionally terminates one worker;
- compiler requests contain source and protocol metadata only;
- poisoning one worker cannot change a known corpus compile in the next worker;
- the client, not the worker, hashes exact returned bytes;
- stale, malformed, oversized, failed, and timed-out responses are bounded;
- external-origin compiler requests are blocked in a real browser;
- all corpus models still match native canonical IR bytes and hashes;
- malformed compiler output cannot become a successful inference operation;
- project edits and stale completions obey the revisioned state contract.

Behavioral changes follow strict RED then GREEN commits. A RED commit freezes a
reviewed observable test. Its GREEN commit changes implementation without
weakening that test. Security comments outside this stated threat model change
this file first; they are not addressed by silently expanding implementation
claims.
