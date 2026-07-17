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
3. A compile request contains only protocol metadata and model source. A
   composed-prior scenario request may contain the current main model source
   and one prior-only model snippet. The compiler worker never receives
   observed data, design data, parameter truth, sampler settings, fits,
   posterior draws, diagnostics, or run artifacts.
4. Pyodide and Bayeswire are loaded from pinned same-origin release assets into
   worker-local memory. Later compiles do not load Python modules or files from
   state produced by an earlier compiler worker.
5. External-origin requests from compiler source are blocked by browser policy;
   worker API restrictions are defense in depth. Same-origin compiler fetches
   can reach only the playground's public static deployment.
6. Compilation has explicit time, source-size, and output-size bounds. Model
   source is rejected above 1 MiB of UTF-8 before a worker is created. For a
   composed-prior scenario, the prior-only snippet and the main source are each
   bounded and together must fit the same 1 MiB UTF-8 budget. Exceeding any
   compile bound produces a visible error.
7. The compiler uses Bayeswire's ordinary canonical serializer. Application
   code does not clone Bayeswire private functions, freeze Python module
   graphs, or maintain a second canonical JSON serializer.
8. Model source may alter its current Python interpreter and may therefore
   cause its own compile to fail or emit unexpected bytes. Such mutation must
   not affect a later compile.
9. The trusted browser client computes SHA-256 over the exact bytes received
   from the worker. For composed-prior scenarios it independently hashes both
   the target and composed IR, and generation proceeds only when the
   scenario-worker target hash equals the current compiled-model hash. A
   worker-supplied digest is neither required nor trusted.
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

1. The UI talks to one runtime interface with `compile(source)`,
   `compileScenario(source, priorSource)`, and `run(request, onProgress)`
   operations. Application code does not import the compiler or Bayesite
   executor directly.
2. `BrowserRuntime` implements that interface with the disposable compiler and
   separate Bayesite WASM workers. A future `LocalRuntime` may implement it over
   a loopback Bayescycle service without changing UI workflow semantics.
3. Compilation finishes and its worker is terminated before any project
   document is sent to an engine worker.
4. Runtime results are named, immutable artifacts using the Bayescycle
   run-directory vocabulary, including `model.ir.json`, `data.json`,
   `posterior.ndjson`, `diagnostics.json`, and the paired
   `generated_datasets.ndjson` artifact defined by
   [`spec/generated-datasets-v0.md`](../spec/generated-datasets-v0.md).
5. Generation reaches the runtime through one immutable exact-key `generate`
   plan. Application/UI modules do not select fixed, prior-predictive, or
   posterior-predictive engine commands. `BrowserRuntime` may use a private
   compatibility command only when it implements the exact requested redraw
   law; otherwise it fails before dispatch until the native bounded generation
   operation is staged.
6. Generated records retain one natural-scale parameter document paired with
   one complete canonical dataset per draw. A requested count redraws the
   parameter source per dataset; fixed parameters repeat only because their
   source is a point mass. An alternative prior is composed with the main model
   in a fresh compiler worker and generation draws from that ordinary closed
   model, while conditioning continues to use the original compiled model.
   Prior-only snippets cannot add or change data or observed slots, and their
   declared parameter dimensions and coordinates must pass Bayeswire's exact
   compatibility checks. Generation never launches one Wasm worker per draw.
7. Conditioning is a separate runtime transition over a selected canonical
   dataset. Recovery truth from a composed model is projected to the original
   model's parameter names without changing the retained generated pair; an
   empty projection skips recovery with a visible notice. A posterior source is
   available only while its exact model/data/fit lineage survives.
8. The UI may parse artifacts through pure renderers. It does not inspect raw
   IR node tags to infer required data, defaults, dimensions, parameter truth,
   partial-observation behavior, or engine capabilities.
9. Observed data, generation design, and fixed parameter values remain explicit
   canonical JSON documents. The browser does not synthesize semantic forms
   from model IR.
10. Model, observed-data, design, fixed-value, prior-snippet, parameter-source,
    generation-setting, inference-setting, selected-draw, and fit revisions
    invalidate only their descendants. Unknown or stale asynchronous
    completions cannot mutate current state.
11. Each compile, generation, and fit attempt owns one cancellation signal.
    One user cancellation ends every concurrently current attempt, and user
    cancellation or reducer-decided invalidation terminates every engine worker
    still owned by those attempts. The first failed sampling chain terminates
    its in-flight siblings; cancellation never waits for sibling chains to
    finish naturally or degrades into a partial-success warning.
12. A failed or cancelled follow-up operation, including recompilation, does
    not erase artifacts or fit lineage from an earlier successful run. A
    successful compile with changed model bytes still invalidates prior run
    descendants.
13. Capability failures are visible and bounded. The frontend never drops
    score factors or claims that every scoreable model is ancestrally
    sampleable.
14. Each sampling-chain posterior response and the merged `posterior.ndjson`
    are bounded to 64 MiB. An oversized chain is rejected in its engine worker
    before transfer and again at the worker-message boundary; aggregate chain
    bytes are checked before main-thread decode or merge. Rejection fails the
    fit without publishing partial posterior, diagnostics, or recovery
    artifacts. This ceiling is scoped to the production `WorkerEngine` sample
    path; aggregate handling by the test-only `InProcessEngine` and generation
    output bounding remain follow-up work under the engine and artifact limits.

## Browser application constraints

- The playground remains a static site with no Node/npm build toolchain.
- Native HTML controls are preferred; JavaScript exists only for browser APIs,
  workers, artifact handling, state transitions, and rendering.
- Empty design documents in fresh and bundled projects derive defaults only
  from the validated compiler schema. Non-empty locally authored documents,
  shared authoring state, and carried share documents remain authoritative and
  are never replaced by a newly synthesized default.
- Vendored runtime assets are pinned and hash-checked at staging or test time.
- Shared projects never compile automatically.
- User-visible failures are bounded and actionable; malformed JSON, invalid
  settings, compile failures, and engine failures do not disappear into the
  developer console. The shared Cancel control is visible for compilation as
  well as generation and fitting.
- Share fragments are rejected above 65,536 compressed characters and are
  decompressed as a stream with a 1 MiB output ceiling.
- User data documents are rejected above 4 MiB of UTF-8 before JSON parsing.
  Normalization accepts at most 32 nesting levels and 100,000 scalar values,
  and flattening is linear in the accepted document size.
- Compiler schemas contain at most 1,024 parameter, data, and observed entries
  in total. Names, priors, constraints, and shape dimension names contain at
  most 512 characters.
  The UI renders parameter forms for at most 200 parameters and design cards
  for at most 50 data slots; larger accepted schemas stay in JSON mode.
- Dashboard SVGs render only for posteriors with at most 200 expanded parameter
  components. Design-form previews share a 100,000-scalar aggregate budget per
  render pass; later previews are omitted once that budget is exhausted.
- Sampling limits are enforced at the runtime boundary before worker dispatch,
  not only by the UI: 1–8 chains, 0–100,000 warmup iterations, 4–100,000 draws,
  and tree depth 1–20.

## Explicit non-guarantees

Browser v0 does not guarantee that:

- arbitrary Python is safe or side-effect free;
- compiler output semantically matches a reader's interpretation of source;
- model source cannot alter its own compile result;
- a prior-only snippet — trusted arbitrary Python at the same level as
  model source — cannot alter the target it composes with; a composed
  generation model is still hashed over its exact bytes, and conditioning
  always fits the original compiled model, never the composed one;
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
- stale, malformed, oversized, failed, cancelled, and timed-out responses are bounded;
- aborting or invalidating a run terminates all owned workers, and stale
  post-abort completions remain inert;
- compressed shares, decompressed shares, model source, data documents, and
  compiler schemas enforce their stated bounds before unbounded work;
- external-origin compiler requests are blocked in a real browser;
- all corpus models still match native canonical IR bytes and hashes;
- malformed compiler output cannot become a successful inference operation;
- oversized chain and aggregate posterior output fails before main-thread
  decode, merge, follow-up diagnostics, or artifact publication; the worker
  boundary accepts an exactly 64 MiB chain, and a fit above the separate 8 MiB
  generation-input limit remains valid through posterior publication;
- project edits and stale completions obey the revisioned state contract;
- empty schema-derived design defaults, non-empty pre-compile documents, source
  schema changes, and exact shared-authoring restores are separately covered
  as observable browser behavior;
- fixed, model-prior (including a separately authored composed prior), and
  posterior generation share one exact plan and redraw law while preserving
  parameter/dataset pairs;
- scenario target divergence, prior-only data extension, incompatible authored
  dimensions, and stale scenario completion fail before engine dispatch;
- composed-pair recovery uses only original-model parameter truth while the
  generated pair remains byte-exact;
- malformed, oversized, unsupported, and lineage-mismatched generation fails
  without deleting earlier artifacts;
- selection and setting edits invalidate only their documented descendants;
- dashboard component limits, linear recovery-truth matching, and aggregate
  design-preview limits bound rendering work before SVG or preview expansion;
- application/UI source contains no private generation command names.

Behavioral changes follow strict RED then GREEN commits. A RED commit freezes a
reviewed observable test. Its GREEN commit changes implementation without
weakening that test. Security comments outside this stated threat model change
this file first; they are not addressed by silently expanding implementation
claims.
