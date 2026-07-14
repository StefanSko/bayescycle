# Playground runtime protocol v0

This provisional protocol separates the shared playground UI from its execution
adapter. Browser v0 implements it with workers; a future local Bayescycle server
may implement the same observable behavior.

## Values

All messages are structured-cloneable objects. Every request carries a unique
`id`. Responses with an unknown or stale id are ignored. Boundary parsers reject
unknown message types and malformed fields.

## Compile

The UI-facing operation is `runtime.compile(source)`. In `BrowserRuntime`, one
call creates one fresh compiler worker and sends only:

```text
WorkerCompileRequest  {type: "compile", protocol, id, source}
WorkerCompileSuccess  {type: "compiled", id, irBytes}
WorkerCompileFailure  {type: "compile-error", id, exceptionType, message, traceback}
```

The worker returns the exact bytes produced by Bayeswire's ordinary canonical
serializer; those bytes are untrusted compiler output. After validating the
response, the browser client terminates the compiler worker and computes
SHA-256 over the exact received bytes with Web Crypto. The UI-facing result is:

```text
CompileSuccess  {ok: true, irBytes, irHash, executionContext: "worker"}
CompileFailure  {ok: false, exceptionType, message, traceback, executionContext: "worker"}
```

A worker-provided digest is not required or trusted. Compilation executes only
after an explicit user action. Timeout, startup failure, worker error,
malformed response, oversized output, and cancellation all terminate that
attempt's worker before the promise settles.

## Run

All requests and plans are structured-cloneable immutable values. Binary
artifacts are defensively copied at construction and runtime boundaries.

```text
GenerateRequest {
  type: "run",
  id,
  operation: "generate",
  plan: Draw(JointPredict(ParameterSource, OutcomesOf))
}
ConditionRequest {
  type: "run",
  id,
  operation: "condition",
  modelIr,
  data,
  settings,
  pairedParameters?
}
RunProgress {type: "progress", id, chainId, retainedDraws, divergences}
RunSuccess  {type: "artifacts", id, artifacts: [{name, mediaType, bytes}]}
RunFailure  {type: "run-error", id, error: {kind, message}}
```

Generation plans, source variants, byte identity, count/seed bounds, redraw
semantics, and paired output are defined by
[`generation-plan-v0.md`](generation-plan-v0.md). `condition` performs
inference over one canonical dataset and may produce posterior, diagnostics,
and paired recovery artifacts. Conditioning remains separate from generation.

The application uses these two workflow operations. BrowserRuntime keeps
Bayesite command names private. A temporary compatibility lowering may run only
when it exactly implements the requested redraw law; unsupported source/count
combinations fail before dispatch until the pinned Wasm exposes native bounded
generation.

## Artifact names

Runtime results use the run-directory vocabulary:

- `model.ir.json`
- `data.json`
- `posterior.ndjson`
- `diagnostics.json`
- `prior_predictive.ndjson`
- `posterior_predictive.ndjson`
- `simulated_data.json`
- `recovery_check.json`
- `generated_datasets.ndjson`

The three source-specific generation artifacts remain outputs of their legacy
operations with existing meanings. Unified `generate` returns only
`generated_datasets.ndjson`; it does not project a legacy artifact. New
selection and recovery behavior uses the paired artifact.

A runtime may return fewer artifacts when an operation does not produce them.
An unsupported follow-up is a failure of that operation only and never removes
artifacts returned by an earlier successful operation.

## UI restriction

The UI may parse artifacts for display through pure renderers. It must not
inspect raw IR implementation tags to derive required inputs, dimensions,
default values, parameter truth shapes, partial-observation behavior, or engine
capabilities.
