# Playground v0 — browser runtime now, local runtime later

**Revised 2026-07-13.** This supersedes the original carve plan. PR #64 proved
that bayeswire models and the Bayesite wasm engine can run entirely in a real
browser, but its review loop also showed that deriving workflow semantics from
raw IR in frontend code is not a maintainable boundary.

## Product direction

The long-term product is a local, out-of-process Bayescycle UI backed by native
Python, durable run directories, the Bayesite executable, and the existing
`uvx` visualization boundary. The first public release is useful earlier as a
fully client-side playground and remains a supported execution adapter rather
than throwaway code.

```text
                         shared UI
                            |
                   PlaygroundRuntime v0
                    /                 \
       BrowserRuntime                 LocalRuntime (future)
  Pyodide + Bayesite wasm       localhost bayescycle
  isolated Web Workers          run/ + external commands
```

The shared boundary is the artifact contract. The browser is an alternative
execution adapter, not an alternative implementation of Bayescycle workflow
semantics.

## Public v0 promise

A user can select or write a model, provide explicit JSON data, compile it to
canonical bayeswire IR in Pyodide, sample it with one universal Bayesite wasm
engine, inspect diagnostics and plots, and download standard Bayescycle
artifacts. Runtime requests are same-origin and no compilation service exists.

This differs structurally from Stan Playground: no server compiles a model and
no per-model wasm artifact is built or downloaded.

## Runtime boundary

The UI uses only the protocol in [`playground-runtime-v0.md`](playground-runtime-v0.md):

- `compile(source)` returns canonical IR bytes and their received-byte hash;
- `run(operation, artifacts, settings)` emits progress and named artifacts;
- results use run-directory filenames such as `model.ir.json`, `data.json`,
  `posterior.ndjson`, and `diagnostics.json`.

The main page does not walk raw IR tags, derive dimensions, merge backend
streams, or compute recovery facts.

## Browser v0 scope

Included:

- native model, observed-data, design-data, and truth textareas;
- bundled examples whose documents are valid immediately;
- explicit compile action and IR hash;
- plain JSON normalization and strict `bayescycle.data.json.v1` input;
- posterior sampling with per-chain progress;
- diagnostics and pure artifact-to-SVG plots;
- one functional dataset-generation workflow with fixed, model-prior, and
  posterior parameter sources, uniform count/seed semantics, paired
  parameter/dataset artifacts, generated-dataset selection, conditioning, and
  engine-owned recovery checks;
- byte-preserving artifact downloads;
- reviewed URL-fragment shares which never compile automatically;
- real-browser tests and same-origin route assertions.

Deferred:

- generated design/truth forms and inferred defaults;
- CSV mapping and standardization;
- CodeMirror;
- partially observed predictive operations the pinned engine does not support;
- persistence, service worker/offline cache, Gist/embed, and analysis scripts;
- the local runtime implementation.

## Explicit documents, not inferred forms

Simulation remains first-class, but the project carries exact design and truth
documents. Examples prefill friendly plain JSON:

```json
{"x": [-1.0, -0.5, 0.0, 0.5, 1.0]}
```

```json
{"alpha": 0.5, "beta": 1.2, "sigma": 0.4}
```

The generic data codec normalizes these documents. Frontend code never guesses
whether a scalar is a count, whether a vector is a scale, how an ordered truth
should expand, or which implementation node represents partial observation.
Generated controls may return only after a normative producer exposes the
metadata they require.

## Generation and conditioning

[`generation-plan-v0.md`](generation-plan-v0.md) defines the shared immutable
plan and paired-artifact contract. The application asks the runtime to generate
datasets; it does not choose Bayesite commands. Fixed values, the authored
prior of one closed model, and a compatible posterior fit differ only by their
explicit parameter-source variant. Every generated dataset remains paired with
the natural-scale parameters that produced it.

Conditioning is a separate transition over observed data or one selected
complete generated dataset. A future model composed by another authoring API is
just another closed IR document and requires no frontend composition logic.

## State discipline

Project state is represented by explicit immutable transitions. Model,
observed-data, design, fixed-value, generation-setting, inference-setting,
selected-draw, and fit revisions identify asynchronous dependencies. An edit
invalidates only descendant artifacts, stale worker messages are ignored, and a
failed follow-up cannot erase an earlier successful artifact.

## Security and trust

[`playground/invariants.md`](../playground/invariants.md) is the normative
browser trust boundary and lists its explicit non-guarantees.

Each explicit compile creates a fresh Pyodide compiler worker. That worker is
sent protocol metadata and source, but never data, truth, sampler settings,
posterior draws, or other project artifacts. It is unconditionally terminated
before the compile promise settles on success, declaration failure, malformed
response, worker error, startup failure, timeout, or cancellation. A later
compile cannot inherit the prior worker's Python modules, heap, or filesystem.

The worker uses Bayeswire's ordinary canonical serializer. Compiler output is
user-controlled: source may alter its own interpreter and therefore its own
compile result. The trusted browser client validates and bounds the exact
returned bytes, ignores any worker digest, computes SHA-256 with Web Crypto,
and sends the unchanged bytes to the separate engine boundary, whose decoder
rejects malformed IR.

External-origin requests from compiler source are blocked by browser policy;
defense-in-depth worker API restrictions are not described as a Python sandbox
or import policy. Same-origin assets are public static files. Shared source is
displayed first and executes only after an explicit compile action. Browser v0
does not provide source-to-IR attestation, a general hostile-Python sandbox, or
guaranteed erasure of browser storage explicitly written by hostile source.

## Testing and publication

Execution follows [`playground-tdd-plan.md`](playground-tdd-plan.md). Unit and
end-to-end tests run in Chromium through pytest + Playwright. Corpus IR hashes,
data-codec parity, engine fixtures, artifacts, state invalidation, hostile
worker behavior, path-prefixed deployment, and origin isolation are executable
requirements.

The release is not complete until GitHub Pages deployment, version display,
Pyodide/engine pin refresh instructions, and a fresh-browser dogfood walk are
finished.

## Future local runtime

A future `bayescycle ui` serves the same assets on loopback behind a random
session token and implements the same runtime contract using native bayeswire,
real run directories, the Bayesite external-command adapter, and pinned `uvx`
visualization tools. The local runtime adds persistence, replay, richer file
workflows, and the complete trustworthy-process product without changing the
browser UI's artifact vocabulary.
