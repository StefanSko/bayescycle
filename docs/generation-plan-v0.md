# Functional generation plan and paired artifact v0

This provisional contract defines the backend-neutral generation operation used
by Bayescycle runtimes. It is subordinate to the Bayeswire IR contract and to
[`playground/invariants.md`](../playground/invariants.md). It does not add model
authoring or composition semantics.

## User operation

The user-facing operation is:

```text
generate_datasets(model, design, parameter_source, count, seed)
```

It lowers to the closed plan:

```text
Draw(
  JointPredict(
    parameter_source,
    OutcomesOf(model, design),
  ),
  count,
  seed,
)
```

The algebra is descriptive data, not a generic `Dist`, `map`, or `flatMap` API.
Plans contain no closures, DOM objects, workers, backend instances, mutable
aliases, or Bayeswire declaration objects.

## Laws

For fixed design/context `x`, generation implements:

```text
for draw_index in 0..count:
    theta[draw_index] ~ parameter_source
    y[draw_index]     ~ outcomes(model, x, theta[draw_index])
```

Each draw contains the pair `(theta, complete_dataset)`. Parameters are redrawn
for every dataset. `Fixed` is a point mass, so its parameter documents repeat.
One parameter draw followed by several outcome draws is a different operation
and is not part of v0.

Generation is not conditioning. Conditioning remains:

```text
Condition(model, dataset, inference_settings)
  -> posterior + diagnostics
```

A posterior may become a later `PosteriorOf` source only after a successful
fit. Generation never performs inference and conditioning is not represented
as inversion of the outcome model.

## Parameter sources

There are exactly three variants.

### `Fixed`

`Fixed(parameter_document)` carries a complete canonical
`bayescycle.data.json.v1` document of natural-scale parameter values. The
runtime rejects missing, extra, incompatible, constrained-invalid, or
non-finite values. It never fills values from model IR in the frontend.

### `ModelPrior`

`ModelPrior(generation_model_ir, authored_provenance)` draws the authored prior
and outcomes from one ordinary closed Bayeswire model. The same closed model is
the `OutcomesOf` model. An Implementation-A composite, if supplied later, is
indistinguishable from any other closed model.

Optional authored provenance is an opaque claim supplied by the caller:

```json
{
  "claimed_source_model_hash": "sha256:...",
  "claimed_outcome_model_hash": "sha256:..."
}
```

Both keys are required when this object is present. They are checked only for
hash syntax, are always labeled `claimed_`, and never change execution. B does
not derive, inspect, attest, or validate composition semantics from them.

### `PosteriorOf`

`PosteriorOf(fit_artifact)` is the empirical distribution over retained
natural-scale fit draws. Sampling is with replacement using the generation
seed. Source-draw lineage is retained in each generated record.

A `fit_artifact` is not arbitrary NDJSON. It is an immutable conditioning result
that co-owns defensive copies of the exact fit-model IR, conditioning-data, and
posterior bytes. Browser v0 does not import standalone posterior streams into a
source. A durable Bayescycle run reconstructs this value only from the
co-travelling files defined below.

Posterior-source validation requires all of the following:

1. exact fit-model bytes equal the outcome-model bytes;
2. a complete `v0-provisional` posterior stream with the posterior-draw
   kind/scope, one header, at least one retained draw, its declared number of
   contiguous draw records, and one final trailer with no trailing records;
3. the header parameter order and shapes agree with every draw; trailer
   parameter order/count agree with the header; header and trailer chain
   order/count, draw count, optional posterior identity, and optional model/data
   fingerprint agree. Shapes are header-owned in posterior v0 and are not
   required in the trailer;
4. every draw has the declared finite constrained values and a unique global
   index whose chain/per-chain draw coordinates agree with stream order;
5. the engine always validates the declared parameter names, order, shapes, and
   constrained values against the decoded model bound to the fit data.
   `posterior_identity_hash`, when present, is additionally present in both
   header and trailer and equals the identity computed from that model and fit
   data;
6. `model_data_fingerprint` is either present in both header and trailer with
   equal values that match the normative exact-byte framing, or absent from
   both. An absent fingerprint is accepted only for a fit value produced by the
   current runtime's conditioning transition with its co-owned model/data
   bytes. It is not accepted from an imported standalone stream.

Thus a portable fit has an exact fingerprint plus model-checked layout; the
browser's fingerprint-less Wasm fit has its runtime association plus posterior
identity. A conforming backend fit without posterior identity remains usable
when its exact fingerprint and model-checked layout succeed.

The generation design may differ from the fit data, but must bind the same
closed model. Before dispatch, the engine binds the outcome model to the new
design and requires every posterior parameter name, order, resolved shape, and
constraint to remain compatible with that generation-bound layout. A
data-dependent size or constraint change therefore fails before source-draw
selection. If the exact fit association is unavailable or stale, the posterior
source is unavailable rather than reconstructed from parameter names.

## Immutable plan values

In-memory plans own defensive copies of exact artifact bytes. They have these
logical variants:

```text
Fixed(parameters_bytes)
ModelPrior(model_ir_bytes, authored_provenance?)
PosteriorOf(fit_artifact)
OutcomesOf(model_ir_bytes, design_bytes)
JointPredict(parameters, outcomes)
Draw(distribution, count, seed)
```

`ModelPrior.model_ir_bytes` must byte-equal `OutcomesOf.model_ir_bytes`.
`PosteriorOf.fit_artifact.model_ir_bytes` must byte-equal
`OutcomesOf.model_ir_bytes`. Constructors reject other combinations.

### Executable request shape

The UI-facing structured-clone value has the exact recursive shape below.
Property names are normative. JavaScript byte fields are `Uint8Array`; Python
uses immutable `bytes`. Constructors defensively copy every byte field and
reject missing or unknown properties at every node.

```text
{
  kind: "draw",
  count: integer,
  seed: integer,
  distribution: {
    kind: "joint-predict",
    parameters: FixedExecutable | ModelPriorExecutable | PosteriorExecutable,
    outcomes: {
      kind: "model-outcomes",
      modelIrBytes: bytes,
      designBytes: bytes
    }
  }
}

FixedExecutable = {
  kind: "fixed",
  parametersBytes: bytes
}

ModelPriorExecutable = {
  kind: "model-prior",
  modelIrBytes: bytes,
  authoredProvenance: null | {
    claimedSourceModelHash: "sha256:...",
    claimedOutcomeModelHash: "sha256:..."
  }
}

PosteriorExecutable = {
  kind: "posterior",
  fitArtifact: {
    kind: "fit-artifact",
    modelIrBytes: bytes,
    dataBytes: bytes,
    posteriorBytes: bytes,
    association: "runtime" | "portable"
  }
}
```

`runtime` association can be constructed only from the current runtime's
successful conditioning result and permits the documented fingerprint-less
Wasm profile. `portable` association requires matching header/trailer exact-byte
fingerprints. Public parsers do not accept an arbitrary caller assertion as a
runtime association.

The JSON plan document is versioned provenance, not an executable request. It
contains hashes over the exact bytes held by the in-memory plan:

```json
{
  "generation_plan_format": "v0-provisional",
  "kind": "draw",
  "count": 100,
  "seed": 0,
  "distribution": {
    "kind": "joint-predict",
    "parameters": {
      "kind": "model-prior",
      "model_hash": "sha256:...",
      "authored_provenance": null
    },
    "outcomes": {
      "kind": "model-outcomes",
      "model_hash": "sha256:...",
      "design_hash": "sha256:..."
    }
  }
}
```

The other parameter-source records are exactly:

```json
{"kind":"fixed","parameters_hash":"sha256:..."}
```

```json
{
  "kind":"posterior",
  "fit_hash":"sha256:...",
  "fit_model_hash":"sha256:...",
  "fit_data_hash":"sha256:..."
}
```

The root requires `generation_plan_format` exactly equal to
`"v0-provisional"` before interpreting node tags. Objects reject unknown and
missing keys. Verified hashes are lowercase SHA-256 strings with the `sha256:` prefix and
are recomputed from bytes owned by the plan. Claimed authored hashes are only
syntax-checked and are never described as verified. Object key order is the
order shown above. JSON identity uses compact UTF-8 JSON followed by one LF. Parsed plans are
reconstructed only when an artifact resolver supplies bytes whose hashes match;
hash-only documents are never executable by themselves. A hash-only posterior
plan resolves as `portable`, never `runtime`, and therefore requires matching
header/trailer model-data fingerprints.

## Bounds

All boundaries reject values rather than clamp them:

- `count`: safe JSON integer in `1..=1000`;
- `seed`: safe JSON integer in `0..=9007199254740991`;
- `generation-plan.json` or generation `run.json`: at most 1 MiB each;
- model IR, design, fixed parameters, fit data, or fit input: at most 8 MiB each;
- one metadata path: at most 255 UTF-8 bytes; backend identity: 1..=64 ASCII
  letters, digits, dots, underscores, or hyphens;
- one paired-artifact NDJSON line: at most 8 MiB including its terminating LF;
- complete paired artifact: at most 64 MiB including every LF;
- JSON container depth: at most 64. The root object has depth 1; entering an
  object or array adds one; scalar values do not add depth.

Byte limits count UTF-8 bytes, not characters. Boundaries reject an oversized
buffer before UTF-8 decoding or JSON parsing. A bounded pre-parse depth scan
rejects depth 65 before constructing nested application values.

These are v0 interoperability ceilings, not claims that every request under a
ceiling is executable in available memory.

## Capability failures

A runtime fails visibly before claiming success when:

- a score factor has no ancestral sampling semantics;
- a required stochastic site or observed distribution is not sampleable;
- fixed values are incomplete, extra, shape-incompatible, or violate a
  constraint;
- declared design values are missing, extra, or incompatible;
- posterior model/data/structural identity does not match;
- the model has any non-Param free value, including a partially observed latent
  value, because browser generation v0 has no semantics for carrying it to a
  new design;
- any generation-protocol integer is outside JavaScript's exactly representable
  range `[-9007199254740991, 9007199254740991]`. This includes canonical
  `int64` values, canonical/schema shape dimensions, count, seed, draw/chain
  indices, and declared counts;
- a document, count, seed, nesting depth, line, or output exceeds a bound;
- the backend does not support the source variant.

Factors are never silently dropped. A v0 parameter document and
`parameter_schema` contain exactly declaration-backed Params in resolved IR
order. Hierarchical and source-only Params are ordinary Params and are retained.
The runtime rejects a generation model whose free-value layout contains
anything else. The frontend does not inspect raw model IR to predict these
failures; the runtime/engine reports them.

The safe-integer rule is a generation-v0 interoperability profile layered on
the canonical data document. It deliberately rejects some otherwise valid
`int64` documents so Python, JavaScript, CLI, and Wasm produce the same result.

## `generated_datasets.ndjson`

The paired artifact is UTF-8 NDJSON with one header, exactly `count` draw
records in zero-based order, and one trailer envelope. Every line is finite JSON
and ends in LF.

### Common identities

The exact marker fields are:

```json
{
  "generated_datasets_format": "v0-provisional",
  "artifact_kind": "generated_dataset_pairs",
  "artifact_scope": "parameter_and_complete_dataset_joint_draws"
}
```

Verified hashes are SHA-256 over exact bytes owned by the executable plan.
`generation_model_hash` identifies the closed model used to generate outcomes,
and `design_hash` identifies the exact canonical design document. Opaque
`claimed_` hashes, when present inside model-prior provenance, are not verified
payload identities and never change execution.

### Header

The header has exactly these keys in this order:

```json
{
  "generated_datasets_format": "v0-provisional",
  "artifact_kind": "generated_dataset_pairs",
  "artifact_scope": "parameter_and_complete_dataset_joint_draws",
  "workflow_phases": ["parse_json", "decode_ir", "bind_design", "draw_parameters", "simulate_outcomes", "emit_artifact"],
  "generation_model_hash": "sha256:...",
  "design_hash": "sha256:...",
  "parameter_source": {"kind":"fixed","parameters_hash":"sha256:..."},
  "count": 2,
  "seed": 0,
  "draw_index_base": "zero_based_generation_order",
  "parameter_schema": [
    {"name":"alpha","dtype":"float64","shape":[]}
  ],
  "dataset_schema": [
    {"name":"x","dtype":"float64","shape":[3]},
    {"name":"y","dtype":"float64","shape":[3]}
  ]
}
```

Schema entries preserve model/runtime order and have exactly `name`, `dtype`,
and `shape`. Supported dtypes are those of `bayescycle.data.json.v1` under the
safe-integer interoperability profile. `parameter_source` has the exact
serialized source shape defined above, including model-prior claimed provenance
when present.

### Draw record

Every draw record has exactly:

```json
{
  "generated_datasets_format": "v0-provisional",
  "artifact_kind": "generated_dataset_pairs",
  "artifact_scope": "parameter_and_complete_dataset_joint_draws",
  "draw_index": 0,
  "draw_count": 2,
  "parameters": {
    "format":"bayescycle.data.json.v1",
    "variables":{"alpha":{"dtype":"float64","shape":[],"values":[0.5]}}
  },
  "dataset": {
    "format":"bayescycle.data.json.v1",
    "variables":{
      "x":{"dtype":"float64","shape":[3],"values":[-1.0,0.0,1.0]},
      "y":{"dtype":"float64","shape":[3],"values":[-0.2,0.5,1.2]}
    }
  },
  "source_lineage": {"kind":"fixed"}
}
```

`parameters` and `dataset` are strict canonical data documents. The dataset is
complete: declared design/context variables followed by generated observed
variables in runtime order. It can be selected and passed directly to
conditioning without model-aware frontend transformation.

Source lineage is one exact variant:

```json
{"kind":"fixed"}
{"kind":"model-prior","source_draw_index":0}
{"kind":"posterior","source_draw_index":7,"chain":1,"draw":3}
```

Posterior indices identify the retained fit record selected with replacement.
For fixed sources, every `parameters` document is equal to the fixed source as
an ordered strict canonical data value after parsing; insignificant source-file
whitespace is not part of this equality. For model-prior sources,
`source_draw_index` equals `draw_index`. For posterior sources, the global
source index, chain, and per-chain draw identify one validated retained record,
and `parameters` equals that record's declared constrained parameter values.
Repeated posterior source indices are valid because selection is with
replacement; every repetition still receives a fresh outcome draw.

A standalone parser verifies exact keys, stream structure, schemas, and internal
index relationships. Resolver-backed verification additionally recomputes all
verified hashes, checks fixed values or posterior parameters against the source
payload, and requires each dataset to begin with every design variable in exact
design order with equal dtype, shape, and values. This comparison uses the
hash-resolved canonical design document and requires no model-IR inspection.
Only resolver-backed verification authorizes recovery claims.

### Trailer

The final line is an object with the sole key `trailer`. Its value has exactly:

```json
{
  "generated_datasets_format": "v0-provisional",
  "artifact_kind": "generated_dataset_pairs",
  "artifact_scope": "parameter_and_complete_dataset_joint_draws",
  "workflow_phases": ["parse_json", "decode_ir", "bind_design", "draw_parameters", "simulate_outcomes", "emit_artifact"],
  "generation_model_hash": "sha256:...",
  "design_hash": "sha256:...",
  "parameter_source": {"kind":"fixed","parameters_hash":"sha256:..."},
  "count": 2,
  "seed": 0,
  "draw_count": 2,
  "complete": true
}
```

Header and trailer repeat the complete verified source descriptor and must
agree byte-for-byte as JSON values. Draw indices are contiguous. Variable order,
dtype, and shape must match the header schemas. A parser rejects truncation,
duplicate or missing indices, inconsistent source lineage, unknown fields,
non-finite or unsafe-integer values, invalid canonical documents, excessive
depth, oversized input, and trailing records.

Selection returns immutable copies of both the canonical parameter document and
canonical complete dataset for one index. The bounded NDJSON parser records the
exact UTF-8 byte span of each nested `parameters` and `dataset` JSON value while
validating it. Standalone selection bytes are that exact span followed by
exactly one LF; they are never produced from parsed numbers or strings. Thus
Python and JavaScript select byte-identical documents without defining a second
cross-language JSON-number serializer. These selected dataset bytes become the
exact `data.json` bytes used for later conditioning and fingerprinting.

Writers emit nested documents compactly, but readers preserve any accepted
insignificant whitespace inside the recorded span. The downloaded paired
artifact itself also preserves its complete exact validated input bytes; parsing
and selection do not authorize reserialization of that download.

## Runtime request

The UI-facing `runtime.run` operation is:

```text
RunRequest {
  type: "run",
  id,
  operation: "generate",
  plan: Draw(JointPredict(...))
}
```

The plan carries exact bytes in memory. `BrowserRuntime` alone lowers it to the
private engine protocol. Application and UI modules must not contain private
engine command names for fixed, prior, or posterior generation.

The native Bayesite operation receives one bounded request containing closed
model IR, canonical design, one source variant and payload, count, and seed. It
performs all requested draws in one core invocation and returns the paired
artifact. The browser must not create one worker or Wasm instance per dataset.

Legacy private command lowering is permitted only when it exactly implements the
requested law. The current prior-predictive command can implement model-prior
count, and one fixed simulation can implement fixed `count=1`. Fixed
multi-count and posterior sampling-with-replacement return
`UnsupportedCapability` without issuing a legacy command until native
`generate` is staged. The UI does not expose a source/count combination before
its exact native path is available.

## Durable generation run

A portable generation-only run has an operation-specific required set; it does
not require `posterior.ndjson` as an output:

```text
run/
  model.ir.json
  design.json
  generation-plan.json
  generated_datasets.ndjson
  run.json
```

`generation-plan.json` is the exact compact plan document. Its payload hashes
resolve only to files in the same moved run directory:

- fixed adds `fixed-parameters.json`;
- model-prior needs no source payload beyond `model.ir.json`;
- posterior adds `source-posterior.ndjson` and `source-fit-data.json`; its fit
  model resolves to the byte-identical `model.ir.json`.

All JSON input files retain the exact bytes hashed by the plan. A generation run
copies posterior source payloads; v0 does not use external or absolute artifact
references. A posterior generation run is materializable only when its source
fit has matching header/trailer model-data fingerprints. A fingerprint-less
`runtime` association may generate an in-browser paired artifact, but attempting
to create a portable run fails before any run-directory file is written; copying
its bytes never promotes it to `portable` authority.

Generation `run.json` is a distinct exact-key profile:

```json
{
  "format": "bayescycle.generation-run.v0",
  "kind": "generate",
  "backend": "bayesite",
  "plan": {"path":"generation-plan.json","sha256":"sha256:...","format":"v0-provisional"},
  "model": {"path":"model.ir.json","sha256":"sha256:...","format":"bayeswire_ir.v1"},
  "inputs": [
    {"role":"design","path":"design.json","sha256":"sha256:...","format":"bayescycle.data.json.v1"},
    {"role":"fixed-parameters","path":"fixed-parameters.json","sha256":"sha256:...","format":"bayescycle.data.json.v1"}
  ],
  "outputs": [
    {"role":"generated-datasets","path":"generated_datasets.ndjson","sha256":"sha256:...","format":"v0-provisional"}
  ]
}
```

`backend` is a bounded backend identity; it does not embed an executable path.
Role-to-path mappings are exact: plan → `generation-plan.json`, model →
`model.ir.json`, design → `design.json`, fixed parameters →
`fixed-parameters.json`, source posterior → `source-posterior.ndjson`, source fit
data → `source-fit-data.json`, and output → `generated_datasets.ndjson`. Input
order is design first, then no source entry for model-prior, one fixed entry for
fixed, or source posterior followed by source fit data for posterior. The
posterior formats are respectively `v0-provisional` and
`bayescycle.data.json.v1`. Paths may not be duplicated. All nested objects
reject unknown or missing keys.

The canonical names are single normalized POSIX path segments. Required files
must be regular files, no required path may be a symbolic link, and resolution
must remain beneath the run directory. Hashes are recomputed from exact file
bytes before plan interpretation or replay.
Generation replay branches on the distinct format marker, consumes closed IR
directly, and does not re-execute or require the original Python model source.
Existing `bayescycle.run.v1` profiles and their external-source verification
remain unchanged. A moved directory therefore replays using only its local
hash-matched payloads.

Conditioning runs retain their existing `model.ir.json`, `data.json`, and
`posterior.ndjson` required set. A selected generated dataset is materialized as
that conditioning run's exact `data.json`; its later fingerprint uses those
actual bytes.

## State and invalidation

State carries separate revisions for:

```text
target model, observed data, design, fixed values, parameter-source selection,
generation settings, inference settings, selected generated draw, fit artifacts
```

Dependencies are:

```text
model + design + source + generation settings -> generated collection
fit model + selected/observed dataset + inference settings -> posterior + diagnostics
fit posterior + design + generation settings -> posterior-generated collection
selected pair + selected-dataset fit -> recovery
```

Therefore:

- switching parameter-source kind or source payload increments the source
  revision, invalidates the current generation key, and rejects an old
  completion;
- editing inference controls does not erase a completed fit or any generated
  collection. It only changes the key for the next conditioning request;
- a newly successful replacement fit changes fit lineage, invalidates
  collections generated from the replaced posterior, and disables that old
  `PosteriorOf`; a failed replacement preserves the old fit and descendants;
- generation-setting edits do not invalidate unrelated observed-data fits;
- selection edits invalidate only selected-dataset fit descendants and recovery;
- model, fit-data, or fit-artifact lineage changes disable `PosteriorOf`;
- fixed and model-prior collections have no fit dependency;
- failed descendants preserve successful ancestors;
- async completions carry the complete dependency key, including explicit
  source identity, and stale keys are ignored;
- source kind is explicit and is never inferred from artifact presence.

## Compatibility and non-goals

`generated_datasets.ndjson` is additive in v0. Existing artifacts retain their
meanings:

- `simulated_data.json` remains one fixed-truth canonical dataset;
- `prior_predictive.ndjson` remains the source-specific prior-predictive stream;
- `posterior_predictive.ndjson` remains the source-fit predictive stream.

Unified `generate` returns `generated_datasets.ndjson` and does not project a
legacy artifact. Private legacy operations may remain during migration and
continue to return their old artifacts with byte semantics unchanged. No fixed
multi-count or posterior-with-replacement collection is projected into a legacy
shape unless a future separately versioned contract defines that projection.
The new UI uses the paired collection for selection and recovery.

This contract does not add Bayeswire component composition, `with_prior`,
parameter-name matching in the frontend, raw-IR inspection, arbitrary
serialized functions, automatic execution, generated semantic forms, or
one-worker-per-draw execution.
