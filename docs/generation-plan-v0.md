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

Optional authored provenance is opaque lineage supplied by the caller:

```json
{
  "source_model_hash": "sha256:...",
  "outcome_model_hash": "sha256:..."
}
```

Both keys are required when this object is present. B does not derive, inspect,
or validate composition semantics from them.

### `PosteriorOf`

`PosteriorOf(fit_artifact, fit_model_ir, fit_data_document)` is the empirical
distribution over retained natural-scale fit draws. Sampling is with replacement
using the generation seed. Source-draw lineage is retained in each generated
record.

The exact fit model bytes must equal the outcome-model bytes. The fit's
`model_data_fingerprint`, when present, must match the exact fit-model and
fit-data bytes. Structural posterior identity remains mandatory for transports
whose fit stream predates that fingerprint. The generation design may differ
from the fit data, but must bind the same closed model. If exact fit lineage is
unavailable or stale, this source is unavailable rather than guessed.

## Immutable plan values

In-memory plans own defensive copies of exact artifact bytes. They have these
logical variants:

```text
Fixed(parameters_bytes)
ModelPrior(model_ir_bytes, authored_provenance?)
PosteriorOf(fit_bytes, fit_model_ir_bytes, fit_data_bytes)
OutcomesOf(model_ir_bytes, design_bytes)
JointPredict(parameters, outcomes)
Draw(distribution, count, seed)
```

`ModelPrior.model_ir_bytes` must byte-equal `OutcomesOf.model_ir_bytes`.
`PosteriorOf.fit_model_ir_bytes` must byte-equal
`OutcomesOf.model_ir_bytes`. Constructors reject other combinations.

The JSON plan document is provenance, not an executable request. It contains
hashes over the exact bytes held by the in-memory plan:

```json
{
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

Objects reject unknown and missing keys. Hashes are lowercase SHA-256 strings
with the `sha256:` prefix. Object key order is the order shown above. JSON
identity uses compact UTF-8 JSON followed by one LF. Parsed plans are
reconstructed only when an artifact resolver supplies bytes whose hashes match;
hash-only documents are never executable by themselves.

## Bounds

All boundaries reject values rather than clamp them:

- `count`: safe JSON integer in `1..=1000`;
- `seed`: safe JSON integer in `0..=9007199254740991`;
- model IR, design, fixed parameters, fit data, or fit input: at most 8 MiB each;
- one paired-artifact NDJSON line: at most 8 MiB;
- complete paired artifact: at most 64 MiB;
- nesting: bounded by the receiving JSON parser, and never less strict than the
  engine's documented parser bound.

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
- a document, count, seed, line, or output exceeds a bound;
- the backend does not support the source variant.

Factors are never silently dropped. The frontend does not inspect raw model IR
to predict these failures; the runtime/engine reports them.

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

Hashes are SHA-256 over received bytes. `generation_model_hash` identifies the
closed model used to generate outcomes. `target_model_hash` is the optional
caller-supplied authored outcome target hash and is otherwise equal to
`generation_model_hash`; this field is provenance only and never changes
execution. `design_hash` identifies the exact canonical design document.

### Header

The header has exactly these keys in this order:

```json
{
  "generated_datasets_format": "v0-provisional",
  "artifact_kind": "generated_dataset_pairs",
  "artifact_scope": "parameter_and_complete_dataset_joint_draws",
  "workflow_phases": ["parse_json", "decode_ir", "bind_design", "draw_parameters", "simulate_outcomes", "emit_artifact"],
  "generation_model_hash": "sha256:...",
  "target_model_hash": "sha256:...",
  "design_hash": "sha256:...",
  "parameter_source": {"kind":"fixed","parameters_hash":"sha256:..."},
  "authored_provenance": null,
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
and `shape`. Supported dtypes are those of `bayescycle.data.json.v1`.
`parameter_source` has the exact serialized source shape defined above.
`authored_provenance` is null or the exact two-key object defined above.

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

### Trailer

The final line is an object with the sole key `trailer`. Its value has exactly:

```json
{
  "generated_datasets_format": "v0-provisional",
  "artifact_kind": "generated_dataset_pairs",
  "artifact_scope": "parameter_and_complete_dataset_joint_draws",
  "workflow_phases": ["parse_json", "decode_ir", "bind_design", "draw_parameters", "simulate_outcomes", "emit_artifact"],
  "generation_model_hash": "sha256:...",
  "target_model_hash": "sha256:...",
  "design_hash": "sha256:...",
  "parameter_source_kind": "fixed",
  "count": 2,
  "seed": 0,
  "draw_count": 2,
  "complete": true
}
```

Header, records, and trailer must agree. Draw indices are contiguous. Variable
order, dtype, and shape must match the header schemas. A parser rejects
truncation, duplicate or missing indices, unknown fields, non-finite values,
invalid canonical documents, oversized input, and trailing records.

Selection returns immutable copies of both the canonical parameter document and
canonical complete dataset for one index. Download paths preserve the exact
validated bytes; parsing does not authorize reserialization of the downloaded
artifact.

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

## State and invalidation

State carries separate revisions for:

```text
target model, observed data, design, fixed values, generation settings,
inference settings, selected generated draw, fit artifacts
```

Dependencies are:

```text
model + design + source + generation settings -> generated collection
fit model + selected/observed dataset + inference settings -> posterior + diagnostics
fit posterior + design + generation settings -> posterior-generated collection
selected pair + selected-dataset fit -> recovery
```

Therefore:

- inference-setting edits do not invalidate generated collections;
- generation-setting edits do not invalidate unrelated observed-data fits;
- selection edits invalidate only selected-dataset fit descendants and recovery;
- model, fit-data, or fit-artifact lineage changes disable `PosteriorOf`;
- failed descendants preserve successful ancestors;
- async completions carry their dependency key and stale keys are ignored;
- source kind is explicit and is never inferred from artifact presence.

## Compatibility and non-goals

`generated_datasets.ndjson` is additive in v0. Existing artifacts retain their
meanings:

- `simulated_data.json` remains one fixed-truth canonical dataset;
- `prior_predictive.ndjson` remains the source-specific prior-predictive stream;
- `posterior_predictive.ndjson` remains the source-fit predictive stream.

Adapters may emit those legacy artifacts as explicit source-specific views in
addition to the paired collection. They must not rename or reinterpret them,
and the new UI uses the paired collection for selection and recovery.

This contract does not add Bayeswire component composition, `with_prior`,
parameter-name matching in the frontend, raw-IR inspection, arbitrary
serialized functions, automatic execution, generated semantic forms, or
one-worker-per-draw execution.
