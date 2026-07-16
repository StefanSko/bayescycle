# Generated datasets, version 0

This provisional contract defines functional dataset generation and the
`generated_datasets.ndjson` artifact shared by Bayescycle, Bayesite, and the
browser Playground. Producers and consumers must check the
`v0-provisional` marker before interpreting a document.

## Operation

For fixed design/context `x`, generation implements:

```text
for draw_index in 0..count:
    theta[draw_index] ~ parameter_source
    y[draw_index]     ~ outcomes(model, x, theta[draw_index])
```

Each draw contains the pair `(theta, complete_dataset)`. Parameters are redrawn
for every dataset. Generation is separate from conditioning and never performs
inference.

There are exactly three parameter sources:

- **fixed** — one complete canonical parameter document; its point-mass value
  repeats for every draw;
- **model-prior** — parameters and outcomes are drawn from the same closed
  Bayeswire model;
- **posterior** — retained natural-scale draws are selected with replacement
  from a structurally compatible fit, preserving source-draw lineage.

A score factor, free value, distribution, design, fixed parameter, or posterior
layout without supported ancestral semantics fails explicitly. Factors are
never silently discarded. Models containing non-`Param` free values are not
supported by this v0 generation profile.

## Generation plan

An executable plan owns exact model, design, and source bytes. Durable
`generation-plan.json` is hash-resolved provenance, encoded as compact UTF-8
JSON followed by one LF:

```json
{
  "generation_plan_format":"v0-provisional",
  "kind":"draw",
  "count":100,
  "seed":0,
  "design_source":{"x":"linspace(-2, 2, 25)"},
  "distribution":{
    "kind":"joint-predict",
    "parameters":{"kind":"fixed","parameters_hash":"sha256:..."},
    "outcomes":{
      "kind":"model-outcomes",
      "model_hash":"sha256:...",
      "design_hash":"sha256:..."
    }
  }
}
```

The other exact parameter-source forms are:

```json
{
  "kind":"model-prior",
  "model_hash":"sha256:...",
  "authored_provenance":null
}
```

```json
{
  "kind":"posterior",
  "fit_hash":"sha256:...",
  "fit_model_hash":"sha256:...",
  "fit_data_hash":"sha256:..."
}
```

Optional top-level `design_source`, when present, appears between `seed` and
`distribution` and maps design slot names to the non-empty expression strings
that authored them. Keys and values must be well-formed Unicode scalar-value
strings; keys are non-empty, and entries are serialized in strictly ascending
lexicographic order by Unicode code points. Parsers reject any other entry
order. The field is provenance only: consumers never evaluate it, require it
to match design variables, or use it for generation. Exact `design.json` bytes
remain authoritative. Plans authored through direct JSON omit the field.

Optional model-prior provenance has exactly
`claimed_source_model_hash` and `claimed_outcome_model_hash`. These are opaque
claims, not verified identities, and never affect execution. Verified hashes
are lowercase SHA-256 strings with the `sha256:` prefix and are recomputed from
resolved payload bytes. Hash-only plans are executable only after all payloads
are resolved and verified; posterior plans reconstructed from files require a
portable fit with matching header/trailer model-data fingerprints.

Objects reject unknown, missing, duplicate, or out-of-order keys.

## Artifact stream

`generated_datasets.ndjson` is UTF-8 NDJSON containing one header, exactly
`count` draw records in zero-based order, and one trailer envelope. Every line
ends in LF and contains finite JSON.

Header, draw, and trailer values repeat these identities:

```json
{
  "generated_datasets_format":"v0-provisional",
  "artifact_kind":"generated_dataset_pairs",
  "artifact_scope":"parameter_and_complete_dataset_joint_draws"
}
```

### Header

The header has exactly these keys in this order:

```json
{
  "generated_datasets_format":"v0-provisional",
  "artifact_kind":"generated_dataset_pairs",
  "artifact_scope":"parameter_and_complete_dataset_joint_draws",
  "workflow_phases":["parse_json","decode_ir","bind_design","draw_parameters","simulate_outcomes","emit_artifact"],
  "generation_model_hash":"sha256:...",
  "design_hash":"sha256:...",
  "parameter_source":{"kind":"fixed","parameters_hash":"sha256:..."},
  "count":2,
  "seed":0,
  "draw_index_base":"zero_based_generation_order",
  "parameter_schema":[{"name":"alpha","dtype":"float64","shape":[]}],
  "dataset_schema":[{"name":"x","dtype":"float64","shape":[3]}]
}
```

Schema entries have exactly `name`, `dtype`, and `shape`, preserve runtime
order, and use the dtypes from [`data-document-v1.md`](data-document-v1.md).
The parameter source uses the same exact form as the plan.

### Draw record

Each draw has exactly:

```json
{
  "generated_datasets_format":"v0-provisional",
  "artifact_kind":"generated_dataset_pairs",
  "artifact_scope":"parameter_and_complete_dataset_joint_draws",
  "draw_index":0,
  "draw_count":2,
  "parameters":{
    "format":"bayescycle.data.json.v1",
    "variables":{"alpha":{"dtype":"float64","shape":[],"values":[0.5]}}
  },
  "dataset":{
    "format":"bayescycle.data.json.v1",
    "variables":{"x":{"dtype":"float64","shape":[3],"values":[-1.0,0.0,1.0]}}
  },
  "source_lineage":{"kind":"fixed"}
}
```

`parameters` and `dataset` are strict canonical data documents. The complete
dataset begins with every design variable in exact design order and then the
generated outcomes in runtime order. It can be passed directly to conditioning
without model-aware frontend transformation.

Source lineage has one exact form:

```json
{"kind":"fixed"}
{"kind":"model-prior","source_draw_index":0}
{"kind":"posterior","source_draw_index":7,"chain":1,"draw":3}
```

For model-prior sources, `source_draw_index` equals `draw_index`. Posterior
indices identify one validated retained fit record selected with replacement.

### Trailer

The final line is an object whose sole key is `trailer`. Its value has exactly:

```json
{
  "generated_datasets_format":"v0-provisional",
  "artifact_kind":"generated_dataset_pairs",
  "artifact_scope":"parameter_and_complete_dataset_joint_draws",
  "workflow_phases":["parse_json","decode_ir","bind_design","draw_parameters","simulate_outcomes","emit_artifact"],
  "generation_model_hash":"sha256:...",
  "design_hash":"sha256:...",
  "parameter_source":{"kind":"fixed","parameters_hash":"sha256:..."},
  "count":2,
  "seed":0,
  "draw_count":2,
  "complete":true
}
```

Header and trailer identities, source, count, and seed must agree. Draw indices
are contiguous, and every variable's order, dtype, and shape matches its header
schema. Parsers reject truncation, trailing records, inconsistent lineage,
non-finite values, and invalid canonical documents.

## Exact bytes and verification

Hashes cover exact payload bytes. Parsers preserve the complete accepted
artifact bytes. Selection records the exact byte span of the nested
`parameters` and `dataset` values and returns that span followed by one LF; it
does not reserialize parsed values. The selected dataset bytes become the exact
`data.json` bytes used for conditioning and model/data fingerprinting.

Standalone parsing validates stream structure, schemas, and internal indices.
Resolver-backed verification additionally checks all hashes, source values,
posterior lineage, and the exact design prefix. Only resolver-backed verification
authorizes recovery claims.

## Bounds

Boundaries reject rather than clamp:

- `count`: safe JSON integer in `1..1000`;
- `seed`: safe JSON integer in `0..9007199254740991`;
- plan or generation `run.json`: at most 1 MiB;
- model IR, design, fixed parameters, fit data, or posterior input: at most
  8 MiB each;
- one NDJSON line: at most 8 MiB including LF;
- complete paired artifact: at most 64 MiB;
- JSON nesting depth: at most 64;
- every generation-profile integer, including canonical `int64` values, shapes,
  and indices, must be exactly representable by JavaScript.

These are interoperability ceilings, not guarantees that every request below a
ceiling fits available memory.

## Compatibility

This artifact is additive to the v0 workflow. Existing
`simulated_data.json`, `prior_predictive.ndjson`, and
`posterior_predictive.ndjson` retain their existing meanings. Unified
functional generation returns only `generated_datasets.ndjson`; it does not
project a legacy artifact.
