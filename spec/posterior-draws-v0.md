# Posterior draws NDJSON v0

`posterior.ndjson` is the provisional posterior-draw stream used by the
bayescycle run-directory v0 contract.

The current wire shape intentionally matches the v0-provisional stream first
emitted by Bayesite so existing diagnostics and ArviZ export tools can consume
runs from multiple backends.

## Stream shape

The stream contains:

1. one header object
2. one draw object per retained draw
3. one trailer object of the form `{"trailer": {...}}`

All JSON objects are newline-delimited. Blank lines are invalid.

## Required marker

The header, draw lines, and trailer use:

```json
"draws_format": "v0-provisional"
```

The marker is deliberately provisional. Consumers must check it before parsing.

## Header facts

The header records:

- artifact identity: `posterior_draws` over
  `observed_data_conditioned_parameter_draws`
- workflow phase labels
- parameter names, shapes, coordinate order, and parameter order
- sampler settings: warmup, retained draws, max tree depth, target acceptance
- seed, chain count/order, and total retained draw count
- optional model/data identity metadata, including `model_data_fingerprint`
- `sample_stats_mode` when per-draw sampler statistics are present

`model_data_fingerprint`, when present, is `sha256:` plus the SHA-256 digest of
`b"bayescycle-model-data-v1\n" + model_ir_bytes + b"\n" + data_json_bytes`, per
the normative [`model-data-fingerprint-v1.md`](model-data-fingerprint-v1.md).
`data_json_bytes`
is the canonical `run/data.json` artifact for both the in-process bayesjax
backend and the Bayesite engine: bayescycle passes that file directly to every
data-consuming engine command, so both backends fingerprint identical bytes.

## Draw facts

Each retained draw records:

- global retained draw index
- chain id and per-chain draw index
- constrained parameter values keyed by parameter name
- parameter and chain metadata sufficient to detect truncation/reordering
- per-draw sampler stats when announced by `sample_stats_mode`

For `sample_stats_mode: "per_draw_v2"`, each draw includes:

```json
{
  "diverging": false,
  "tree_depth": 3,
  "tree_accept": 0.91,
  "energy": 123.4
}
```

`per_draw_v1` is the legacy mode with `diverging`, `tree_depth`, and
`tree_accept` but no `energy`.

## Trailer facts

The trailer records:

- completion metadata: seed, draw count, chain order, parameter order
- per-chain sampler summaries: retained draw count, divergences,
  tree-depth histogram, final step size, and mean acceptance
- cross-chain R-hat and ESS maps

Unavailable R-hat/ESS values are encoded as JSON `null`, never `NaN` or
`Infinity`.

## Backend boundary

Sampler facts must come from the backend that ran NUTS. Bayescycle may serialize
those facts into this artifact, but it must not invent sampler semantics or reach
into backend-private state.
