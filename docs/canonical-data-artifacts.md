# Canonical data artifacts

Bayescycle owns the JSON data artifact exchanged between workflow stages:

```text
bayescycle.data.json.v1
```

A data document is a JSON object with a `format` marker and a `variables` map.
Each variable is a typed, explicitly shaped array whose `values` are flattened in
row-major order:

```json
{
  "format": "bayescycle.data.json.v1",
  "variables": {
    "x": {"dtype": "float64", "shape": [3], "values": [0.1, 0.2, 0.3]},
    "y": {"dtype": "int64", "shape": [3], "values": [0, 1, 0]}
  }
}
```

Supported v1 dtypes are `bool`, `int32`, `int64`, `float32`, and `float64`.
`prod(shape)` must equal `len(values)`. Ragged arrays, object arrays, and
backend-specific metadata are rejected before a run directory is created.

Legacy plain JSON inputs such as `{"x": [0.1, 0.2], "y": 1}` are still accepted
at CLI boundaries and normalized into the canonical document in `run/data.json`.
Backend adapters materialize that canonical file into backend-native runtime
inputs. For Bayesite, these transient files live under `run/.bayesite/` and are
not the cross-stage artifact.

## Walkthrough: same-backend simulation and recovery fit

Start with fixed covariates and known truth:

```bash
uv run bayescycle simulate model.py \
  --data inputs.json \
  --truth truth.json \
  -o run-sim/ \
  --backend bayesite \
  --seed 1
```

The durable output is canonical:

```text
run-sim/simulated_data.json  # bayescycle.data.json.v1
```

A Bayesite recovery fit can consume that same canonical artifact. The Bayesite
adapter converts it to `run-recover-fit/.bayesite/data.json` for the engine:

```bash
uv run bayescycle sample model.py \
  --data run-sim/simulated_data.json \
  -o run-recover-fit/ \
  --backend bayesite \
  --seed 2 --chains 4 --warmup 400 --draws 500
```

## Walkthrough: explicit mixed-backend interop boundary

The same canonical simulated data can be consumed by the in-process jaxstanv5
adapter without exposing Bayesite-native data to jaxstanv5:

```bash
uv run bayescycle sample model.py \
  --data run-sim/simulated_data.json \
  -o run-recover-fit-jax/ \
  --backend jaxstanv5 \
  --seed 2 --chains 4 --warmup 400 --draws 500
```

The jaxstanv5 adapter materializes `bayescycle.data.json.v1` as the plain
array-like mapping expected by `model.bind(...)`. No Bayesite-native
`dtype`/`shape`/`values` file is passed across the workflow-stage boundary.

## Prior-predictive inputs

Prior-predictive runs use the same canonical data snapshot for declared inputs:

```bash
uv run bayescycle prior-predictive model.py \
  --data inputs.json \
  -o run-prior/ \
  --backend bayesite \
  --seed 123 --draws 400
```

`run-prior/data.json` records the canonical `bayescycle.data.json.v1` input, and
`run-prior/manifest.json` records the data artifact format for consumers.
