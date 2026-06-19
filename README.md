# bayescycle

`bayescycle` is the Python workflow CLI that connects the `jaxstanv5` authoring
eDSL to the Bayesite Rust engine.

It is intentionally glue:

```text
model.py -> jaxstanv5 ModelMeta -> jaxstanv5 IR JSON -> bayesite engine -> run/
```

## Package split

- **bayesite**: Rust engine and IR-level CLI. It accepts IR plus data and writes
  draws. It stays dependency-free and WASM-clean.
- **jaxstanv5**: Python model declaration eDSL and IR serializer. It does not own
  workflow orchestration.
- **bayescycle**: Python workflow harness. It executes a Python model file,
  serializes IR, invokes the Bayesite binary, and owns run-directory ergonomics.

## Initial CLI

```bash
bayescycle sample model.py --data data.json -o run/
```

By default this prepares:

```text
run/model.ir.json
run/data.json
run/dims.json       # optional; written only when the model declares dimension metadata
```

and invokes the engine as:

```bash
bayesite sample --model run/model.ir.json --data run/data.json
```

Engine stdout is written to:

```text
run/posterior.ndjson
```

Use `--dry-run` to prepare the run directory and print the planned engine command
without executing it.

If a Python file declares more than one model, choose one explicitly:

```bash
bayescycle sample model.py --model LogisticRegression --data data.json -o run/
```

Additional engine flags can be forwarded after `--`:

```bash
bayescycle sample model.py --data data.json -o run/ -- --seed 123
```

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

## Boundary invariant

`bayescycle` may execute Python authoring code and depend on `jaxstanv5`. It may
serialize authoring-side metadata that `jaxstanv5` explicitly exposes into the
run directory, such as optional dimension labels in `dims.json`. It must not
invent model semantics, infer labels from shapes/names, contain inference
algorithms, distribution math, IR evaluation, or sampler logic. Those belong in
Bayesite or `jaxstanv5`.
