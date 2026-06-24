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
bayesite sample --model run/model.ir.json --data run/data.json --out run/posterior.ndjson
```

The engine is asked to write draws to:

```text
run/posterior.ndjson
```

Use `--dry-run` to prepare the run directory and print the planned engine command
without executing it.

If a Python file declares more than one model, choose one explicitly:

```bash
bayescycle sample model.py --model LogisticRegression --data data.json -o run/
```

Common sampler settings are first-class workflow options:

```bash
bayescycle sample model.py --data data.json -o run/ --seed 123 --chains 2 --warmup 100 --draws 100
```

Additional engine flags can be forwarded after `--`. The engine `--out` flag is reserved
so the run directory always contains `run/posterior.ndjson`.

```bash
bayescycle sample model.py --data data.json -o run/ -- --experimental-engine-flag
```

Run-directory follow-up phases can be orchestrated without repeating owned paths:

```bash
bayescycle diagnose run/
bayescycle posterior-predictive run/ --seed 456
```

These invoke Bayesite with `run/posterior.ndjson` as the fit input and write:

```text
run/diagnostics.json
run/posterior_predictive.ndjson
```

## Agentic study workflow

The optional `.agents/skills/bayescycle-study/` skill defines a lightweight,
human-gated Bayesian study protocol. It keeps canonical study state in JSON,
records proposed state transitions as patches, treats Bayesite NDJSON as
telemetry rather than scientific state, and makes visualization a required
phase-gate concern through `bayesite-viz` artifacts.

The skill is intentionally outside `src/bayescycle`: it may use the CLI, but the
Python package remains a narrow deterministic workflow harness. See
[`docs/agentic-workflow.md`](docs/agentic-workflow.md).

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
