# bayescycle

`bayescycle` is the Python workflow CLI that connects the `jaxstanv5` authoring
eDSL to backend engine runs and owns the neutral run-directory contract.

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
  serializes IR, invokes a backend, and owns run-directory ergonomics and the
  run artifact contract.

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

The default backend invokes the Bayesite executable. An in-process JAX backend is
available when the optional dependencies are installed:

```bash
bayescycle sample model.py --data data.json -o run/ --backend jaxstanv5 \
  --seed 123 --chains 2 --warmup 100 --draws 100 --max-treedepth 10 --target-accept 0.8
```

`--engine /path/to/bayesite` remains the Bayesite executable override and only
applies to `--backend bayesite`. Install the in-process dependencies with
`bayescycle[inproc]`.

Additional engine flags can be forwarded after `--`. The engine `--out` flag is reserved
so the run directory always contains `run/posterior.ndjson`.

```bash
bayescycle sample model.py --data data.json -o run/ -- --experimental-engine-flag
```

The run-directory contract is documented in [`docs/run-directory-v0.md`](docs/run-directory-v0.md)
and [`docs/posterior-draws-v0.md`](docs/posterior-draws-v0.md). Bayesite is the
initial backend that emits this contract, but the contract is owned by
`bayescycle` rather than by a specific sampler.

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
telemetry rather than scientific state, uses DAGs as scaffolding for causal
generative models, and makes visualization a required phase-gate concern through
`bayesite-viz` artifacts.

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
run directory, such as optional dimension labels in `dims.json`, and sampler
facts explicitly exposed by the selected backend. It must not invent model
semantics, infer labels from shapes/names, contain inference algorithms,
distribution math, IR evaluation, or sampler logic. Those belong in Bayesite or
`jaxstanv5`.
