# bayescycle

`bayescycle` is the Python workflow CLI that connects the `bayeswire` authoring
eDSL to backend engine runs and owns the neutral run-directory contract.

It is intentionally glue:

```text
model.py -> bayeswire ModelMeta -> bayeswire IR JSON -> bayesite engine -> run/
```

## Package split

- **bayeswire**: Python model declaration eDSL, the `bayeswire_ir` wire format,
  its normative spec, and the conformance corpus. Stdlib only.
- **bayesite**: Rust engine and IR-level CLI. It accepts IR plus data and writes
  draws. It stays dependency-free and WASM-clean.
- **jaxstanv5**: JAX/BlackJAX sampling backend for bayeswire models. Optional
  here; installed via the `[inproc]` extra.
- **bayescycle**: Python workflow harness. It executes a Python model file,
  serializes IR, invokes a backend, and owns run-directory ergonomics and the
  run artifact contract.

## Trust surface of the default path

With the default `--backend bayesite`, running a workflow executes `model.py`
in a Python environment containing exactly one stdlib-only package
(`bayeswire`, plus `bayescycle` itself) and invokes one auditable
zero-dependency Rust binary. No JAX on the default path, ever. That is the
honest version of the original zero-dependency claim: not "no supply-chain
danger", but every step of the default path is small enough to audit, pinned
enough to reproduce, and deterministic enough to replay.

## Initial CLI

```bash
bayescycle sample model.py --data data.json -o run/
```

By default this prepares:

```text
run/model.ir.json
run/data.json       # canonical bayescycle.data.json.v1 snapshot
run/manifest.json   # artifact format manifest
run/dims.json       # optional; written only when the model declares dimension metadata
```

and invokes the Bayesite adapter, which passes the canonical `run/data.json`
straight to the engine (the engine parses the `bayescycle.data.json.v1` wrapper
natively, so both backends fingerprint the same bytes; see
`docs/posterior-draws-v0.md`):

```bash
bayesite sample --model run/model.ir.json --data run/data.json --out run/posterior.ndjson
```

The engine is asked to write draws to:

```text
run/posterior.ndjson
```

Use `--show-plan` to print the planned backend action without creating or rewriting
run-directory artifacts. Plan output records both the selected backend and the
integration mode: Bayesite is an `external-command` adapter, while jaxstanv5 is
an `in-process-python` adapter.

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
applies to `--backend bayesite`. Before execution, Bayescycle preflights the
selected Bayesite binary for existence, executability, and required subcommand
support so stale engines fail before run-directory writes. Install the
in-process dependencies with `bayescycle[inproc]`.

Additional engine flags can be forwarded after `--`. The engine `--out` flag is reserved
so the run directory always contains `run/posterior.ndjson`.

```bash
bayescycle sample model.py --data data.json -o run/ -- --experimental-engine-flag
```

The run-directory contract is documented in [`docs/run-directory-v0.md`](docs/run-directory-v0.md),
[`docs/canonical-data-artifacts.md`](docs/canonical-data-artifacts.md), and
[`docs/posterior-draws-v0.md`](docs/posterior-draws-v0.md). Bayesite is the
initial backend that emits this contract, but the contract is owned by
`bayescycle` rather than by a specific sampler.

An executed mixed-backend walkthrough is available as
[`docs/mixed-backend-workflow.html`](docs/mixed-backend-workflow.html). It is
generated from a real, intentionally non-linear run: a wide-prior model is
rejected at the prior-predictive gate and respecified, then an explicit mixed
TOML plan drives the canonical data artifact handoff from a Bayesite simulation
to a jaxstanv5 in-process recovery fit checked back against truth by Bayesite.

A fully worked end-to-end walkthrough of the complete workflow is available as a
self-contained page:
[`docs/workflow-walkthrough.html`](docs/workflow-walkthrough.html). It is also
non-linear: a first model with wide priors fails the prior-predictive gate and is
respecified before the simulation gate (prior predictive, simulate, recover, sbc)
and the real fit (diagnostics, posterior check, ArviZ visualization) proceed, with
the filesystem effects and append-only `run.json` provenance of each command
annotated.

Both pages, plus the SQLite run index, are regenerated from scratch by
[`docs/build-walkthroughs.sh`](docs/build-walkthroughs.sh), which drives the CLI
through both workflows and then runs
[`docs/workflow-walkthrough.py`](docs/workflow-walkthrough.py),
[`docs/mixed-backend-workflow.py`](docs/mixed-backend-workflow.py), and
[`docs/run-provenance-db.py`](docs/run-provenance-db.py). The append-only
`run.json` (`bayescycle.run.v1`) records serialize directly into
[`docs/walkthrough-runs.sqlite`](docs/walkthrough-runs.sqlite) (tables `runs`,
`run_inputs`, `run_outputs`, and a derived `workflow_edges` graph that captures
the cross-backend handoff). Difficulties hit while regenerating these are logged
in [`docs/walkthrough-difficulties.md`](docs/walkthrough-difficulties.md).

Simulation-gate commands are also first-class and own their run-directory output paths:

```bash
bayescycle prior-predictive model.py --data inputs.json -o runs/prior-0001 --seed 123 --draws 500
bayescycle simulate model.py --data inputs.json --truth truth.json -o runs/sim-0001 --seed 1
bayescycle recover model.py --scenario scenario.json -o runs/recover-0001
bayescycle sbc model.py --scenario scenario.json -o runs/sbc-0001 --replicates 100
```

Run directories are append-only discovery records. Commands refuse to reuse a
non-empty output directory or overwrite an existing follow-up artifact; use a new
run id for a new attempt. Fresh model-level runs also write `run.json` with
narrow provenance metadata for future indexing.

`prior-predictive` supports both backends. `simulate`, `recover`, and `sbc` are
currently Bayesite-backed; selecting `--backend jaxstanv5` returns a clear
unsupported-profile error rather than falling back silently.

`simulate` writes `runs/sim-0001/simulated_data.json` as canonical
`bayescycle.data.json.v1`, so a downstream sample command can use it with the
same backend or with `--backend jaxstanv5` through the adapter boundary. For
multi-stage intent, validate the backend plan before creating run directories:

```bash
bayescycle workflow-plan --backend bayesite
bayescycle workflow-plan --config workflow.toml
```

Run-directory follow-up phases can be orchestrated without repeating owned paths:

```bash
bayescycle diagnose run/
bayescycle posterior-predictive run/ --seed 456
bayescycle posterior-check run/ --seed 456
bayescycle recover-check run/ --truth truth.json --targets targets.json --interval 0.8
```

These invoke Bayesite with `run/posterior.ndjson` as the fit input and write:

```text
run/diagnostics.json
run/posterior_predictive.ndjson
run/posterior_check.json
run/recovery_check.json
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

`bayescycle` may execute Python authoring code and depends on `bayeswire` for
authoring semantics and IR serialization. It may serialize authoring-side
metadata that `bayeswire` explicitly exposes into the run directory, such as
optional dimension labels in `dims.json`, and sampler facts explicitly exposed
by the selected backend. It must not invent model semantics, infer labels from
shapes/names, contain inference algorithms, distribution math, IR evaluation,
or sampler logic. Authoring semantics belong to `bayeswire`; sampler facts
belong to `jaxstanv5` or Bayesite.
