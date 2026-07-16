# bayescycle

Bayescycle is the Python workflow CLI between
[Bayeswire](../bayeswire/) model files and either the Bayesite Rust engine
(default) or the optional [Bayesjax](../bayesjax/) in-process backend. It owns
canonical workflow inputs, run-directory paths and provenance, backend planning,
and replay—not model or sampler semantics.

## Install and sample

```bash
uv tool install bayescycle
bayescycle sample model.py --data data.json -o run/
```

The default installation contains only Bayescycle and stdlib-only Bayeswire.
The first Bayesite-backed command downloads and SHA-256-verifies the pinned
engine release when no suitable executable is on `PATH`.

A completed conditioning run contains at least:

```text
run/
  model.ir.json
  data.json
  posterior.ndjson
  run.json
```

Use `--show-plan` to inspect the backend action without writing a run directory.
When a file has multiple root models, select one with `--model ModelName`.

## Optional Bayesjax backend

```bash
uv tool install 'bayescycle[inproc]'
bayescycle sample model.py --data data.json -o run-jax/ --backend bayesjax
```

The `[inproc]` extra is the only path by which JAX enters Bayescycle's dependency
closure. Unsupported backend capabilities fail explicitly rather than falling
back to another backend.

## Generation and replay

Generate paired natural-scale parameters and complete datasets from fixed
values, the model prior, or a portable posterior source:

```bash
bayescycle generate model.py \
  --design design.json \
  --source fixed \
  --parameters parameters.json \
  --count 100 --seed 123 \
  -o generation-run/
```

Generation runs carry local hash-verified payloads and can be replayed after the
directory is moved and the original model source is unavailable:

```bash
bayescycle replay generation-run/ -o replayed-run/
bayescycle replay generation-run/ -o replayed-run/ --check-only
```

Simulation-gate commands remain available for prior prediction, one-scenario
simulation/recovery, and SBC:

```bash
bayescycle prior-predictive model.py --data inputs.json -o prior-run/
bayescycle simulate model.py --data inputs.json --truth truth.json -o sim-run/
bayescycle recover model.py --scenario scenario.json -o recovery-run/
bayescycle sbc model.py --scenario scenario.json -o sbc-run/
```

## Inspect and visualize

Follow-up commands own their paths and refuse to overwrite existing artifacts:

```bash
bayescycle diagnose run/
bayescycle posterior-predictive run/ --seed 456
bayescycle posterior-check run/ --seed 456
bayescycle recover-check run/ --truth truth.json
```

Export and plot through pinned standalone `uvx` environments:

```bash
bayescycle idata run/          # writes run/fit.nc
bayescycle plot trace run/     # creates fit.nc first when needed
bayescycle warmup              # pre-populates both uvx environments
```

`bayesite-idata` owns NetCDF export and `bayesite-viz` owns plotting. Their
ArviZ stack never enters the default Bayescycle environment.

## Engine management

```bash
bayescycle engine ensure
bayescycle engine path
bayescycle engine info
```

Use `--engine /path/to/bayesite` to override resolution or
`--no-auto-provision`/`BAYESCYCLE_NO_AUTO_PROVISION=1` to prohibit downloads.

## Contracts

Normative artifact formats live in the root [`spec/`](../../spec/), including
canonical data, model/data fingerprints, posterior draws, generated datasets,
and run directories. Bayescycle writes only model metadata exposed by Bayeswire
and sampler facts exposed by the selected backend.

The optional [Bayescycle study skill](../../.agents/skills/bayescycle-study/)
is an agent protocol outside the runtime package.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

See [`AGENTS.md`](AGENTS.md) and [`docs/invariants.md`](docs/invariants.md) for
working and architecture rules.
