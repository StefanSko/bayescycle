# Adapter: jaxstanv5-bayesite-v1

This adapter describes the current concrete implementation profile for a
bayescycle study.

```text
jaxstanv5 model.py -> bayescycle CLI -> Bayesite engine -> run directory
```

The scientific workflow remains tool-agnostic. This adapter is swappable.

## Responsibilities

- `jaxstanv5`: model declaration and IR serialization boundary.
- `bayescycle`: Python workflow harness that loads model files, writes run
  directories, and invokes Bayesite.
- `bayesite`: engine that consumes IR/data and emits draws/diagnostics.
- `bayesite-viz`: first-class visualization boundary over exported ArviZ fit
  files derived from run artifacts.

## Current bayescycle commands

Prepare a run and sample:

```bash
bayescycle sample models/model.py --data data/data.json -o runs/fit-0001 \
  --seed 123 --chains 4 --warmup 1000 --draws 1000
```

Show the plan without invoking Bayesite:

```bash
bayescycle sample models/model.py --data data/data.json -o runs/fit-0001 \
  --seed 123 --chains 4 --warmup 1000 --draws 1000 --show-plan
```

Run diagnostics for an existing run directory:

```bash
bayescycle diagnose runs/fit-0001
```

Run simulation-gate commands through bayescycle-owned run directories:

```bash
bayescycle prior-predictive models/model.py --data data/inputs.json -o runs/prior-0001 \
  --seed 123 --draws 500
bayescycle simulate models/model.py --data data/inputs.json --truth data/truth.json \
  -o runs/sim-0001 --seed 1
bayescycle recover models/model.py --scenario scenarios/recover.json -o runs/recover-0001
bayescycle sbc models/model.py --scenario scenarios/sbc.json -o runs/sbc-0001 --replicates 100
```

Run diagnostics and follow-up checks for an existing run directory:

```bash
bayescycle diagnose runs/fit-0001
bayescycle posterior-predictive runs/fit-0001 --seed 456
bayescycle posterior-check runs/fit-0001 --seed 456
bayescycle recover-check runs/fit-0001 --truth data/truth.json --interval 0.8
```

## Standard run directory

A bayescycle sample run owns:

```text
runs/fit-0001/
  model.ir.json
  data.json
  dims.json                 # optional, explicit jaxstanv5 metadata only
  posterior.ndjson
  diagnostics.json          # after diagnose
  prior_predictive.ndjson
  posterior_predictive.ndjson
  simulated_data.json
  recovery.json
  sbc.json
  recovery_check.json
  posterior_check.json
  fit.nc                    # optional; produced by bayesite-idata for visualization
```

Treat `posterior.ndjson` as raw engine output. Treat `diagnostics.json` as a
machine-readable summary. Treat `fit.nc` as a derived visualization input.
Register these in `state.json` only through run/artifact records, not by copying
large contents into state.

## Visualization path

For a completed run, use bayesite-viz through its exporter boundary:

```bash
bayesite-idata runs/fit-0001 -o runs/fit-0001/fit.nc --validate require
bayesite-viz trace runs/fit-0001/fit.nc -o artifacts/fit-0001-trace.png
bayesite-viz rank runs/fit-0001/fit.nc -o artifacts/fit-0001-rank.png
bayesite-viz posterior runs/fit-0001/fit.nc --kind hist -o artifacts/fit-0001-posterior.png
bayesite-viz ppc runs/fit-0001/fit.nc --kind dist -o artifacts/fit-0001-ppc-dist.png
```

See `adapters/bayesite-viz.md` for the full visual vocabulary. Visual artifacts
are required at simulation, fit, and critique gates unless explicitly waived.

## Model source

Keep concrete model code under the study directory, for example:

```text
models/adoption_hazard.py
```

Register it as an artifact of kind `model_spec`. The model source is an
implementation artifact; the scientific generative model belongs in a separate
`generative_model` artifact.

## Data snapshots

Prepare immutable data snapshots under the study directory, for example:

```text
data/austin-cats-cycle1.json
```

Register the snapshot as `data_snapshot`. If a preprocessing script is used,
register it or mention it in the data snapshot artifact.

## Simulation and recovery

Use the first-class bayescycle commands for prior predictive, fake-data
simulation, recovery checks, single-scenario recovery, and SBC. Direct Bayesite
calls are not needed for the simulation gate in this profile. Record the exact
bayescycle command in the artifact and do not treat the output as approved until
the human gate accepts it.

`simulate`, `recover`, and `sbc` run on the Bayesite backend only in this
profile; requesting them with `--backend jaxstanv5` returns an explicit
unsupported-profile error.

## Scenario and targets file schemas

These schemas are easy to get wrong and the engine errors are terse; use the
shapes below verbatim.

`recover` scenario files require the version marker `recover_scenario`:

```json
{
  "recover_scenario": "v0-provisional",
  "data": { "x": [1.0, 2.0] },
  "sample": { "chains": 4, "warmup": 400, "draws": 500, "target_accept": 0.9 },
  "seed": 7
}
```

`sbc` scenarios are the same with `"sbc_scenario": "v0-provisional"` and an
optional `"replicates"`. The `sample` object accepts only
`chains|warmup|draws|max_treedepth|target_accept`.

`recover-check --targets` files must be a wrapped list, not a flat map:

```json
{
  "targets": [
    { "name": "alpha", "truth": "alpha", "posterior": "alpha" }
  ]
}
```

Omitting `--targets` auto-maps each truth name to the matching posterior
parameter, which is the simplest path when names line up.

## Operational notes

- Run every `bayescycle` invocation from the bayescycle project directory with
  absolute paths for study files: `uv run` extras (for example
  `--extra inproc`) have no effect outside the project and the entry point may
  not resolve from a study or run directory.
- Set `PYTHONDONTWRITEBYTECODE=1` when loading study model files so
  `__pycache__/` does not pollute study trees.
- bayesite-viz requires Python >= 3.13 and cannot share the bayescycle (3.12)
  virtualenv; run it from its own checkout or environment, for example
  `uv run --no-project --with <path-to-bayesite-viz> --python 3.13 -- bayesite-viz ...`.

## Public contracts used

- `bayescycle sample ...`
- `bayescycle prior-predictive ...`
- `bayescycle simulate ...`
- `bayescycle recover ...`
- `bayescycle sbc ...`
- `bayescycle diagnose ...`
- `bayescycle posterior-predictive ...`
- `bayescycle posterior-check ...`
- `bayescycle recover-check ...`
- run-directory files documented by `bayescycle`
- `bayesite-idata <run-dir> -o <fit.nc>`
- `bayesite-viz <verb> <fit.nc> -o <artifact>`

## Forbidden assumptions

- Do not inspect `jaxstanv5` private APIs.
- Do not parse or depend on Bayesite private runtime internals.
- Do not infer model semantics from file names, shapes, or array labels.
- Do not treat Bayesite stdout/stderr as posterior semantics unless summarized by
  an explicit diagnostics artifact.
- Do not assume `bayesite-viz` owns run-directory discovery; use `bayesite-idata`
  as the exporter boundary.

## Boundary rules

- If this adapter conflicts with a tool repository's `AGENTS.md` or invariants,
  the tool repository wins and the agent must stop and ask.
- Do not add plotting/reporting behavior to `src/bayescycle` for this workflow.
- Do not mutate real-data fit state after changing model/data artifacts without
  invalidating downstream artifacts.
