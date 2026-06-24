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

Dry-run without invoking Bayesite:

```bash
bayescycle sample models/model.py --data data/data.json -o runs/fit-0001 \
  --seed 123 --chains 4 --warmup 1000 --draws 1000 --dry-run
```

Run diagnostics for an existing run directory:

```bash
bayescycle diagnose runs/fit-0001
```

Generate posterior predictive draws when supported by the engine profile:

```bash
bayescycle posterior-predictive runs/fit-0001 --seed 456
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
  posterior_predictive.ndjson
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

Prefer the bayescycle CLI for runs that can be expressed through current
commands. If the Bayesite binary exposes additional commands such as
`prior-predictive`, `recover`, or `sbc` before bayescycle wraps them, direct
Bayesite use is allowed as an exploratory implementation detail. Record the
exact command in the artifact and do not treat the output as approved until the
human gate accepts it.

## Public contracts used

- `bayescycle sample ...`
- `bayescycle diagnose ...`
- `bayescycle posterior-predictive ...`
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
