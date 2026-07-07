# Adapter: jaxstanv5-inproc-v1

This adapter describes the in-process sampling profile for a bayescycle study.

```text
jaxstanv5 model.py -> bayescycle CLI -> jaxstanv5/BlackJAX sampler -> bayescycle run directory
```

The scientific workflow remains tool-agnostic. This profile swaps only the
sampling backend; the run-directory artifact contract is the same bayescycle v0
contract used by the Bayesite profile.

## Responsibilities

- `jaxstanv5`: model declaration, data binding, BlackJAX NUTS execution, and
  public sampler facts.
- `bayescycle`: Python workflow harness that loads model files, writes run
  directories, invokes the in-process backend, and serializes the posterior
  artifact contract.
- `bayescycle idata`: exporter from run directory to ArviZ DataTree/NetCDF.
- `bayescycle plot`: visualization boundary over exported ArviZ fit files,
  reaching `bayesite-viz` through a single pinned `uvx` source.

## Current bayescycle commands

Prepare a run and sample in-process:

```bash
bayescycle sample models/model.py --data data/data.json -o runs/fit-0001 \
  --backend jaxstanv5 --seed 123 --chains 4 --warmup 1000 --draws 1000
```

Show the plan without sampling:

```bash
bayescycle sample models/model.py --data data/data.json -o runs/fit-0001 \
  --backend jaxstanv5 --seed 123 --chains 4 --warmup 1000 --draws 1000 --show-plan
```

Run prior predictive in-process through the same artifact contract:

```bash
bayescycle prior-predictive models/model.py --data data/inputs.json -o runs/prior-0001 \
  --backend jaxstanv5 --seed 123 --draws 500
```

Run diagnostics and derived exports through the same run directory:

```bash
bayescycle diagnose runs/fit-0001
bayescycle idata runs/fit-0001 -o runs/fit-0001/fit.nc --validate require
bayescycle plot energies runs/fit-0001 -o artifacts/fit-0001-energies.png
```

`simulate`, `recover`, `sbc`, `posterior-check`, and `recover-check` are not yet
served by this profile. If selected with `--backend jaxstanv5`, bayescycle must
return a clear unsupported-profile error rather than silently falling back.

`posterior.ndjson` includes `sample_stats_mode: "per_draw_v2"` and per-draw
`energy`, so energy/BFMI visual checks are available after export.

## Standard run directory

```text
runs/fit-0001/
  model.ir.json
  data.json
  dims.json                 # optional, explicit jaxstanv5 metadata only
  posterior.ndjson
  diagnostics.json          # after diagnose
  prior_predictive.ndjson
  posterior_predictive.ndjson
  simulated_data.json       # Bayesite-backed profile only today
  recovery.json             # Bayesite-backed profile only today
  sbc.json                  # Bayesite-backed profile only today
  recovery_check.json       # Bayesite-backed profile only today
  posterior_check.json      # Bayesite-backed profile only today
  fit.nc                    # optional; produced by `bayescycle idata` for visualization
```

Treat `posterior.ndjson` as raw backend output serialized into the bayescycle
contract. Treat `diagnostics.json` as a machine-readable summary. Treat `fit.nc`
as a derived visualization input.

## Dependency note

This profile requires the in-process dependencies:

```bash
bayescycle[inproc]
```

If they are not installed, use the `jaxstanv5-bayesite-v1` profile or install the
extra before running this adapter.

## Forbidden assumptions

- Do not inspect `jaxstanv5` private APIs.
- Do not parse or depend on BlackJAX internals.
- Do not add plotting/reporting behavior to `src/bayescycle`.
- Do not infer model semantics from file names, shapes, or array labels.
- Do not mutate real-data fit state after changing model/data artifacts without
  invalidating downstream artifacts.
