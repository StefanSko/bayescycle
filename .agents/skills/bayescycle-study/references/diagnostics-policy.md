# Diagnostics policy

Diagnostics are evidence for workflow and computation quality. They are not the
scientific state itself.

## Raw telemetry

Bayesite may emit NDJSON such as:

```text
runs/fit-0001/posterior.ndjson
runs/fit-0001/posterior_predictive.ndjson
```

Keep raw NDJSON unchanged. Do not paste large streams into agent context.
Summarize or filter first.

## Machine summaries

Expected summaries, when available:

```text
runs/fit-0001/diagnostics.json
runs/fit-0001/diagnostics.md
```

`diagnostics.json` is the preferred machine-readable input for a fresh agent.
`diagnostics.md` is a human-facing explanation.

## Workflow checks

Flag as errors:

- real-data fit exists before approved simulation/recovery, unless a waiver is
  recorded in state
- approved artifact appears in `invalidated`
- current phase advances while blocking questions remain open
- model/data artifact changes after recovery without invalidating recovery
- missing diagnostics for a run marked as fit-complete

Flag as warnings:

- posterior predictive checks missing before critique
- required bayesite-viz visual report is planned but not produced
- diagnostics use unregistered run directories
- raw NDJSON exists but no summarized diagnostics artifact is registered

## Computational checks

If reported, flag as errors:

- chain failures
- nonzero divergences for final inference unless explicitly justified
- R-hat above the project threshold
- ESS below the project threshold
- max treedepth saturation that changes posterior interpretation
- missing or inconsistent seeds for recovery runs

If thresholds are not in state, ask the human to choose defaults before treating
borderline diagnostics as pass/fail.

## Severity levels

Use:

- `info`: useful context, does not block
- `warning`: needs review, may not block
- `error`: blocks approval until resolved or waived

A waiver must be a recorded decision with consequences.

## Agent behavior

A diagnostic pass may propose state patches. It must not silently approve a fit
or critique. When in doubt, write a diagnostics artifact and ask for a gate
decision.

Diagnostics and visualization are complementary: `diagnostics.json` can flag
sampler pathologies, while bayesite-viz plots expose the failures that humans are
likely to catch visually. Do not use one as a silent substitute for the other
unless the human records a waiver.
