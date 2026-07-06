# Diagnostics policy

Diagnostics are evidence for workflow and computation quality. They are not the
scientific state itself. See `mcmc-diagnostics.md` for the light intuition behind
these checks.

## Raw telemetry

Bayesite may emit NDJSON such as:

```text
runs/fit-0001/posterior.ndjson
runs/fit-0001/prior_predictive.ndjson
runs/fit-0001/posterior_predictive.ndjson
```

Keep raw NDJSON unchanged. Do not paste large streams into agent context.
Summarize or filter first.

## Machine summaries

Expected summaries, when available:

```text
runs/fit-0001/diagnostics.json
runs/fit-0001/posterior_check.json
runs/fit-0001/recovery_check.json
runs/fit-0001/recovery.json
runs/fit-0001/sbc.json
runs/fit-0001/diagnostics.md
```

`diagnostics.json`, `posterior_check.json`, `recovery_check.json`,
`recovery.json`, and `sbc.json` are preferred machine-readable inputs for a
fresh agent when relevant. They are factual reports, not automatic approval
verdicts. `diagnostics.md` is a human-facing explanation.

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
- required `bayescycle plot` visual report is planned but not produced
- diagnostics use unregistered run directories
- raw NDJSON exists but no summarized diagnostics artifact is registered
- a `data_snapshot` is used by a fit without an approved `data_audit` artifact
  or a recorded waiver

## Computational checks

If reported, flag as errors:

- chain failures
- nonzero divergences for final inference unless explicitly justified
- R-hat above the project threshold
- ESS below the project threshold for important parameters or the approved estimand
- max treedepth saturation that changes posterior interpretation
- warmup/adaptation draws used as posterior draws
- missing or inconsistent seeds for recovery runs

Thresholds live in `state.thresholds` (`rhat_max`, `ess_bulk_min`,
`ess_tail_min`, `divergences_max`). If they are not in state, ask the human to
choose defaults, record them as a state patch, and only then treat borderline
diagnostics as pass/fail. Do not rely only on raw-parameter diagnostics
when the reported result is a derived estimand, contrast, or prediction.

## SBC interpretation

`sbc.json` reports rank statistics and histograms only; the engine deliberately
emits no uniformity verdict or p-value. Do not invent one. Instead:

- produce a rank-histogram visual (`recovery_visual_report`) for the SBC run
- read the standard shapes: roughly uniform ranks are consistent with a
  well-calibrated estimator; a U shape suggests an overdispersed posterior; a
  peaked (inverted-U) shape suggests an underdispersed posterior; a monotone
  slope suggests bias
- treat a gross, visually unambiguous deviation as an `error`
- treat a borderline or ambiguous histogram as `warning` and ask for a human
  gate decision rather than deciding alone; record the outcome as a decision
  or waiver

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
sampler pathologies, while `bayescycle plot` plots expose the failures that
humans are likely to catch visually. Do not use one as a silent substitute for
the other unless the human records a waiver.
