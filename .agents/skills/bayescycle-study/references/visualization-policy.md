# Visualization policy

Visualization is a first-class part of the study workflow. It is not merely a
final reporting step. Visual artifacts are the human-facing interface for model
criticism, prior checking, recovery checking, and scientific sense-making.

## Principle

Every phase gate should explicitly answer:

```text
What visual evidence should a human inspect before approving this phase?
```

The answer may be `not applicable`, but it must be stated.

## Phase expectations

### Estimand

Useful visualizations:

- variable/schema audit
- outcome/event status counts
- missingness/censoring overview
- estimand card or contrast sketch

Do not use outcome visualizations to answer the question before the estimand is
approved. Use them to clarify measurement and observation status.

### Generative model

Required when applicable:

- DAG or generative graph
- observation process diagram for missing/censored/competing events
- prior predictive visualization plan

### Estimator plan

Required when applicable:

- diagram mapping generative quantities to the estimand
- planned recovery plots and pass/fail criteria
- planned posterior predictive checks tied to scientific failures

### Simulation

Expected visualizations:

- prior predictive plots
- fake-data recovery plots, such as true-vs-estimated estimand
- rank, calibration, or coverage visuals when available
- comparison against known-bad estimator if useful

Simulation should not pass merely because a sampler ran. The human should be
able to see whether simulated data and recovery behavior make sense.

### Fit

Expected visualizations:

- sampler diagnostic plots when available, such as trace/rank/energy
- posterior distribution of the approved estimand
- posterior uncertainty summaries relevant to decisions

A real-data fit should not be approved if required diagnostic visuals are
missing, unless a human records a waiver.

### Critique

Expected visualizations:

- posterior predictive checks
- stratified posterior predictive checks for scientifically important groups
- residual/calibration/error visualizations when appropriate
- sensitivity comparison plots across model variants

Critique visuals should identify where the model fails, not merely decorate the
result.

## Artifact conventions

Register visual outputs as artifacts. Do not store image blobs or large rendered
reports in `state.json`.

Example:

```json
{
  "id": "A0012",
  "kind": "prior_predictive_visual_report",
  "phase": "simulation",
  "path": "artifacts/0012-prior-predictive.md",
  "status": "proposed",
  "producer_profile": "bayesite-viz"
}
```

Recommended visual artifact kinds:

- `schema_visual_audit`
- `estimand_visual_card`
- `dag_visualization`
- `observation_process_visualization`
- `prior_predictive_visual_report`
- `recovery_visual_report`
- `fit_diagnostic_visual_report`
- `posterior_estimand_visual_report`
- `posterior_predictive_visual_report`
- `sensitivity_visual_report`

## Bayesite-viz role

When `state.toolchain.visualization` is `bayesite-viz`, prefer bayesite-viz for
standard diagnostics and posterior/predictive visual reports. If the local
`bayesite-viz` CLI/API is unavailable or its command surface is unknown:

1. Try to inspect `bayesite-viz --help` or project documentation.
2. If unavailable, write a visualization plan artifact instead of inventing a
   command.
3. Ask the human whether to proceed with an alternate plotting script.
4. Record the exact command or script used in the artifact.

## Human review

A phase report should include a visualization status:

- `not_applicable`: no meaningful visual check for this phase
- `planned`: visual outputs are specified but not produced yet
- `produced`: visual outputs exist but need human review
- `reviewed`: human reviewed them and accepted the phase implications
- `blocked`: missing or failed visual evidence blocks approval

Visual review is a scientific gate. The agent may summarize and recommend, but
the human approves or rejects the interpretation.
