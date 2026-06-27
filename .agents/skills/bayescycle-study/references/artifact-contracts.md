# Artifact contracts

Artifacts are files referenced from `state.json`. They are more detailed than the
state snapshot and should be mostly immutable after proposal. If a claim changes,
write a new artifact and supersede or invalidate the old one.

## Common fields in state

```json
{
  "id": "A0001",
  "kind": "estimand_proposal",
  "phase": "estimand",
  "path": "artifacts/0001-estimand-proposals.md",
  "status": "proposed",
  "producer_profile": "bayescycle-study"
}
```

## Artifact kinds

### `estimand_proposal`

Markdown. Contains candidate estimands, required human decisions, assumptions,
and recommended choice. Must not include model fits.

### `estimand`

Markdown. Contains one approved estimand, target population/standardization,
observation status decisions, and interpretation.

### `generative_model`

Markdown and optional diagrams/code. Contains variables, causal/generative
relationships, observation process, missing/censoring treatment, and prior
predictive implications. For causal estimands, it must reference DAG assumptions
or explicitly state why DAG reasoning is not applicable.

### `dag_assumptions`

Markdown. Contains exposure/treatment, outcome, DAG sketch, paths, backdoor
paths, forks/pipes/colliders/descendants, candidate adjustment sets, bad
controls, and unmeasured-confounding risks.

### `adjustment_set_report`

Markdown. Consumes a DAG artifact and justifies the chosen adjustment set for the
approved estimand. Should state which causal paths are intentionally left open.

### `bad_controls_report`

Markdown. Lists mediators, colliders, descendants of colliders, selection nodes,
or proxies that should not be conditioned on for the approved estimand.

### `estimator_plan`

Markdown. Maps estimand to posterior quantity or other estimator. Defines
simulation/recovery tests and pass/fail criteria.

### `model_spec`

Concrete model source, e.g. `models/adoption_hazard.py`, or serialized IR. It is
an implementation artifact, not the scientific generative model itself.

### `data_snapshot`

Immutable data file or manifest used by a run. Should include source, transform
script if any, and hash if available.

### `prior_predictive_run`

Run directory containing `prior_predictive.ndjson` produced by
`bayescycle prior-predictive`.

### `recovery_run`

Run directory or report showing fake-data recovery for known parameters or the
approved estimand. Prefer bayescycle-owned artifacts such as
`simulated_data.json`, `recovery_check.json`, `recovery.json`, or `sbc.json`.

### `real_fit_run`

Run directory for conditioning on real observations. Must not be approved before
simulation/recovery approval unless the state explicitly records a waiver.

### `diagnostics_report`

Machine or human summary of sampler and workflow diagnostics. Raw NDJSON is not
a diagnostics report until summarized.

### `schema_visual_audit`

Visual audit of variables, missingness, event status, censoring, or grouping
structure used to clarify the estimand and observation process.

### `dag_visualization`

DAG, generative graph, or observation-process diagram associated with the
approved generative model.

### `prior_predictive_visual_report`

Prior predictive visualizations and human-readable interpretation. Required for
most non-trivial model approvals before real-data fitting.

### `recovery_visual_report`

Fake-data recovery plots, calibration/rank/coverage visuals, and pass/fail
interpretation.

### `fit_diagnostic_visual_report`

bayesite-viz diagnostic plots such as trace, rank, autocorrelation, forest,
posterior, pair, or energies when available.

### `posterior_estimand_visual_report`

Visualization of the approved estimand posterior, including uncertainty and any
standardization/contrast implied by the estimand.

### `posterior_predictive_report`

Posterior predictive checks and plots. Should state what model failures would
matter scientifically.

### `sensitivity_visual_report`

Visual comparison across model variants, priors, censoring assumptions, or
estimand definitions.

### `critique`

Final criticism artifact for a cycle. May recommend revision, sensitivity
analysis, or stopping.

## Invalidation rules

Changing an approved artifact invalidates downstream artifacts:

- estimand change invalidates generative model and later artifacts
- generative model change invalidates estimator plan and later artifacts
- DAG/adjustment-set change invalidates estimator plan and later artifacts for
  causal estimands
- estimator plan change invalidates simulation, fit, and critique
- model code/data change after recovery invalidates recovery-to-fit continuity
- failed diagnostics block fit approval unless explicitly waived
- missing required visualization artifacts block phase approval unless explicitly
  waived
