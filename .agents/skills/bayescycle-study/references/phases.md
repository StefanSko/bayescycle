# Bayescycle study phases

Each invocation performs one phase and stops at a gate. A later phase may read
approved artifacts from earlier phases, but must not rely on unapproved scratch
work.

## 1. estimand

Goal: state what the study is trying to learn.

Allowed work:

- inspect data schema and metadata
- propose candidate scientific questions and estimands
- identify whether the estimand is causal, descriptive, predictive, or decision-oriented
- for causal candidates, identify exposure/treatment `X`, outcome `Y`, and effect type
- identify human decisions needed to define observation status and scope

Forbidden work:

- fitting models
- choosing priors as if the model were final
- reporting empirical effects as answers

Visualization expectation: schema/missingness/event-status visuals or a stated
reason they are not applicable. These visuals clarify measurement; they do not
answer the scientific question before the estimand is approved.

Exit criterion: one approved estimand, causal/descriptive status, and any
adjustment/standardization target needed to interpret it.

## 2. generative_model

Goal: describe how the data could arise under the scientific assumptions.

Allowed work:

- propose DAGs or structural relationships
- identify observed, latent, missing, censored, and future variables
- classify forks, pipes, colliders, descendants, and bad controls for causal estimands
- list backdoor paths and candidate adjustment sets when the estimand is causal
- propose priors through prior predictive implications

Visualization expectation: DAG/generative graph and observation-process visual
when the model is causal or involves missingness, censoring, or competing events.

Exit criterion: approved generative story, variables, observation process, DAG
assumptions when causal, and scientific assumptions.

## 3. estimator_plan

Goal: connect the estimand to a computational estimator implied by the
generative model.

Allowed work:

- specify likelihood/conditioning structure
- specify posterior summaries that target the estimand
- for causal estimands, justify the chosen adjustment set using the DAG
- state which mediators, colliders, descendants, or proxies are intentionally excluded
- define fake-data recovery criteria

Visualization expectation: planned recovery and posterior predictive checks are
listed, with pass/fail criteria tied to the estimand, DAG assumptions, and
scientific failures.

Exit criterion: approved estimator plan with explicit estimand mapping and, for
causal estimands, an adjustment-set justification.

## 4. simulation

Goal: test plumbing before real-data inference.

Allowed work:

- simulate fake data from the approved generative model using `bayescycle simulate`
- for causal estimands, simulate according to the approved DAG skeleton
- run prior predictive checks using `bayescycle prior-predictive`
- run recovery checks for known parameters/estimands using `bayescycle recover-check`
- run single-scenario recovery or SBC reports with `bayescycle recover` / `bayescycle sbc`
- compare against intentionally wrong estimators or bad-control adjustment when useful

Visualization expectation: prior predictive and recovery visual reports are
produced or explicitly waived by the human.

Exit criterion: approved simulation/recovery artifact. If recovery fails, move to
revision instead of fit.

## 5. fit

Goal: condition on real observations using the approved estimator.

Allowed work:

- prepare immutable data snapshots
- run `bayescycle sample`
- run diagnostics before posterior interpretation
- check diagnostics for the approved estimand or derived quantity when practical
- summarize posterior estimand only if diagnostics pass or failures are stated

Visualization expectation: bayesite-viz diagnostic plots and posterior estimand
visuals are produced or explicitly waived by the human.

Exit criterion: approved real-fit artifact and diagnostics status.

## 6. critique

Goal: compare fitted model back against reality and the scientific question.

Allowed work:

- posterior predictive checks
- residual and calibration checks
- sensitivity checks, including unmeasured-confounding sensitivity when relevant
- diagnostic review
- identify model failures and revised assumptions
- revisit DAG assumptions, bad controls, and selection/censoring paths

Visualization expectation: posterior predictive and sensitivity visuals are
reviewed by the human, with failures tied back to the scientific question.

Exit criterion: approved criticism. If revision is needed, open a revision gate.

## 7. revision

Goal: change an earlier approved artifact deliberately.

Allowed work:

- supersede decisions
- invalidate downstream artifacts
- move phase backward

Exit criterion: approved invalidation plan and new target phase.
