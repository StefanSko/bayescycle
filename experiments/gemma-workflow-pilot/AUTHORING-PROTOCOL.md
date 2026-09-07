# Matched compact authoring probe — registered before either probe session

## Why a separate probe

The full NumPyro and CLI trials terminated with `length` after near-32K input
contexts, before creating a model or fit. The direct-Bayesjax trial is still
running at registration and is also repeatedly reading inputs. Those results
are retained unchanged. The broad task has not isolated scientific API usability.

This follow-up asks whether the same installed local model can author the fixed
hierarchical model when the task is limited to that phase. It does not count as
full workflow completion, human approval, model criticism or self-directed inference.
The decision to run this narrower probe was made after observing the broad-task
failures; it is exploratory, not a previously registered primary endpoint.

## Comparison

- A-model: NumPyro model function.
- B-model: Bayeswire declarative model, subsequently evaluated by Bayesjax.

No third authoring arm is needed: the CLI and direct backend use the same
Bayeswire declaration interface. This probe does not test CLI orchestration.

Same installed `ollama/gemma4-pi:12b`, unchanged 32K/4096 configuration and existing
decoding settings. Fresh sessions, sequential A-model then B-model; four-minute
budget each. Only read/write/edit tools: authoring needs no shell, installed
packages, data inspection or numerical execution. Each receives the same model
math/data schema and a short API reference for its assigned library. The cards
contain public syntax and a scalar-location example, not a solved hierarchical
model. No raw data, hosted solution or expected posterior result is supplied.
This changes task size, information presentation and tool surface relative to the
broad trial; improvement cannot be attributed to any one of these factors alone.

## Deliverable and checks

Author exactly one model.py. No estimator/report/predictive script. No claim of
human approval or successful fitting. No hosted repairs or hidden source fixes.

The evaluator reviews the actual source against the identical fixture and may
run a separate fixed sampling driver in the shared scientific environment to
check compatibility and posterior agreement. The driver, not the local agent,
executes inference and diagnostics. Keep that distinction explicit in the report.

Check imports/public API validity, exact prior/likelihood structure, non-centered
clinic intercepts, requested vector dimensions and shape-compatible binding.
For correct executable models, use the original float64/4 chains/500 warmup/1000
retained draws, seed 4201, target acceptance 0.9 and depth 10. Compare to the
corresponding hosted main-data result with the same 4*hypot(MCSEs) screen. Review
all-parameter rank Rhat/bulk and tail ESS and divergences. Numerical agreement
alone is not proof of model equivalence. No outcome-dependent retuning.

One attempt per interface is a capability probe, not a reliable comparative error
rate. Both succeeding means the local model can author both with this support;
it would not establish a unique advantage for the custom library. A narrow
interface failure is retained rather than fixed by the evaluator.
