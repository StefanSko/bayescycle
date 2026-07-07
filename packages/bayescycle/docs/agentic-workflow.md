# Agentic study workflow

This repository contains an optional agent skill for running a gated Bayesian
study workflow. It lives outside `src/bayescycle` because the Python package
remains a narrow CLI harness from `bayeswire` model files to Bayesite engine run
directories.

## Boundary

Core runtime:

```text
src/bayescycle/ -> deterministic CLI orchestration
```

Agent protocol:

```text
.agents/skills/bayescycle-study/ -> human-gated study workflow instructions
```

The skill may use the `bayescycle` CLI, but the CLI does not know about chats,
phase gates, decisions, or study notebooks.

The skill composes independent tools through public contracts only. If the skill
conflicts with a tool repository's `AGENTS.md`, invariants, or documented public
contract, the tool repository wins and the agent should stop for human review.

## Study layout

A study keeps canonical state in JSON and rich prose in artifacts:

```text
study/
  state.json
  events.jsonl
  artifacts/
  patches/
  runs/
```

- `state.json`: current machine-readable snapshot
- `events.jsonl`: append-only audit log
- `artifacts/`: phase reports, estimands, model criticism, figures
- `patches/`: proposed JSON Patch state transitions
- `runs/`: bayescycle/Bayesite run directories, usually not committed

## Workflow

The skill runs one phase at a time:

1. estimand
2. generative model
3. estimator plan
4. simulation/recovery
5. real fit
6. critique
7. revision

Each phase starts by re-reading `state.json`; prior conversation is
non-authoritative unless recorded in state or approved artifacts.

## DAGs and generative models

For causal questions, the `generative_model` phase uses DAG reasoning as the
structural scaffold: identify exposure/treatment `X`, outcome `Y`, paths,
backdoor paths, candidate adjustment sets, mediators, colliders, descendants,
and unmeasured-confounding risks. The DAG is not the full model; it is followed
by probability distributions, priors, measurement, censoring, and observation
status.

The `estimator_plan` phase then consumes the DAG to justify adjustment and to
warn against bad controls and Table 2 interpretations.

## Toolchain profiles

The current concrete adapters are `bayesjax-bayesite-v1` and
`bayesjax-inproc-v1`:

```text
bayeswire model.py -> bayescycle CLI -> Bayesite engine -> run directory
bayeswire model.py -> bayescycle CLI -> bayesjax in-process backend -> run directory
```

The simulation gate should use first-class `bayescycle` commands rather than
calling the engine directly:

```bash
bayescycle prior-predictive models/model.py --data data/inputs.json -o runs/prior-0001
bayescycle simulate models/model.py --data data/inputs.json --truth data/truth.json -o runs/sim-0001
bayescycle sample models/model.py --data runs/sim-0001/simulated_data.json -o runs/recover-fit-0001
bayescycle recover-check runs/recover-fit-0001 --truth data/truth.json
```

`runs/sim-0001/simulated_data.json` is a canonical `bayescycle.data.json.v1`
artifact, so a recovery fit can use the same file with the same backend or an
explicit different backend through adapter materialization.

The scientific state tracks abstract artifact kinds, so adapters can swap a
backend implementation without rewriting the workflow protocol.

## Visualization

Visualization is first-class in the study protocol. The current visualization
adapter is [`bayesite-viz`](https://github.com/StefanSko/bayesite-viz):

```text
bayescycle/Bayesite run directory -> bayesite-idata -> fit.nc -> bayesite-viz -> image
```

Every phase gate should state the visualization status:

```text
not_applicable | planned | produced | reviewed | blocked
```

Visual outputs are registered as artifacts such as
`prior_predictive_visual_report`, `recovery_visual_report`,
`fit_diagnostic_visual_report`, and `posterior_predictive_report`. They are
evidence for human review, not canonical state.
