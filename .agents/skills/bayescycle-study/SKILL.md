---
name: bayescycle-study
description: Runs one gated phase of an agentic Bayesian study workflow using JSON study state, append-only events, bayescycle/Bayesite run directories, diagnostics, and human approval gates. Use for estimand definition, generative modeling, estimator planning, simulation/recovery, fitting, and model criticism.
---

# Bayescycle Study

Use this skill to run **one** phase of an agentic Bayesian workflow at a time.
The skill is a protocol layer. It must not turn `bayescycle` into a reporting
framework or modify `src/bayescycle` unless the user explicitly asks for runtime
changes.

## Core rule

Treat the current chat as non-authoritative. The authoritative study memory is:

1. `<study>/state.json`
2. `<study>/events.jsonl`
3. artifacts referenced from `state.json` with `status: "approved"`
4. run diagnostics referenced from `state.json`

If an earlier chat claim is not reflected in those files, ask whether to record
it instead of acting on it.

## Study directory

The user should provide a study directory, for example:

```text
/skill:bayescycle-study examples/black-cats-study phase=estimand
```

If no study directory is clear, ask for one. A study directory should contain:

```text
state.json
events.jsonl
artifacts/
patches/
runs/          # usually ignored by git; raw execution outputs live here
```

Use the templates in `templates/` when initializing a new study.

## Required reads

For every phase invocation:

1. Read `<study>/state.json`.
2. Read relevant entries from `<study>/events.jsonl` if present.
3. Read approved artifacts needed for the current phase.
4. Read the active adapter named by `state.toolchain.profile` from `adapters/`.
5. Read `references/phases.md`, `references/state-schema.md`,
   `references/artifact-contracts.md`, `references/diagnostics-policy.md`, and
   `references/visualization-policy.md` when the task touches those concerns.
6. Use `schemas/state.v1.schema.json` and `schemas/event.v1.schema.json` as the
   machine-readable contracts when validating or proposing state changes.

## Phase discipline

Do exactly one phase per invocation:

1. `estimand`
2. `generative_model`
3. `estimator_plan`
4. `simulation`
5. `fit`
6. `critique`
7. `revision`

Never advance to the next phase unless `state.gate` and blocking questions allow
it. If a human decision is needed, stop and ask.

Visualization is cross-cutting, not a final reporting add-on. At every phase
gate, explicitly state the visualization status: `not_applicable`, `planned`,
`produced`, `reviewed`, or `blocked`.

## State mutation policy

Do not silently edit canonical state while doing phase work.

Allowed direct writes:

- initialize a new study from templates when explicitly requested
- write proposed artifacts under `<study>/artifacts/`
- write proposed JSON patches under `<study>/patches/`
- append proposed events to `<study>/events.jsonl` only when the user has asked
  for logging or patch application
- apply a patch only after explicit human approval

For ordinary phase work, produce:

1. a markdown phase artifact
2. a JSON Patch proposal
3. a concise human gate question, if needed

## Toolchain boundary

The scientific workflow is tool-agnostic. Concrete execution is selected by
`state.toolchain.profile`.

The default profile is `jaxstanv5-bayesite-v1`:

```text
jaxstanv5 model.py -> bayescycle CLI -> Bayesite engine -> run directory
```

Use `bayescycle` as the stable command boundary. Call `bayesite` directly only
when the adapter explicitly says to, or when debugging an engine-level issue.

## Diagnostics

Bayesite NDJSON and diagnostics files are telemetry, not canonical scientific
state. Summarize them into proposed diagnostics artifacts or patch entries. Do
not paste large raw NDJSON into the conversation unless a small filtered snippet
is needed.

Before approving fit or critique artifacts, check for:

- divergences or failed chains
- bad R-hat / ESS if reported
- missing diagnostics for a run that claims to be fit
- missing required bayesite-viz visual reports or an explicit waiver
- real-data fits before simulation/recovery approval
- model or data hashes changing after approval without invalidation
- approved artifacts listed under `invalidated`

## Human gates

When a decision is required, present:

- the question
- concrete options
- consequences for the estimand/model/estimator
- the proposed state patch if the user chooses the recommended option

Then stop.

## Output format

End each invocation with:

```text
Phase: <phase>
Wrote: <paths or none>
Visualization: <not_applicable | planned | produced | reviewed | blocked>
Gate: <open | awaiting_human | approved | blocked>
Next human decision: <question or none>
```
