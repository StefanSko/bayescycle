---
name: bayescycle-study
description: Runs one gated phase of an agentic Bayesian study workflow using JSON study state, append-only events, bayescycle/Bayesite run directories, diagnostics, and human approval gates. Use for estimand definition, generative modeling, estimator planning, simulation/recovery, fitting, model criticism, and reporting the estimand answer.
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

## Tool invariant precedence

This skill is an anti-corruption layer across independent tools. It may compose
workflows through public contracts, but it must not weaken the invariants of
`bayesjax`, Bayesite, `bayescycle`, or `bayesite-viz`.

If this skill conflicts with a tool repository's `AGENTS.md`, documented
invariants, or public contract, the tool repository wins. Stop and ask before
changing code, relying on private APIs, or encoding cross-repository assumptions.

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

Initialize a study through the validated standalone CLI rather than copying
or editing canonical JSON by hand:

```bash
uv run --package bayescycle-study bayescycle-study init <study> \
  --study-id <slug> --title "<title>" --actor <actor>
```

Use the files in `templates/` only as documentation and proposal examples.

## Required reads

For every phase invocation:

1. Read `<study>/state.json`.
2. Read relevant entries from `<study>/events.jsonl` if present.
3. Read approved artifacts needed for the current phase.
4. Read the active adapter named by `state.toolchain.profile` from `adapters/`.
5. Read `references/phases.md`, `references/state-schema.md`,
   `references/artifact-contracts.md`, `references/diagnostics-policy.md`,
   `references/mcmc-diagnostics.md`, `references/visualization-policy.md`, and
   `references/dag-generative-models.md` when the task touches those concerns.
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
7. `report`
8. `revision`

Never advance to the next phase unless `state.gate` and blocking questions allow
it. If a human decision is needed, stop and ask.

Visualization is cross-cutting, not a final reporting add-on. At every phase
gate, explicitly state the visualization status: `not_applicable`, `planned`,
`produced`, `reviewed`, or `blocked`.

For causal questions, DAG reasoning is cross-cutting too: the `estimand` phase
must identify whether the target is causal, the `generative_model` phase must
state DAG assumptions, the `estimator_plan` phase must justify adjustment, and
`simulation` must recover the estimand under the assumed DAG.

## State mutation policy

Do not silently edit canonical state while doing phase work.

Allowed direct writes:

- write proposed artifacts under `<study>/artifacts/`
- write proposed JSON patches under `<study>/patches/`

Canonical state mutation must go through the standalone CLI. Initialize only
when explicitly requested. Apply a patch only after explicit human approval:

```bash
uv run --package bayescycle-study bayescycle-study apply \
  <study> <study>/patches/<patch>.json --actor <actor>
```

The CLI validates the resulting state and appends the audit event. Do not edit
`state.json` or `events.jsonl` directly.

For ordinary phase work, produce:

1. a markdown phase artifact
2. a JSON Patch proposal
3. a concise human gate question, if needed

## Toolchain boundary

The scientific workflow is tool-agnostic. Concrete execution is selected by
`state.toolchain.profile`.

The default profile is `bayesjax-bayesite-v1`:

```text
bayesjax model.py -> bayescycle CLI -> Bayesite engine -> run directory
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
- diagnostic review for the approved estimand or derived quantity when practical
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
