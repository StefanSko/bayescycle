# Bayescycle guidance

## Responsibility

Bayescycle is the Python workflow CLI between Bayeswire model files and either
the Bayesite executable (default) or Bayesjax (`[inproc]`). It owns model-file
loading, IR materialization, canonical workflow inputs, run-directory paths and
provenance, backend planning, backend invocation, and the narrow study-state
mechanism exposed under `bayescycle study`.

Study-state ownership is limited to initialization, structural validation,
append-only events, and explicit approved-patch application. Bayescycle owns no
scientific phase policy, model semantics, distribution math, sampler algorithms,
plotting, reports, or interpretation of study artifacts. The optional root
`.agents/skills/bayescycle-study/` protocol owns scientific policy.

Keep [`docs/invariants.md`](docs/invariants.md) true.

## Boundaries

- Normalize CLI input quickly into immutable typed requests and explicit plans.
- Resolve complete backend intent before writing a run directory.
- Run artifacts are append-only; never clear or overwrite an existing output.
- Study events are append-only; study state changes only through an explicit,
  validated patch application with an actor recorded.
- Bayesite commands are data before execution. Preflight the engine before
  materializing workflow artifacts.
- Backend adapters own backend options and sampler facts. Bayescycle serializes
  only facts explicitly exposed by a backend and never infers model semantics.
- The default install and execution path must remain free of JAX.
- Use `uv run ...` and `uv run python ...` in code, docs, and validation.

## Validation

Run the shared package checks from this directory. Changes to backend selection,
provisioning, or dependency wiring also require the root guards and the no-JAX
profile. See the root `AGENTS.md` for commands and changelog rules.
