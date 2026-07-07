# AGENTS.md

## Project identity

`bayescycle` is the Python workflow CLI that connects `bayeswire` model files
to the Bayesite Rust engine (default) or the bayesjax in-process backend
(optional, via the `[inproc]` extra).

It owns:

- loading a Python model file
- serializing `bayeswire` IR
- preparing a run directory
- serializing narrow run-directory metadata explicitly exposed by `bayeswire`
- invoking the Bayesite engine CLI

It does not own model semantics, distribution math, inference algorithms,
plotting, reports, or artifact/session management.

The optional `.agents/skills/bayescycle-study/` directory is an agent protocol
asset, not core runtime. It may document how agents should use `bayescycle`, but
it must not expand `src/bayescycle` into a study notebook, reporting system, or
artifact database without an explicit architecture decision.

## Architecture

Prefer:

- typed dataclasses for phase boundaries
- explicit state transitions
- narrow CLI-facing APIs
- stdlib-first implementation

Avoid:

- hidden global configuration
- broad workflow frameworks
- engine semantics in Python
- inventing model semantics that `bayeswire` did not expose
- untyped dictionaries in core code
- speculative abstractions

## Tooling

Use `uv run ...` for Python/project commands so scripts do not depend on a bare
`python` executable being present on `PATH`. Prefer `uv run python ...` over
`python ...` in docs, scripts, and ad-hoc validation commands.

## Validation

Before reporting completion, normally run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```
