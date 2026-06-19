# AGENTS.md

## Project identity

`bayescycle` is the Python workflow CLI that connects `jaxstanv5` model files to
the Bayesite Rust engine.

It owns:

- loading a Python model file
- serializing `jaxstanv5` IR
- preparing a run directory
- serializing narrow run-directory metadata explicitly exposed by `jaxstanv5`
- invoking the Bayesite engine CLI

It does not own model semantics, distribution math, inference algorithms,
plotting, reports, or artifact/session management.

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
- inventing model semantics that `jaxstanv5` did not expose
- untyped dictionaries in core code
- speculative abstractions

## Validation

Before reporting completion, normally run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```
