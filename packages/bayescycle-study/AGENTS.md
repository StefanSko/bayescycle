# Bayescycle Study CLI guidance

## Responsibility

This package owns the stdlib-only mechanism for study-state initialization,
validation, append-only events, and explicitly approved JSON Patch application.
It owns no scientific phase policy, model semantics, run execution, diagnostics
interpretation, visualization, or reporting.

## Boundaries

- Normalize JSON and CLI input immediately into validated typed boundaries.
- Refuse malformed state, malformed events, duplicate patches, and unsafe paths.
- Keep `events.jsonl` append-only and replace `state.json` atomically.
- Never apply a proposed patch without an explicit CLI invocation and actor.
- Do not import Bayescycle runtime internals; run directories are opaque paths.
- Keep the package stdlib-only.

## Validation

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```
