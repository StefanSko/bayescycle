# bayescycle-study

A stdlib-only CLI for initializing, validating, and applying approved state
transitions to Bayescycle study directories. It owns study-state mechanics, not
scientific phase policy or sampler execution.

```bash
uv run --package bayescycle-study bayescycle-study init example \
  --study-id example --title "Example study"
uv run --package bayescycle-study bayescycle-study validate example
uv run --package bayescycle-study bayescycle-study apply example \
  example/patches/0001.json --actor human
```

`apply` accepts an RFC 6902 JSON Patch array, validates the resulting state,
appends a hash-linked audit event, refuses duplicate patches, and leaves the
canonical files unchanged on validation failure.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```
