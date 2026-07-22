# Repository guidance

## Scope

This monorepo publishes six lockstep Python distributions:

- `packages/bayeswire` — stdlib-only model language, IR codec, wire specs, and corpus.
- `packages/bayesjax` — JAX/BlackJAX NUTS backend and float64 corpus oracle.
- `packages/bayescycle` — workflow CLI and run-directory owner.
- `packages/bayescycle-study` — stdlib-only study-state mechanism; it owns no
  scientific phase policy or run execution.
- `packages/bayesite-idata` and `packages/bayesite-viz` — standalone heavy
  projects invoked by Bayescycle through pinned `uvx` process boundaries.

Bayeswire, Bayesjax, Bayescycle, and Bayescycle Study are root uv-workspace members. The two
visualization projects have independent locks so their ArviZ stack cannot enter
Bayescycle's default dependency closure. Bayesite is a separate Rust repository;
it vendors `spec/` and the Bayeswire corpus and is consumed as a pinned binary.

Read the nearest package `AGENTS.md` before changing package code. Read its
`docs/invariants.md` before changing architecture or semantics.

## Hard rules

- Keep phase boundaries explicit and core state typed; do not use `Any` or
  untyped structured dictionaries in core code.
- Bayeswire stays stdlib-only. Bayescycle's default install stays free of JAX;
  Bayesjax is available only through the `[inproc]` extra.
- `spec/` is normative. Changes to IR tags, fields, or canonical bytes require a
  spec changelog entry, corpus regeneration, an explicit IR-version decision,
  and byte review.
- Use `uv run ...` for project commands and `uv run python ...` rather than a
  bare `python` executable.
- User-visible changes update `CHANGELOG.md` under `Unreleased`.
- Do not broaden package responsibilities without user agreement.

## Validation

Root workspace guards:

```bash
uv run pytest tests -q
```

From a workspace package directory, normally run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

For `bayesite-viz` and `bayesite-idata`, run `uv sync` and the same checks from
that standalone project directory. Playground validation is documented in
[`playground/README.md`](playground/README.md).

Release procedure: [`docs/releasing.md`](docs/releasing.md).
