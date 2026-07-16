# Bayescycle Playground

The Playground is a static browser client for Bayeswire model authoring and the
pinned Bayesite WebAssembly engine. It has no server, npm build, or separate
workflow semantics.

The UI talks to one runtime boundary:

```text
runtime.compile(source, { signal })
runtime.run(request, onProgress, { signal })
```

Compilation runs explicit user-controlled Python in a fresh disposable Pyodide
worker. The trusted client validates and hashes the returned IR bytes before a
separate engine worker consumes them. Long-running attempts are cancellable,
and accepted source, share, schema, and JSON document inputs have explicit
browser-safe bounds. The complete trust boundary and explicit non-guarantees
are in [`invariants.md`](invariants.md).

Runtime artifacts use the toolchain contracts in:

- [`../spec/data-document-v1.md`](../spec/data-document-v1.md)
- [`../spec/generated-datasets-v0.md`](../spec/generated-datasets-v0.md)
- [`../spec/run-directory-v0.md`](../spec/run-directory-v0.md)

## Development

Stage pinned assets, then run the browser suite:

```bash
uv run --no-sync python playground/scripts/stage_assets.py
uv run --no-sync ruff format --check playground
uv run --no-sync ruff check playground
uv run --no-sync pytest playground/tests -q
```

The staging script refreshes `site/VERSION.json`, the Bayeswire source used by
Pyodide, the pinned Bayesite Wasm module, and Pyodide assets. GitHub Pages runs
the same staging step before publication.
