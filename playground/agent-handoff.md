# Agent handoff: functional generation workflow

## Architecture

The static Playground has one UI-facing boundary:

```text
runtime.compile(source)
runtime.run(request, onProgress)
```

`BrowserRuntime` lowers immutable generation plans to Bayesite v0.3.0's one
bounded native `generate` operation. Application code does not choose private
fixed/prior/posterior engine commands or derive semantics from raw IR.

Normative contracts:

- [`invariants.md`](invariants.md)
- [`../docs/generation-plan-v0.md`](../docs/generation-plan-v0.md)
- [`../docs/playground-runtime-v0.md`](../docs/playground-runtime-v0.md)
- [`../packages/bayescycle/docs/run-directory-v0.md`](../packages/bayescycle/docs/run-directory-v0.md)

## Functional workflow

1. Compile a Bayeswire model to exact IR bytes.
2. Supply explicit canonical design/context JSON.
3. Choose fixed values, the closed model prior, or a surviving compatible fit.
4. Generate one paired `generated_datasets.ndjson` collection with explicit
   count and generation seed.
5. Select one parameter/dataset pair.
6. Condition separately on that complete selected dataset.
7. Run engine-owned recovery against its paired natural-scale parameters.

Generation, selection, and conditioning have separate revision scopes. Failed
replacement attempts preserve successful ancestors. Selection edits invalidate
only selected-dataset fit descendants. Inference edits preserve completed fits
and generation collections. Replacing a fit invalidates collections sourced
from the old posterior.

## Publication and replay

Fixed and model-prior browser generations expose a portable local bundle:

```text
model.ir.json
design.json
generation-plan.json
fixed-parameters.json       # fixed source only
generated_datasets.ndjson
run.json
```

A posterior source is publishable only when its fit and exact conditioning data
are file-backed. Browser fits are runtime-associated, so the Playground exposes
only `generated_datasets.ndjson` for that case instead of claiming portability.

The native equivalent is:

```bash
uv run bayescycle generate model.py \
  --design design.json --source fixed --parameters parameters.json \
  --count 100 --seed 123 -o generation-run

mv generation-run /another/location/
uv run bayescycle replay /another/location/generation-run -o replayed-run
```

Replay resolves only regular files contained by the moved run, verifies exact
SHA-256 metadata and generated-artifact lineage, and requires no original
Python source.

## Validation

```bash
cd packages/bayescycle
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest tests -q

cd ../../playground
uv run ruff format --check .
uv run ruff check .
uv run pytest tests -q

cd ..
uv run pytest tests -q
```

Keep browser generation to one native request per collection. Preserve received
bytes for hashes and replay. Do not reintroduce browser loops, predictive
lowering, mutable plan aliases, runtime-only portability claims, or source-path
fallbacks during replay.
