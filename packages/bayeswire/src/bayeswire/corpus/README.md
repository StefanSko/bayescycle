# Golden fixture corpus

This directory is the conformance corpus for the `bayeswire_ir` v1 wire
format. Every producer and consumer in the toolchain tests against these
files; none may substitute private fixtures.

- `<model>.json` — golden IR documents for the reference models declared in
  `tests/conformance/reference_models.py`, pretty-printed for diffability.
- `hashes.json` — `sha256(canonical_bytes(meta))` for each reference model:
  the pinned canonical encoding.
- `fixtures/<model>.json` — cross-backend evaluation fixtures bundling each
  IR document with concrete bind data and log-density plus gradient values
  at deterministic unconstrained points.

The evaluation values in `fixtures/` are **JAX-oracle outputs**: computed by
the jaxstanv5 backend in float64 (`jax_enable_x64`), regenerated with
jaxstanv5's `scripts/generate_ir_fixtures.py` against this corpus. Consumers
reproduce them within the tolerance policy stated in
`spec/ir-format-v1.md` — the tolerances live in the spec, not in any
consumer's tests.

- `artifacts/<model>/` — golden **run artifacts**: real run directories
  (`model.ir.json`, `data.json`, optional `dims.json`, `posterior.ndjson`
  with `per_draw_v2` sample stats, `diagnostics.json`) produced by a real
  bayesite binary from corpus models via
  `scripts/generate_artifact_corpus.py`. Downstream exporters
  (`bayesite_idata`) consume them as test fixtures instead of synthesizing
  their own run directories.

Regenerate the documents and hashes with `scripts/regenerate_corpus.py`.
Any corpus diff is a wire-format change: it requires a spec changelog entry,
a version decision, and a coordinated consumer-pin bump.
