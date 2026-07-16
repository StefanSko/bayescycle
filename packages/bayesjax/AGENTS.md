# Bayesjax guidance

## Responsibility

Bayesjax binds Bayeswire models, compiles executable log densities and
transforms, runs BlackJAX NUTS, reports essential diagnostics, simulates
supported prior-predictive fragments, and exposes the typed
InferenceData-compatible schema.

It does not own the declaration language, IR, workflow orchestration, plotting,
reporting, artifact storage, or additional inference algorithms. NUTS is the
only sampler.

Keep [`docs/invariants.md`](docs/invariants.md) true. Follow the local
`.pi/skills/rust-style-python` guidance for API and architecture changes.

## Boundaries

- `bind_model(...)` is the explicit transition from Bayeswire metadata to typed
  runtime state.
- JAX and BlackJAX implementation details remain behind backend APIs.
- Model semantics land in Bayeswire first, with corpus coverage; Bayesjax then
  proves consume-conformance against that installed corpus.
- Preserve separate authoring, binding, compilation, and sampling phases.
- Test public behavior through public APIs. Prefer real collaborators over
  mocks and immutable typed values over untyped dictionaries.
- Numerical changes need adversarial support, tail, and gradient coverage before
  happy-path implementation.

Bayesjax is the float64 oracle for corpus evaluation fixtures. Fixture changes
are compatibility events and must not include unrelated numerical drift.

## Validation

Run the shared package checks from this directory. For backend-boundary changes,
also run the relevant optional Stan or SBC script described in
[`reference/stan/README.md`](reference/stan/README.md). See the root
`AGENTS.md` for the command set and changelog rules.
