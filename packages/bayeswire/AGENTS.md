# AGENTS.md

## Project identity

`bayeswire` is the **model declaration language and wire format for the
bayes\* toolchain**. It contains no inference, no distribution math, no
plotting, no workflow orchestration, and imports nothing outside the standard
library.

It exists to:

- define models with `@model`, `Param`, `Data`, `Observed`, and
  `PartiallyObserved`
- resolve declarations into `ModelMeta`, the serialization boundary
- serialize/deserialize `ModelMeta` as the `bayeswire_ir` v1 wire format
- carry dimension labels in a separate sidecar document
- own the normative wire spec (`spec/`) and the golden fixture corpus
  (`corpus/`) that every producer and consumer conformance-tests against

It does **not** exist to bind data, evaluate log densities, run samplers,
interpret diagnostics, or host any backend. Invalid states the identity makes
unrepresentable: a JAX (or any non-stdlib) import anywhere in `src/` is a bug
by definition, enforced by a repo-level test that walks every module.

Known consumers: [jaxstanv5](https://github.com/StefanSko/jaxstanv5) (JAX/
BlackJAX backend), [bayesite](https://github.com/StefanSko/bayesite) (Rust
engine; vendors spec and fixtures by file, never by package dependency), and
[bayescycle](https://github.com/StefanSko/bayescycle) (workflow harness).
Consumers pin bayeswire by exact version; bumping the pin is a PR whose diff
is the compatibility review.

## Communication

Be precise and brief. Do not pad responses. State assumptions, uncertainty,
and tradeoffs explicitly.

## File handling

Read full relevant files before editing. Do not make changes from partial
context unless the file is trivial. Prefer small, targeted edits.

## Typing

Use strict typing throughout. Never use `Any`. Avoid untyped structured
dictionaries; prefer dataclasses, `TypedDict`, enums, protocols, and explicit
result types.

## Architecture

Follow `.pi/skills/rust-style-python` and keep the invariants in
[`docs/invariants.md`](docs/invariants.md) true.

Prefer:

- immutable or effectively immutable data
- narrow public APIs
- explicit state transitions
- typed return values
- small modules with clear responsibilities
- designs that make invalid states hard to represent

Avoid:

- hidden mutation
- hidden control flow
- duck typing in core semantics
- speculative abstractions
- broad convenience APIs

### Phase boundaries

When a design has distinct phases, make those phases explicit in code. Use
separate types and named transition functions instead of clever mixed
representations or hidden conversions. Class-body syntax capture, resolved
model metadata, and serialized IR are three phases; `bind` is a fourth phase
that lives in backend packages, never here.

## IR compatibility invariants

The wire format is the product. Read `spec/ir-format-v1.md` and
`spec/ir-v1-tags.md` before changing the codec, the registry, or any
registered dataclass.

- The serialized `ModelMeta` is the producer/consumer boundary.
- Decoding executes no user code.
- Node tags and field lists are the wire contract, not Python class names.
- Entry-array order is semantic and must never be reordered.
- `free_values` defines the flat unconstrained NUTS state layout.
- `stochastic_sites` defines log-density factors and value expressions.
- `data` plus `observed_nodes` define required bind inputs.
- Consumers hash received canonical bytes; they never reserialize to hash.
- Unknown non-core tags fail explicitly.
- Any change to canonical bytes, tags, or field lists requires a spec
  changelog entry, a regenerated corpus (`scripts/regenerate_corpus.py`), and
  a coordinated consumer-pin bump. Golden-file diffs plus an IR version
  decision, always.

## Development process

Work collaboratively with the user. Do not treat the project instructions as
an autonomous end-to-end workflow. Wait for user confirmation when scope,
architecture, test strategy, or cleanup direction is unclear.

Use staged red-green TDD from the current agreed state; not every task starts
at step 1. For new distributions or other conceptual modeling surfaces, start
with a corpus reference model and produce-conformance coverage, then drive
the declaration path with unit tests.

Test public behavior through public APIs; do not interrogate objects in ways
that leak private internals into higher-level tests. Avoid monkeypatching and
mocking; prefer real collaborators and behavioral tests.

## Validation

Before reporting completion, run the relevant checks, normally:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

The pytest run includes the no-JAX module walk and produce-conformance
against `corpus/`. Briefly report the result.
