# Bayeswire guidance

## Responsibility

Bayeswire owns the model declaration language, resolved `ModelMeta`,
`bayeswire_ir` codec, dimension sidecar, wire-related root specs, and golden
conformance corpus. It owns no binding, arrays, log-density math, inference, plotting, or
workflow orchestration and must import only the Python standard library.

Keep [`docs/invariants.md`](docs/invariants.md) true. Follow the local
`.pi/skills/rust-style-python` guidance for API and architecture changes.

## Boundaries

- Class-body syntax, resolved metadata, serialized IR, and backend binding are
  distinct phases. Binding never belongs here.
- `ModelMeta` is the producer/consumer boundary. Decoding executes no user code.
- Wire tags and field lists—not Python class names—are the contract. Entry-array
  order is semantic; `free_values` fixes the flat NUTS state layout.
- Composition closes to ordinary flat metadata. No composition node or
  authoring dependency enters IR.
- Test public behavior through public APIs and prefer immutable typed values.

## Wire changes

Before changing the codec, registry, or a registered dataclass, read
`../../spec/ir-format-v1.md` and `../../spec/ir-v1-tags.md`. Any change to tags,
fields, or canonical bytes requires:

1. a spec changelog entry and explicit IR-version decision;
2. deliberate corpus regeneration with `scripts/regenerate_corpus.py`;
3. byte review of the corpus diff;
4. regenerated Bayesjax evaluation fixtures when required;
5. a later byte-reviewed Bayesite vendor refresh.

New modeling surfaces start with declaration tests and a corpus case. Existing
corpus documents must remain byte-identical unless a reviewed format change says
otherwise.

## Validation

Run the package checks from this directory. `pytest` includes the no-JAX module
walk and produce-conformance against the corpus. See the root `AGENTS.md` for the
shared command set and release rules.
