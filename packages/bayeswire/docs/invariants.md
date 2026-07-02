# Invariants

Core invariants that should remain true as the codebase changes.

## Scope

- bayeswire owns the model declaration language, the resolved-metadata IR,
  the `bayeswire_ir` v1 wire format, the dimension sidecar format, the
  normative spec, and the conformance corpus.
- bayeswire contains no inference, no distribution math, no plotting, and no
  workflow orchestration.
- bayeswire imports nothing outside the Python standard library. Every
  module must import with `jax` and `blackjax` blocked; the no-JAX walk
  covers the whole package.

## Declaration language

- `Param`, `Data`, and `Observed` are declarations.
- `Param(...)` is latent and contributes a prior term.
- `Data.scalar()`, `Data.vector(...)`, `Data.matrix(...)`, and
  `Data.array(...)` are known inputs with shape/rank schemas and contribute no
  log-density term.
- `Observed(...)` is known input and contributes a likelihood term.
- `PartiallyObserved.vector(...)` contributes one continuous log-density factor
  over an assembled vector whose observed coordinates are fixed data and whose
  missing coordinates are free NUTS values.
- A model has one or more stochastic declarations: `Param`, `Observed`, or
  `PartiallyObserved`.
- `Observed` nodes are optional; prior-only models are valid.
- Declaration aliases are invalid: one declaration object maps to one class
  attribute name.
- `Dim(...)` labels and coordinates are authoring-side semantic metadata only;
  they do not change log-density, transforms, sampling, or distribution shapes.
- Dimension coordinates are optional JSON-scalar metadata and are validated
  against known static axis sizes at declaration time; validation against
  concrete bound shapes is a backend concern.
- Declaration classes have no base class other than `object`. A model's meaning
  is local to one decorated class body so that model text stays statically
  parseable by a standalone validator; inheritance is rejected at `@model` time
  even when the base class carries no declarations.
- Declaration-time bound validation (Uniform and Truncated supports against
  constraints) compares exact Python floats; declaration semantics never
  depend on import order or backend float configuration.

## Phase boundaries

- Class-body syntax capture and resolved model metadata are separate phases.
- Class-body arithmetic, indexing, and supported `bayeswire.math` helpers create
  private deferred syntax, never final expression IR.
- Declaration expressions support Python scalar literals as constants; fixed
  non-scalar inputs must be represented explicitly as shaped `Data` declarations.
- Non-scalar fixed distribution parameters in model declarations are invalid;
  they must enter through named `Data` declarations.
- Distributions with symbolic declaration parameters must expose those fields as
  dataclass fields. Opaque non-dataclass distributions may contain only concrete
  parameters.
- Backend array functions are not declaration-language operations; supported
  symbolic math functions cross the declaration boundary through explicit helper
  nodes.
- `_resolve_model_declaration(...)` is the only transition from declaration
  symbols to named references.
- Binding is a backend phase. No module in this package binds data, holds
  arrays, or attaches runtime methods to model classes.
- `_deferred.py` is private class-body syntax capture.
- `core.py` does not construct final expression IR.
- `expr.py` is resolved/final IR only.
- Final expression trees contain no declaration symbols, raw declarations,
  deferred syntax tokens, or raw Python tuple/slice indexes.
- Final expression trees may contain explicit unary operation nodes only for
  supported declaration-language unary operations such as `neg`, `exp`, and
  `sigmoid`.

## IR and serialization

- `ModelMeta` contains resolved metadata only, including resolved data schemas,
  free NUTS values, and stochastic log-density sites.
- `ModelMeta` is the serialization boundary: `bayeswire.ir` round-trips resolved
  metadata only, executes no user code on decode, and uses only the standard
  library.
- Distribution and constraint classes are serializable metadata only. They
  must not define runtime methods such as `log_prob`, `sample`, `transform`,
  `inverse_transform`, or Jacobian evaluation; those operations live in
  backend packages.
- Serialized IR node tags, not Python class names, are the wire contract; tag,
  field, or encoding changes require a regenerated corpus, a spec changelog
  entry, and a format version decision (see `spec/ir-format-v1.md`).
- In serialized `ModelMeta`, `free_values` defines flat NUTS state layout,
  `stochastic_sites` defines log-density factors, and `data` plus
  `observed_nodes` define required bind inputs.
- A model reconstructed with `bindable_from_meta(...)` is indistinguishable
  from one produced by `@model` through the public hooks. Dimension labels
  travel in a separate sidecar document (`spec/dimension-sidecar-v1.md`);
  without the sidecar the reconstructed model carries no dimension metadata.
- The corpus under `corpus/` is the single conformance baseline. Producers
  reproduce it byte-for-byte; consumers decode it and reproduce the recorded
  JAX-oracle evaluations within the spec's tolerance policy.
