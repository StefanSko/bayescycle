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

- `Param`, `Data`, `Observed`, and `Submodel` are declarations.
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
  attribute name, including `Submodel` instances.
- `Submodel(Model)` composes one already-resolved model as a closed namespace:
  it accepts no input wiring, includes every child stochastic factor (including
  `Observed` likelihoods), and prefixes child data bind keys.
- `with_prior(Target, prior=Source)` is immutable, complete prior replacement
  between two independently closed model classes. It never binds data or
  exposes open wiring.
- The source is structurally prior-only: its free values are exactly its Params,
  every Param has one declaration-backed site, and it has no observed values,
  non-Param free values, or additional factors. Source-only hierarchical Params
  are retained and their dependencies must follow source Param order.
- Every target Param has an exact same-name source Param with matching
  constraint, size, dimensions, and coordinates. The source distribution is the
  replacement prior; there is no renaming, broadcasting, partial replacement,
  or constraint subtyping.
- Composition removes exactly declaration-backed target Param sites, retains
  target outcomes, partially observed values, and additional factors, and
  rejects role collisions or dangling references before and after the merge.
  Source data precedes compatible target-only data; source Params and their
  sites precede retained target free values and factors.
- The result closes to ordinary flat `ModelMeta`; composition state and
  `model_dependencies(...)` provenance never enter IR or the dimension sidecar.
- Parents may reference composed parameters, data, derived expressions, and
  partially observed values. Child `Observed` declarations contribute factors
  but are not expression values, matching ordinary same-class behavior.
- Repeated submodel instances are independent. Their attribute names determine
  distinct dotted prefixes; child variable names, dimension labels, and
  coordinate keys are all prefixed.
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
  symbols to named references. Submodel composition occurs at this boundary by
  prefixing and flattening already-resolved child metadata; no `Submodel` or
  submodel-member token enters final expression IR.
- Prior composition begins only from resolved, closed model metadata and uses
  explicit private factorization, same-name wiring, composition, and closure
  phases. Its result is validated at the ordinary metadata/dimension boundary;
  private phase values never enter final expression IR or public hooks.
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
- Declaration syntax `linear(matrix).apply(vector)` resolves to a narrow
  `MatVecOp` with exact `[m, n] @ [n] -> [m]` semantics. A `LinearMap` is a
  complete immutable authoring value that may be named and reused; it never
  enters `ModelMeta`, the dimension sidecar, or serialized IR. The operation
  adds no batching, broadcasting, vector-matrix, or matrix-matrix behavior;
  concrete shape validation remains a backend binding concern.

## IR and serialization

- `ModelMeta` contains resolved metadata only, including resolved data schemas,
  free NUTS values, and stochastic log-density sites. Composed models use the
  same flat representation with opaque dotted names; hierarchy is not a wire
  concept and consumers never split names on `.`.
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
- A `VectorBounds` free value is owned by exactly one same-name stochastic
  site evaluated directly at its `ParamRef`, either alone or as a
  `VectorScatterOp.missing_values`; differently named factors never determine
  its base-support folding.
- Ancestral consumers classify declaration-backed stochastic sites by
  structural declaration matching, never expression-reference heuristics or
  site order. Additional density factors stay part of log-density evaluation
  but must be rejected by prior-predictive workflows that have no factor-aware
  sampling semantics; silently dropping or independently drawing them changes
  the model.
- A model reconstructed with `bindable_from_meta(...)` is indistinguishable
  from one produced by `@model` or `with_prior(...)` through execution-facing
  public hooks. Dimension labels travel in a separate sidecar document
  (`spec/dimension-sidecar-v1.md`); without the sidecar the reconstructed model
  carries no dimension metadata. Authoring-only model dependencies are exposed
  separately by `model_dependencies(...)` and are not serialized.
- The corpus under `src/bayeswire/corpus/` (shipped as package data) is
  the single conformance baseline. Producers
  reproduce it byte-for-byte; consumers decode it and reproduce the recorded
  JAX-oracle evaluations within the spec's tolerance policy.
