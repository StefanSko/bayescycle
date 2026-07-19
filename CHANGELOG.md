# Changelog

All notable user-visible changes to the five lockstep Python distributions are
recorded here: `bayeswire`, `bayesjax`, `bayescycle`, `bayesite-viz`, and
`bayesite-idata`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The packages follow semantic versioning together; versioned wire and artifact
formats retain their own compatibility policies.

## [Unreleased]

### Added

- Add an **Another prior** Playground parameter source that composes a small
  prior-only model with the compiled model in a fresh disposable worker,
  rejects target divergence, data extension, and incompatible dimensions,
  simulates from the composed prior, and conditions on the original model with
  recovery projected to its shared parameters.
- Add a Playground Cancel affordance that promptly terminates every worker
  owned by an in-flight compile, generation, or fit while preserving earlier
  artifacts.

### Changed

- Pin native Bayesite provisioning to v0.3.1, including its corrected ancestral forward-simulation planning and partially observed ancestor propagation.
- Rework the Playground's data-authoring workflow into a schema-driven
  **Simulate data** section with model-derived design prefills, design expressions
  and previews, generated fixed-parameter forms, JSON escape hatches, and
  count-aware simulation controls.
- Make observed-data fitting and simulated-pair recovery clearer in the Playground
  with compact bundled data, a schema-aware ready-to-fit cue, and a labelled precis
  axis with true-value markers and legend.
- Preserve explicit pre-compile Playground design JSON while deriving form defaults
  only for empty documents, and re-derive automatic authoring defaults after model
  source changes.
- Remove stale implementation logs, completed plans, generated walkthroughs,
  and duplicated documentation so maintained guidance has a clear owner.
- Bound Playground model source, example sidecars, share decompression, JSON
  documents, runtime sampler settings, compiler schemas, schema-driven forms,
  aggregate design previews, dashboard SVG rendering, and per-chain plus merged
  posterior output. Sampling and conditioning now accept posterior output up to
  64 MiB, while portable generation inputs and published posterior sources stay
  capped at 8 MiB with an explicit publication guard. Generation snapshots
  validated plan bytes before use, and runtime posterior bound errors name their
  64 MiB ceiling explicitly; malformed or oversized input remains actionable.

## [0.6.0] - 2026-07-15

### Added

- Add Bayeswire `with_prior(Target, prior=Source)` for immutable, complete
  same-name prior replacement. Prior-only sources may add hierarchical Params;
  target outcomes, partially observed values, dimensions, and non-prior factors
  are retained. Composition closes to ordinary flat `bayeswire_ir` v1, and
  Bayescycle plus the Playground select the composed model automatically —
  through the new public `model_dependencies(...)` hook — when source, target,
  and result share a file.
- Add a fully client-side Bayescycle Playground: compile Bayeswire models in
  an isolated Pyodide worker, run the pinned Bayesite wasm engine per chain,
  normalize explicit JSON data/design/truth documents, inspect posterior
  diagnostics and SVG plots, generate paired parameter/complete-dataset
  collections from fixed, model-prior, or posterior sources, select and fit a
  generated dataset with recovery, download portable generation-run artifacts,
  and share reviewed projects without a backend. Each explicit compile uses a
  fresh disposable worker; the browser validates and hashes untrusted compiler
  bytes before separate engine workers consume them.
- Add `bayescycle generate` and source-free `bayescycle replay` support for
  portable `bayescycle.generation-run.v0` directories containing exact closed
  IR, design, generation plan, source payloads, and paired generated datasets.

### Changed

- Rework the Playground run UI into explicit Generate datasets and Condition
  on a dataset directions with one immutable generation plan, uniform count and
  seed semantics, selectable parameter and dataset sources, and scoped
  generation/conditioning state. Generation edits preserve fits to observed
  data but invalidate fits to generated selections; inference-setting edits
  preserve completed fits and generated collections, and stale asynchronous
  completions cannot replace newer lineage.
- Pin native Bayesite provisioning and the Playground Wasm engine to Bayesite
  v0.3.0, which supplies the bounded functional `generate` operation.
- Trim the bundled Playground models to show only the Bayeswire imports each
  declaration actually uses.

### Fixed

- Mark R-hat and ESS unavailable in merged browser posterior trailers instead
  of misreporting the first chain's diagnostics as multi-chain summaries.

## [0.5.0] - 2026-07-10

### Added

- Add `Submodel(Model)` for closed, namespaced model composition. Child
  parameters, data, expressions, dimensions, partially observed values, and
  observed factors flatten under opaque dotted names before IR serialization.
- Add composed-model and adversarial `VectorBounds` owner cases to the shared
  Bayeswire conformance corpus.

### Changed

- Define a `VectorBounds` free value's support owner as its unique same-name
  direct or scatter stochastic site, independent of stochastic-site order.
- Classify declaration-backed sites before BayesJAX prior-predictive drawing;
  additional density Factors are rejected because they have no supported
  ancestral sampling semantics.
- Pin the default provisioned Bayesite engine to v0.2.1.

### Fixed

- Prevent differently named factors from changing `VectorBounds` support-edge
  folding.
- Reject missing, duplicate, or malformed same-name `VectorBounds` owners with
  repair-oriented bind errors.
- Preserve direct aliases of composed submodel members and reject names that
  would collide with `Submodel` internals.

### Compatibility

- `bayeswire_ir` remains version 1. No existing canonical model bytes changed;
  this release adds corpus cases and clarifies owner and ancestral-simulation
  semantics.

## [0.4.0] - 2026-07-09

### Added

- Add data-dependent per-coordinate `VectorBounds` constraints for censored
  `PartiallyObserved` values.
- Add BayesJAX transforms, support folding, prior-predictive simulation, and
  statistical validation for bounded partially observed models.
- Add censored Exponential and interval-censored Normal conformance fixtures.

### Fixed

- Align vectorized distribution support through `missing_idx`, reject
  zero-mass bounds, and keep strict-support transforms one ULP inside finite
  endpoints.

## [0.3.0] - 2026-07-07

### Changed

- Publish `bayeswire`, `bayesjax`, `bayescycle`, `bayesite-viz`, and
  `bayesite-idata` together as the first lockstep monorepo release.
- Keep visualization/export dependencies behind standalone `uvx` process
  boundaries so the default Bayescycle install remains lightweight.
- Publish each distribution through its own PyPI trusted-publishing
  environment.

Earlier package history lived in the predecessor repositories and is not
reconstructed here.

[Unreleased]: https://github.com/StefanSko/bayescycle/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/StefanSko/bayescycle/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/StefanSko/bayescycle/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/StefanSko/bayescycle/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/StefanSko/bayescycle/releases/tag/v0.3.0
