# Changelog

All notable user-visible changes to the five lockstep Python distributions are
recorded here: `bayeswire`, `bayesjax`, `bayescycle`, `bayesite-viz`, and
`bayesite-idata`.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The packages follow semantic versioning together; versioned wire and artifact
formats retain their own compatibility policies.

## [Unreleased]

### Added

- Add a fully client-side Bayescycle Playground: compile Bayeswire models in
  an isolated Pyodide worker, run the pinned Bayesite wasm engine per chain,
  normalize explicit JSON data/design/truth documents, inspect posterior
  diagnostics and SVG plots, simulate and check recovery, download standard
  run-directory artifacts, and share reviewed projects without a backend. Each
  explicit compile uses a fresh disposable worker; the browser validates and
  hashes untrusted compiler bytes before separate engine workers consume them.

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

[Unreleased]: https://github.com/StefanSko/bayescycle/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/StefanSko/bayescycle/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/StefanSko/bayescycle/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/StefanSko/bayescycle/releases/tag/v0.3.0
