# bayescycle invariants

## Responsibility

- `bayescycle` is a workflow harness, not a sampler and not a modeling library.
- Python model execution and IR serialization are delegated to `bayeswire`,
  the owner of authoring semantics and the wire format.
- Sampling is delegated to an engine backend such as the Bayesite CLI or an
  explicitly selected in-process backend.
- Run-directory preparation, canonical data-artifact serialization, command
  orchestration, and narrow run provenance metadata are the only core
  responsibilities.
- Run-directory metadata sidecars are allowed only when they serialize command
  provenance, artifact paths and hashes, or metadata explicitly exposed by
  `bayeswire`; `bayescycle` must not invent model semantics such as dimension
  labels. Authoring semantics belong to bayeswire; sampler facts belong to the
  selected backend (jaxstanv5 or Bayesite).

## Boundaries

- Public input is loose CLI input; normalize it quickly into typed requests,
  explicit run plans, and materialized commands.
- The transition from `model.py` to IR is explicit and occurs before engine
  invocation.
- Concrete data crossing workflow-stage boundaries uses
  `bayescycle.data.json.v1`; backend-native data files are adapter-private
  materializations.
- Multi-stage backend intent is resolved as a single backend plan or a complete
  explicit mixed plan before any run-directory writes.
- The Bayesite engine command is data, represented before it is executed.
- Durable run artifacts are append-only: commands must refuse existing output
  artifacts instead of clearing or overwriting them.
- Bayesite engine paths and required subcommands are preflighted before
  execution creates run artifacts.
- Backend stdout/stderr and exit status are not interpreted as sampler semantics
  unless a later explicit diagnostics phase is added.
- When bayescycle serializes sampler facts, those facts must be explicitly
  exposed by the selected backend; bayescycle must not infer or invent sampler
  telemetry.

## Non-goals

- No inference algorithms.
- No distribution math.
- No plotting, report generation, notebooks, or artifact product layer.
- No hidden discovery of remote engines or environments.

## Module invariants

This section is intentionally last. Update it whenever modules move so the source
layout keeps matching the architecture.

- `bayescycle._cli` parses CLI input, builds logical requests, and wires
  capability-specific backend handles into workflow operations. It must not own
  artifact schemas, sampler semantics, concrete backend construction, or backend
  execution details.
- `bayescycle._backend_runtime` resolves concrete first-party runtime backend
  adapters from explicit backend selection into capability-specific backend
  protocol objects. It may import concrete backend adapters and run Bayesite
  preflight, but it must not own workflow artifact schemas, sampler semantics,
  or command execution.
- `bayescycle._workflow.capabilities` owns the closed workflow-capability ADT,
  open backend identity value, and immutable first-party backend catalog. It may
  validate backend/capability support, but it must not import concrete backend
  adapters or execute backend code.
- `bayescycle._workflow.requests` contains backend-neutral logical CLI input
  normalized into typed immutable requests. Backend selection, backend-private
  passthrough flags, and backend-native option rendering must live outside these
  request dataclasses. It must not read or write the filesystem.
- `bayescycle._workflow.contexts` contains planned path/model/data contexts. It
  may name workflow-owned paths, but it must not materialize them.
- `bayescycle._workflow.plans` contains immutable run plans that pair workflow
  paths with backend actions. It must not know concrete backend implementation
  details.
- `bayescycle._workflow.protocols` defines narrow backend capability protocols.
  It may depend on typed requests, contexts, and plan descriptions, but not on
  first-party backend modules.
- `bayescycle._workflow.operations` owns explicit planning and materialization
  transitions. It may load models, validate inputs, write workflow-owned run
  inputs, and call backend capability methods; it must not run samplers or
  interpret sampler telemetry.
- `bayescycle._workflow.documents` renders typed plans to JSON-ready dry-plan
  documents. It must only lower backend-provided descriptions; it must not infer
  backend settings.
- `bayescycle._workflow.filesystem` contains generic file/path guards and copy
  helpers. It must not know model, sampler, or backend semantics.
- `bayescycle._workflow.backend_plan` resolves multi-stage backend intent before
  run-directory writes. It must reject implicit mixed plans and backend-specific
  options that do not correspond to a selected backend.
- `bayescycle.data` is the stable public import surface for canonical data
  artifact helpers. It may re-export the canonical data API, but it must not
  grow workflow orchestration or backend-specific behavior.
- `bayescycle._run_artifacts` owns durable artifact references, format markers,
  serializers, and compatibility rules for the run-directory contract. It must
  not import workflow orchestration or concrete backend adapters.
- `bayescycle._integrations.descriptions` owns the closed plan-description ADT
  for integration modes. It distinguishes `external-command` from
  `in-process-python` without selecting concrete backends.
- `bayescycle._integrations.external_command` owns the Unix-style subprocess
  boundary: argv, owned output clearing, exit code. It must not contain
  Bayesite-specific semantics.
- `bayescycle.backends.bayesite` is a first-party external-command adapter. It
  may build Bayesite argv, render logical settings as Bayesite CLI flags,
  validate Bayesite passthrough flags, materialize Bayesite-private files under
  `.bayesite/`, preflight the selected binary, and canonicalize generated data;
  it must not expose Bayesite-private files as workflow-stage artifacts.
- `bayescycle.backends.jaxstanv5` is a first-party in-process Python adapter. It
  may call public jaxstanv5 runtime APIs (`bind_model`, `sample`,
  `simulate_prior_predictive`) and serialize their explicitly exposed results;
  it must not make BlackJAX or JAX implementation details part of the workflow
  contract. It is the only module allowed to import the `jaxstanv5` package,
  which is an optional dependency installed via the `[inproc]` extra.
