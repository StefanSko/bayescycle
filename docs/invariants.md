# bayescycle invariants

## Responsibility

- `bayescycle` is a workflow harness, not a sampler and not a modeling library.
- Python model execution and IR serialization are delegated to `jaxstanv5`.
- Sampling is delegated to an engine backend such as the Bayesite CLI or an
  explicitly selected in-process backend.
- Run-directory preparation, artifact-contract serialization, and command
  orchestration are the only core responsibilities.
- Run-directory metadata sidecars are allowed only when they serialize metadata
  explicitly exposed by `jaxstanv5`; `bayescycle` must not invent model
  semantics such as dimension labels.

## Boundaries

- Public input is loose CLI input; normalize it quickly into typed request and
  prepared-run values.
- The transition from `model.py` to IR is explicit and occurs before engine
  invocation.
- The Bayesite engine command is data, represented before it is executed.
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
