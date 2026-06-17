# bayescycle invariants

## Responsibility

- `bayescycle` is a workflow harness, not a sampler and not a modeling library.
- Python model execution and IR serialization are delegated to `jaxstanv5`.
- Sampling is delegated to the Bayesite engine CLI.
- Run-directory preparation and command orchestration are the only core
  responsibilities.

## Boundaries

- Public input is loose CLI input; normalize it quickly into typed request and
  prepared-run values.
- The transition from `model.py` to IR is explicit and occurs before engine
  invocation.
- The Bayesite engine command is data, represented before it is executed.
- Engine stdout/stderr and exit status are not interpreted as sampler semantics
  unless a later explicit diagnostics phase is added.

## Non-goals

- No inference algorithms.
- No distribution math.
- No plotting, report generation, notebooks, or artifact product layer.
- No hidden discovery of remote engines or environments.
