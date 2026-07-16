# bayesjax

Bayesjax is the JAX/BlackJAX NUTS backend for
[Bayeswire](../bayeswire/) models. It binds concrete data, compiles transformed
log densities, runs NUTS, reports essential diagnostics, and provides supported
prior-predictive simulation.

The declaration language, IR, distributions and constraints metadata, specs,
and conformance corpus belong to Bayeswire. Workflow orchestration and durable
run directories belong to Bayescycle.

## Sample a model

```python
import jax.numpy as jnp

from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Normal, Truncated
from bayesjax import bind_model
from bayesjax.diagnostics import ess, rhat
from bayesjax.inference import sample


@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(
        Truncated(Normal(0.0, 1.0), lower=0.0),
        constraint=Positive(),
    )

    x = Data.vector()
    y = Observed(Normal(alpha + beta * x, sigma))


x = jnp.linspace(-3.0, 3.0, 50)
y = 2.0 + 0.5 * x
bound = bind_model(LinearRegression, {"x": x, "y": y})
result = sample(
    bound,
    seed=42,
    num_chains=4,
    num_warmup=200,
    num_samples=500,
)

print(rhat(result.samples))
print(ess(result.samples))
```

`result.samples` maps parameter names to constrained arrays shaped
`(num_chains, num_samples, *param_shape)`. Warmup and retained-draw NUTS
statistics are separate in `result.diagnostics`; divergences are available at
`result.diagnostics.sampling.is_divergent`.

## Backend semantics

- NUTS is the only inference algorithm; BlackJAX is internal.
- `bind_model(...)` is the explicit transition from Bayeswire metadata to a
  validated `BoundModel`.
- Constraints define transforms and Jacobians. A prior needing truncation
  normalization must use explicit `Truncated(...)` with matching bounds.
- Discrete distributions are valid observed likelihoods and prior-predictive
  outcomes, but discrete latent parameters are unsupported by NUTS.
- `OrderedLogistic` observed labels are zero-based.
- Partially observed continuous vectors use an explicit observed/missing index
  partition; discrete missing latents are unsupported.
- Additional density factors are evaluated by the log-density compiler but are
  rejected by prior-predictive simulation when no ancestral sampling semantics
  exist.

Complete backend invariants are in [`docs/invariants.md`](docs/invariants.md).

## InferenceData-compatible schema

Bayesjax does not depend on ArviZ, xarray, NetCDF, or Zarr. Downstream exporters
can consume its typed adapter:

```python
from bayesjax.interop.inferencedata import inferencedata_groups

schema = inferencedata_groups(bound, result)
```

The schema contains posterior draws, post-warmup sample statistics, observed
and constant data, and declared dimension metadata without constructing an
ArviZ object.

## Conformance and references

Bayesjax consumes every installed Bayeswire corpus fixture and is the float64
oracle used to generate its expected log-density and gradient values. Optional
Stan fixtures and validation commands are documented in
[`reference/stan/README.md`](reference/stan/README.md).

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

See [`AGENTS.md`](AGENTS.md) for numerical and architecture rules.
