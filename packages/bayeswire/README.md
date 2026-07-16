# bayeswire

Bayeswire is the stdlib-only model declaration language and wire format for the
Bayescycle toolchain. It owns authoring semantics, resolved `ModelMeta`,
`bayeswire_ir` serialization, dimension sidecars, the wire-related documents in
root [`spec/`](../../spec/), and the golden conformance corpus.

It contains no arrays, log-density math, binding, inference, plotting, or
workflow orchestration.

## Declare a model

```python
from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Normal, Truncated


@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(
        Truncated(Normal(0.0, 1.0), lower=0.0),
        constraint=Positive(),
    )

    x = Data.vector()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))
```

Resolve and serialize without binding data:

```python
from bayeswire.ir import canonical_bytes, meta_from_dict, meta_to_dict
from bayeswire.model import model_meta

meta = model_meta(LinearRegression)
document = meta_to_dict(meta)
wire_bytes = canonical_bytes(meta)
restored = meta_from_dict(document)
```

`ModelMeta` is the serialization boundary. Decoding executes no user code.
Canonical bytes are stable input to model hashes and downstream conformance.
Dimension labels and coordinates travel in a separate sidecar and do not change
the model hash.

## Composition

`Submodel(Model)` reuses a complete model under an explicit namespace. It
prefixes and flattens the child's parameters, data, dimensions, expressions,
and stochastic factors before serialization; hierarchy is not an IR concept.

`with_prior(Target, prior=Source)` returns a new closed model with a complete
same-name replacement prior:

```python
from bayeswire import Param, model, with_prior
from bayeswire.constraints import Positive
from bayeswire.distributions import HalfNormal, Normal


@model
class SimulationPrior:
    alpha = Param(Normal(0.0, 0.25))
    beta = Param(Normal(1.0, 0.2))
    sigma = Param(HalfNormal(0.5), constraint=Positive())


SimulationModel = with_prior(LinearRegression, prior=SimulationPrior)
```

The source must be structurally prior-only and provide every target parameter
with the same name, constraint, size, and dimensions. Composition is immutable
and closes to ordinary flat `bayeswire_ir` v1. The precise durable rules are in
[`docs/invariants.md`](docs/invariants.md).

## Contracts and corpus

The root [`spec/`](../../spec/) defines the shared interoperability formats.
The corpus under
[`src/bayeswire/corpus/`](src/bayeswire/corpus/) contains golden IR documents,
canonical hashes, data documents, and Bayesjax-oracle evaluation fixtures.
Producers reproduce its bytes; consumers evaluate it within the spec tolerance.

Any tag, field, or canonical-byte change requires an explicit IR-version
decision and deliberate corpus regeneration.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

The suite includes a no-JAX import walk and produce-conformance. See
[`AGENTS.md`](AGENTS.md) for working rules.
