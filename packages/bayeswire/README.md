# bayeswire

**The model declaration language and wire format for the bayes\* toolchain.**

bayeswire owns the language: a declarative Python eDSL for Bayesian models,
the resolved-metadata IR it compiles to, the `bayeswire_ir` v1 wire format
that IR serializes as, the dimension-label sidecar format, the normative spec
for all of it, and the golden fixture corpus every producer and consumer
conformance-tests against.

It contains no inference, no distribution math, no plotting, and no workflow
orchestration, and it imports nothing outside the Python standard library.

```python
from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Normal, Truncated
from bayeswire.ir import canonical_bytes, meta_to_dict
from bayeswire.model import model_meta

@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())

    x = Data.vector()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))

document = meta_to_dict(model_meta(LinearRegression))   # {"bayeswire_ir": 1, "model": ...}
model_hash = canonical_bytes(model_meta(LinearRegression))
```

Decoding is code-free: `meta_from_dict(document)` reconstructs resolved
metadata without executing any user code, and
`bindable_from_meta(meta, dimensions=...)` returns a pure metadata class that
backends bind and sample.

## The toolchain

| Repository | Role |
|---|---|
| **bayeswire** (here) | The language: eDSL, IR codec, wire spec, fixture corpus |
| [jaxstanv5](https://github.com/StefanSko/jaxstanv5) | JAX/BlackJAX backend: bind, compile log densities, NUTS, diagnostics |
| [bayesite](https://github.com/StefanSko/bayesite) | Zero-dependency Rust engine; vendors spec and fixtures by file |
| [bayescycle](https://github.com/StefanSko/bayescycle) | File-based workflow harness for agents |
| [bayesite-viz](https://github.com/StefanSko/bayesite-viz) | Fit-artifact exporter and dashboards |

Consumers pin bayeswire by exact version and vendor or conformance-test
against `corpus/`; the copies of the spec in downstream repositories are
generated and hash-checked, never edited.

## Layout

- `src/bayeswire/` — the package: `model`, `distributions`, `constraints`,
  `math`, `ir`
- `spec/` — the normative wire spec: `ir-format-v1.md`, generated
  `ir-v1-tags.md`, `dimension-sidecar-v1.md`
- `corpus/` — golden IR documents, canonical hashes, and JAX-oracle
  evaluation fixtures (see `corpus/README.md`)
- `scripts/regenerate_corpus.py` — regenerates corpus documents, hashes, and
  the generated tag spec after a deliberate format change

## Development

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

The test suite includes a no-JAX walk (every module must import with `jax`
and `blackjax` blocked) and produce-conformance (reference declarations must
reproduce the corpus byte-for-byte). See `AGENTS.md` for the working
discipline and `docs/invariants.md` for the invariants.
