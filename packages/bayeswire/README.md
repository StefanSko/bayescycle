# bayeswire

**The reproducible, agent-verifiable Bayesian workflow — one spec, a reference
engine, a fast engine that must agree with it, and a harness that refuses to
let either you or the agent skip a step.**

## Quickstart

You write a model; `bayescycle` provisions the sampling engine and the
plotting tool for you.

Install the workflow harness (it pins `bayeswire`, so nothing else to
install yet):

```bash
uv tool install git+https://github.com/StefanSko/bayescycle.git
```

Write `model.py`:

```python
from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Normal, Truncated

@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())

    x = Data.vector()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))
```

and `data.json`:

```json
{"x": [-1.0, -0.5, 0.0, 0.5, 1.0], "y": [-1.6, -0.3, 0.4, 1.1, 2.2]}
```

Sample, then plot:

```bash
bayescycle sample model.py --data data.json -o run/
bayescycle plot trace run/
```

`sample` downloads and sha256-verifies the pinned Bayesite engine release
into a local cache the first time it runs; later runs reuse it. `plot`
reaches `bayesite-viz` through `uvx` against a pinned commit, so there is
nothing else to `pip install`. See
[bayescycle](https://github.com/StefanSko/bayescycle) for the rest of the
CLI: prior-predictive checks, simulation/recovery, SBC, and diagnostics.

## The toolchain

| Repository | Role |
|---|---|
| **bayeswire** (here) | One spec: eDSL, resolved IR, wire codec, dimension sidecars, normative docs, and fixture corpus |
| [jaxstanv5](https://github.com/StefanSko/jaxstanv5) | Reference engine: JAX/BlackJAX backend for binding models, compiling log densities, running NUTS, and emitting diagnostics |
| [bayesite](https://github.com/StefanSko/bayesite) | Fast engine that must agree with the reference: zero-dependency Rust engine that vendors specs and fixtures by file |
| [bayescycle](https://github.com/StefanSko/bayescycle) | Harness that refuses skipped steps: file-based workflow runner for agents and humans |
| [bayesite-viz](https://github.com/StefanSko/bayesite-viz) | Downstream visualization: fit-artifact exporter and dashboards |

Every workflow step is a command that reads files and writes files; every run
is seeded and replayable; run directories are append-only; provenance records
what was actually done. The customer is an agent and the human auditing it;
the product is trustworthy process, not a sampler.

Consumers pin bayeswire by exact version and vendor or conformance-test
against the corpus; the copies of the spec in downstream repositories are
generated and hash-checked, never edited.

## This repository

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

meta = model_meta(LinearRegression)
document = meta_to_dict(meta)          # {"bayeswire_ir": 1, "model": ...}
wire_bytes = canonical_bytes(meta)     # sha256(wire_bytes) is the model hash
```

Decoding is code-free: `meta_from_dict(document)` reconstructs resolved
metadata without executing any user code, and
`bindable_from_meta(meta, dimensions=...)` returns a pure metadata class that
backends bind and sample.

## Layout

- `src/bayeswire/` — the package: `model`, `distributions`, `constraints`,
  `math`, `ir`
- `spec/` — the normative wire spec: `ir-format-v1.md`, generated
  `ir-v1-tags.md`, `dimension-sidecar-v1.md`, `data-document-v1.md`,
  `model-data-fingerprint-v1.md`
- `src/bayeswire/corpus/` — golden IR documents, canonical hashes, canonical
  data documents, fingerprint test vectors, and JAX-oracle evaluation
  fixtures, shipped as package data so Python consumers conformance-test
  against the installed pin (see its README)
- `scripts/regenerate_corpus.py` — regenerates corpus documents, hashes,
  data documents, fingerprints, and the generated tag spec after a
  deliberate format change
- `docs/releasing.md` — the release procedure: when to tag, how to cut the
  tag, and the consumer pin-bump order

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
