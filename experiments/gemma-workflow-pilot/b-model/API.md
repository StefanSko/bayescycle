# Bayeswire authoring reference

Imports:
```python
from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import HalfNormal, Normal
from bayeswire.math import linear
```

Required public interface: `@model` decorating `class ClinicModel: ...`.
Latents are class attributes: `name = Param(distribution)`.
Distributions: `Normal(location, scale)` and `HalfNormal(scale)`.
An eight-element iid latent uses `Param(distribution, size=8)`.
A positive-scale HalfNormal parameter must declare `constraint=Positive()`.
Inputs are class attributes `Data.vector()` or `Data.matrix()`.
Matrix-vector expressions use `linear(matrix).apply(vector)`, NOT Python `@`.
Symbolic parameter/data attributes support ordinary addition and multiplication.
Observed outcome: `y = Observed(Normal(mu, sigma))`.

Unrelated scalar-location example (not the target hierarchical model):
```python
@model
class Example:
    location = Param(Normal(0.0, 1.0))
    observation = Observed(Normal(location, 1.0))
```

Do not import or run Bayesjax binding/sampling or ArviZ. The evaluator owns execution.
