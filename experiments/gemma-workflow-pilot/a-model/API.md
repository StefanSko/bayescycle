# NumPyro authoring reference

Imports:
```python
import numpyro
import numpyro.distributions as dist
```

Required public interface: `def model(x, clinic_design, y=None): ...`
Each latent is a named sample site: `value = numpyro.sample("name", distribution)`.
Distributions: `dist.Normal(location, scale)` and `dist.HalfNormal(scale)`.
For an eight-element iid latent vector, a Normal can use
`.expand((8,)).to_event(1)`. Ordinary array multiplication, addition and matrix
multiplication `matrix @ vector` work inside the function.
An observed site uses `numpyro.sample("y", distribution, obs=y)`.
Normal accepts vector locations, broadcasting a scalar scale.

Unrelated scalar-location example (not the target hierarchical model):
```python
def example(observation=None):
    location = numpyro.sample("location", dist.Normal(0.0, 1.0))
    numpyro.sample("observation", dist.Normal(location, 1.0), obs=observation)
```

Do not import or run MCMC, NUTS, Predictive or ArviZ. The evaluator owns execution.
