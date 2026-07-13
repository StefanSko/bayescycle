from dataclasses import replace

from bayeswire import Data, Dim, Observed, Param, PartiallyObserved, Submodel, model
from bayeswire.constraints import Interval, Ordered, Positive, UnitInterval
from bayeswire.distributions import (
    Bernoulli, Beta, Exponential, HalfNormal, MultivariateNormal, Normal,
    OrderedLogistic, Poisson, Truncated,
)
from bayeswire.math import exp
from bayeswire.model.decorator import ResolvedStochasticSite

@model
class VectorBoundsNamedOwner:
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    missing_upper = Data.vector(n_mis)

    y = PartiallyObserved.vector(
        Exponential(1.0),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
        missing_upper=missing_upper,
    )

declared_meta = VectorBoundsNamedOwner._model_meta
owner = declared_meta.stochastic_sites[0]
factor = ResolvedStochasticSite(
    name="penalty",
    distribution=Normal(0.0, 1.0),
    value=owner.value,
)
# The eDSL has no general Factor declaration yet. Construct the adversarial
# resolved metadata explicitly: a differently named full-vector factor comes
# before the same-name PartiallyObserved owner and must not supply its support.
meta = replace(declared_meta, stochastic_sites=(factor, owner))

VectorBoundsNamedOwner._model_meta = meta
