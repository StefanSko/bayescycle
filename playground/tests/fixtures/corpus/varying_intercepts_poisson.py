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
class VaryingInterceptsPoisson:
    n_groups = Data.scalar()
    group_idx = Data.vector()
    x = Data.vector()

    alpha_pop = Param(Normal(0.0, 0.5))
    sigma_alpha = Param(HalfNormal(0.4), constraint=Positive())
    z_alpha = Param(Normal(0.0, 1.0), size=n_groups)

    alpha = alpha_pop + sigma_alpha * z_alpha
    eta = alpha[group_idx] + 0.25 * x
    y = Observed(Poisson(exp(eta)))
