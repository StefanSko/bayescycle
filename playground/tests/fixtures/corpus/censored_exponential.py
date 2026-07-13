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
class CensoredExponential:
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    missing_lower = Data.vector(n_mis)

    rate = Param(Exponential(1.0), constraint=Positive())
    y = PartiallyObserved.vector(
        Exponential(rate),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
        missing_lower=missing_lower,
    )
