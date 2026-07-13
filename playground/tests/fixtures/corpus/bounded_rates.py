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
class BoundedRates:
    p = Param(Beta(2.0, 2.0), constraint=UnitInterval())
    level = Param(
        Truncated(Normal(1.0, 1.0), lower=-1.0, upper=3.0),
        constraint=Interval(-1.0, 3.0),
    )
    y = Observed(Bernoulli(p))
