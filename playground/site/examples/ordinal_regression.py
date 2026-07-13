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
class OrdinalRegression:
    n_cutpoints = Data.scalar()
    x = Data.vector()

    beta = Param(Normal(0.0, 1.0))
    cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())

    eta = beta * x
    y = Observed(OrderedLogistic(eta, cutpoints))
