from dataclasses import replace

from bayeswire import Data, Dim, Observed, Param, PartiallyObserved, Submodel, model
from bayeswire.constraints import Interval, Ordered, Positive, UnitInterval
from bayeswire.distributions import (
    Bernoulli, Beta, Exponential, HalfNormal, MultivariateNormal, Normal,
    OrderedLogistic, Poisson, Truncated,
)
from bayeswire.math import exp
from bayeswire.model.decorator import ResolvedStochasticSite

school = Dim("school", coords=("A", "B", "C", "D", "E", "F", "G", "H"))

@model
class EightSchoolsNonCentered:
    n_schools = Data.scalar()
    sigma = Data.vector(n_schools, dims=(school,))

    mu = Param(Normal(0.0, 5.0))
    tau = Param(HalfNormal(5.0), constraint=Positive())
    z = Param(Normal(0.0, 1.0), size=n_schools, dims=(school,))
    theta = mu + tau * z
    y = Observed(Normal(theta, sigma), dims=(school,))
