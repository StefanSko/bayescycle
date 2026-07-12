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
class Measurement:
    offset = Data.scalar()
    location = Param(Normal(offset, 1.0))
    centered = location - offset
    values = Observed(Normal(centered, 1.0))

@model
class ComposedMeasurements:
    first = Submodel(Measurement)
    second = Submodel(Measurement)
    contrast = first.centered - second.centered
    comparison = Observed(Normal(contrast, 1.0))

del Measurement
