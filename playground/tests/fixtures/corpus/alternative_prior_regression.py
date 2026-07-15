from bayeswire import Data, Observed, Param, model, with_prior
from bayeswire.constraints import Positive
from bayeswire.distributions import HalfNormal, Normal


@model
class RegressionTarget:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(HalfNormal(1.0), constraint=Positive())
    x = Data.vector()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))


@model
class SimulationPrior:
    alpha = Param(Normal(-0.1, 0.2))
    beta = Param(Normal(1.25, 0.1))
    sigma = Param(HalfNormal(2.0), constraint=Positive())


AlternativeRegression = with_prior(RegressionTarget, prior=SimulationPrior)
