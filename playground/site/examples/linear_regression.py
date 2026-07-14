from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Normal, Truncated


@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())
    x = Data.vector()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))
