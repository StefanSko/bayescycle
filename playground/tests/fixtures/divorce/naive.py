from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Exponential, Normal


@model
class DivorceNaive:
    alpha = Param(Normal(0.0, 0.2))
    beta_m = Param(Normal(0.0, 0.5))
    sigma = Param(Exponential(1.0), constraint=Positive())
    M = Data.vector()
    A = Data.vector()
    mu = alpha + beta_m * M
    D = Observed(Normal(mu, sigma))
