from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Exponential, Normal


@model
class DivorceAdjusted:
    alpha = Param(Normal(0.0, 0.2))
    beta_m = Param(Normal(0.0, 0.5))
    beta_a = Param(Normal(0.0, 0.5))
    sigma = Param(Exponential(1.0), constraint=Positive())
    M = Data.vector()
    A = Data.vector()
    mu = alpha + beta_m * M + beta_a * A
    D = Observed(Normal(mu, sigma))
