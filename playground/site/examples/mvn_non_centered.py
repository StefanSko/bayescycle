from bayeswire import Data, Observed, Param, model
from bayeswire.distributions import MultivariateNormal, Normal


@model
class MvnNonCentered:
    n = Data.scalar()
    mean = Data.vector(n)
    latent_chol = Data.matrix(n, n)
    observation_chol = Data.matrix(n, n)

    z = Param(Normal(0.0, 1.0), size=n)
    theta = mean + latent_chol @ z
    y = Observed(MultivariateNormal(theta, observation_chol))
