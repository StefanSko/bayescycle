from jaxstanv5 import Dim, Observed, Param, model
from jaxstanv5.distributions import Normal

obs = Dim("obs", coords=["a", "b", "c", "d"])


@model
class SmokeNormal:
    mu = Param(Normal(0.0, 1.0))
    y = Observed(Normal(mu, 1.0), dims=(obs,))
