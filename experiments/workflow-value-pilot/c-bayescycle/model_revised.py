"""Scripted revised proposal; awaits human scientific review."""
from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import HalfNormal, Normal
from bayeswire.math import linear


@model
class ClinicModel:
    alpha = Param(Normal(0.0, 2.0))
    beta = Param(Normal(0.0, 0.25))
    tau = Param(HalfNormal(1.0), constraint=Positive())
    z = Param(Normal(0.0, 1.0), size=8)
    sigma = Param(HalfNormal(1.0), constraint=Positive())
    x = Data.vector()
    clinic_design = Data.matrix()
    mu = alpha + tau * linear(clinic_design).apply(z) + beta * x
    y = Observed(Normal(mu, sigma))
