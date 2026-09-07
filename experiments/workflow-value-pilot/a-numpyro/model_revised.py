"""Proposed scripted revision: only beta prior SD changes from 1 to 0.25."""
import numpyro
import numpyro.distributions as dist


def model(x, clinic_design, y=None):
    alpha = numpyro.sample('alpha', dist.Normal(0., 2.))
    beta = numpyro.sample('beta', dist.Normal(0., 0.25))
    tau = numpyro.sample('tau', dist.HalfNormal(1.))
    z = numpyro.sample('z', dist.Normal(0., 1.).expand([8]).to_event(1))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1.))
    mu = alpha + tau * (clinic_design @ z) + beta * x
    with numpyro.plate('observations', x.shape[0]):
        numpyro.sample('y', dist.Normal(mu, sigma), obs=y)
