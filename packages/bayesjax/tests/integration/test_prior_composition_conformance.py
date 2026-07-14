"""Composed models cross the backend boundary as ordinary closed metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import jax
import jax.numpy as jnp
from bayeswire import Data, Observed, Param, model, model_dimensions, with_prior
from bayeswire.constraints import Positive
from bayeswire.distributions import HalfNormal, Normal
from bayeswire.ir import bindable_from_meta, register_distribution
from bayeswire.model import model_meta

from bayesjax.compiler import compile_log_density
from bayesjax.inference import sample
from bayesjax.model import bind_model
from bayesjax.simulation import simulate_prior_predictive


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


@model
class HandwrittenAlternativeRegression:
    alpha = Param(Normal(-0.1, 0.2))
    beta = Param(Normal(1.25, 0.1))
    sigma = Param(HalfNormal(2.0), constraint=Positive())
    x = Data.vector()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))


@dataclass(frozen=True)
class MappedNormal:
    parameters: dict[str, object]

    def log_prob(self, x: object) -> object:
        value = cast(jax.Array, x)
        loc = cast(jax.Array, self.parameters["loc"])
        scale = cast(jax.Array, self.parameters["scale"])
        standardized = (value - loc) / scale
        return -0.5 * standardized**2 - jnp.log(scale) - 0.5 * jnp.log(2.0 * jnp.pi)


register_distribution(MappedNormal, tag="MappedNormalPriorCompositionTest")


@model
class MappedTarget:
    offset = Data.scalar()
    theta = Param(Normal(0.0, 1.0))
    y = Observed(Normal(theta + offset, 1.0))


@model
class MappedPrior:
    theta = Param(Normal(0.5, 0.75))


AlternativeRegression = with_prior(RegressionTarget, prior=SimulationPrior)
X = jnp.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
Y = jnp.asarray([-1.6, -0.3, 0.4, 1.1, 2.2])


def test_composed_binding_layout_log_density_and_gradient_match_handwritten_model() -> None:
    composed = bind_model(AlternativeRegression, {"x": X, "y": Y})
    handwritten = bind_model(HandwrittenAlternativeRegression, {"x": X, "y": Y})

    assert composed.meta == handwritten.meta
    assert composed.param_shapes == handwritten.param_shapes
    assert composed.n_params == handwritten.n_params == 3

    q = jnp.asarray([0.1, 0.2, 0.3])
    composed_log_density = compile_log_density(composed)
    handwritten_log_density = compile_log_density(handwritten)

    assert jnp.allclose(composed_log_density(q), handwritten_log_density(q))
    assert jnp.allclose(
        jax.grad(composed_log_density)(q),
        jax.grad(handwritten_log_density)(q),
    )


def test_composed_seeded_prior_predictive_matches_handwritten_model() -> None:
    composed = simulate_prior_predictive(
        AlternativeRegression,
        seed=17,
        num_samples=8,
        data={"x": X},
        observed_shapes={"y": X.shape},
    )
    handwritten = simulate_prior_predictive(
        HandwrittenAlternativeRegression,
        seed=17,
        num_samples=8,
        data={"x": X},
        observed_shapes={"y": X.shape},
    )

    assert tuple(composed.parameters) == ("alpha", "beta", "sigma")
    assert tuple(composed.observed) == ("y",)
    for name in composed.parameters:
        assert jnp.array_equal(composed.parameters[name], handwritten.parameters[name])
    assert jnp.array_equal(composed.observed["y"], handwritten.observed["y"])


def test_composed_registered_map_fields_execute_after_binding() -> None:
    meta = model_meta(MappedTarget)
    observed = meta.observed_nodes[0]
    assert isinstance(observed.distribution, Normal)
    mapped = MappedNormal(
        {
            "loc": observed.distribution.loc,
            "scale": observed.distribution.scale,
        }
    )
    target = bindable_from_meta(
        replace(
            meta,
            observed_nodes=(replace(observed, distribution=mapped),),
            stochastic_sites=tuple(
                replace(site, distribution=mapped) if site.name == "y" else site
                for site in meta.stochastic_sites
            ),
        ),
        dimensions=model_dimensions(MappedTarget),
    )
    composed = with_prior(target, prior=MappedPrior)
    bound = bind_model(
        composed,
        {"offset": jnp.asarray(0.25), "y": jnp.asarray(0.5)},
    )

    value = compile_log_density(bound)(jnp.asarray([0.1]))

    assert jnp.isfinite(value)


def test_composed_model_completes_posterior_sampling_smoke() -> None:
    bound = bind_model(AlternativeRegression, {"x": X, "y": Y})

    result = sample(
        bound,
        seed=19,
        num_chains=1,
        num_warmup=10,
        num_samples=10,
        max_tree_depth=3,
    )

    assert tuple(result.samples) == ("alpha", "beta", "sigma")
    assert all(values.shape == (1, 10) for values in result.samples.values())
    assert all(jnp.all(jnp.isfinite(values)) for values in result.samples.values())
