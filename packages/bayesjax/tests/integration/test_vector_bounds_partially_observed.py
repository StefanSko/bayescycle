"""Integration tests for VectorBounds on censored PartiallyObserved free values."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from _helpers import bind_model
from bayeswire import Data, PartiallyObserved, model
from bayeswire.distributions import Beta, Exponential, Normal, Uniform

from bayesjax.compiler.core import compile_log_density


@model
class LowerCensoredExponential:
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    missing_lower = Data.vector(n_mis)
    y = PartiallyObserved.vector(
        Exponential(2.0),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
        missing_lower=missing_lower,
    )


@model
class IntervalCensoredNormal:
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    missing_lower = Data.vector(n_mis)
    missing_upper = Data.vector(n_mis)
    y = PartiallyObserved.vector(
        Normal(0.0, 1.0),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
        missing_lower=missing_lower,
        missing_upper=missing_upper,
    )


@model
class IntervalCensoredUniformVectorSupport:
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    support_low = Data.vector(n)
    support_high = Data.vector(n)
    missing_lower = Data.vector(n_mis)
    missing_upper = Data.vector(n_mis)
    y = PartiallyObserved.vector(
        Uniform(support_low, support_high),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
        missing_lower=missing_lower,
        missing_upper=missing_upper,
    )


@model
class LowerCensoredBeta:
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    missing_lower = Data.vector(n_mis)
    y = PartiallyObserved.vector(
        Beta(2.0, 3.0),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
        missing_lower=missing_lower,
    )


@model
class UpperCensoredBeta:
    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    missing_upper = Data.vector(n_mis)
    y = PartiallyObserved.vector(
        Beta(2.0, 3.0),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
        missing_upper=missing_upper,
    )


def _lower_censored_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "n": 4,
        "n_obs": 2,
        "n_mis": 2,
        "observed_idx": jnp.asarray([0, 2]),
        "missing_idx": jnp.asarray([1, 3]),
        "observed_values": jnp.asarray([0.5, 1.5]),
        "missing_lower": jnp.asarray([1.0, 2.0]),
    }
    values.update(overrides)
    return values


def _interval_censored_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "n": 4,
        "n_obs": 2,
        "n_mis": 2,
        "observed_idx": jnp.asarray([0, 2]),
        "missing_idx": jnp.asarray([1, 3]),
        "observed_values": jnp.asarray([0.5, -1.5]),
        "missing_lower": jnp.asarray([-1.0, 2.0]),
        "missing_upper": jnp.asarray([1.0, 4.0]),
    }
    values.update(overrides)
    return values


def _uniform_vector_support_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "n": 4,
        "n_obs": 2,
        "n_mis": 2,
        "observed_idx": jnp.asarray([0, 2]),
        "missing_idx": jnp.asarray([1, 3]),
        "observed_values": jnp.asarray([1.0, 22.0]),
        "support_low": jnp.asarray([0.0, 10.0, 20.0, 30.0]),
        "support_high": jnp.asarray([5.0, 15.0, 25.0, 35.0]),
        "missing_lower": jnp.asarray([11.0, 31.0]),
        "missing_upper": jnp.asarray([14.0, 34.0]),
    }
    values.update(overrides)
    return values


def _beta_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "n": 2,
        "n_obs": 1,
        "n_mis": 1,
        "observed_idx": jnp.asarray([0]),
        "missing_idx": jnp.asarray([1]),
        "observed_values": jnp.asarray([0.5]),
    }
    values.update(overrides)
    return values


def test_lower_bounded_censored_exponential_log_density_matches_hand_calculation() -> None:
    bound = bind_model(LowerCensoredExponential, **_lower_censored_values())
    log_density = compile_log_density(bound)
    q = jnp.asarray([-0.7, 0.4])

    missing = jnp.asarray([1.0, 2.0]) + jnp.exp(q)
    assembled = jnp.asarray([0.5, missing[0], 1.5, missing[1]])
    expected = jnp.sum(jnp.log(2.0) - 2.0 * assembled) + jnp.sum(q)

    assert jnp.allclose(log_density(q), expected)
    assert bool(jnp.all(jnp.isfinite(jax.grad(log_density)(q))))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"missing_lower": jnp.asarray([1.0])}, "y"),
        ({"missing_lower": jnp.asarray([1.0, jnp.inf])}, "y"),
        ({"missing_lower": jnp.asarray([-0.5, 1.0])}, "y"),
    ],
)
def test_lower_censored_exponential_bind_validates_vector_bounds(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        bind_model(LowerCensoredExponential, **_lower_censored_values(**overrides))


def test_interval_censored_bind_rejects_unordered_vector_bounds() -> None:
    with pytest.raises(ValueError, match="y"):
        bind_model(
            IntervalCensoredNormal,
            **_interval_censored_values(missing_upper=jnp.asarray([-2.0, 4.0])),
        )


def test_uniform_vector_support_gathers_support_at_missing_indexes() -> None:
    bound = bind_model(IntervalCensoredUniformVectorSupport, **_uniform_vector_support_values())

    assert bound.n_params == 2


def test_uniform_vector_support_rejects_bound_outside_missing_coordinate_support() -> None:
    with pytest.raises(ValueError, match="y"):
        bind_model(
            IntervalCensoredUniformVectorSupport,
            **_uniform_vector_support_values(missing_lower=jnp.asarray([9.0, 31.0])),
        )


def test_beta_lower_bound_at_upper_support_edge_is_rejected() -> None:
    with pytest.raises(ValueError, match="y"):
        bind_model(LowerCensoredBeta, **_beta_values(missing_lower=jnp.asarray([1.0])))


def test_beta_upper_bound_at_lower_support_edge_is_rejected() -> None:
    with pytest.raises(ValueError, match="y"):
        bind_model(UpperCensoredBeta, **_beta_values(missing_upper=jnp.asarray([0.0])))
