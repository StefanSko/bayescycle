"""Integration tests for VectorBounds on censored PartiallyObserved free values."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from _helpers import bind_model
from bayeswire import Data, PartiallyObserved, model
from bayeswire.distributions import Exponential, Normal

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
