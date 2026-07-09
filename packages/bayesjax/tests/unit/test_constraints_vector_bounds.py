"""Tests for resolved per-coordinate vector bounds constraints."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bayesjax._backends.jax.constraints import (
    ResolvedVectorBounds,
    inverse_transform,
    log_abs_det_jacobian,
    transform,
)


def _assert_finite_array(value: jax.Array) -> None:
    assert bool(jnp.all(jnp.isfinite(value)))


def _assert_branch_matches_autodiff(
    constraint: ResolvedVectorBounds, unconstrained: jax.Array
) -> None:
    constrained = inverse_transform(constraint, unconstrained)
    round_trip = transform(constraint, constrained)
    log_jac = log_abs_det_jacobian(constraint, unconstrained)
    jacobian = jax.jacfwd(lambda u: inverse_transform(constraint, u))(unconstrained)
    expected_log_jac = jnp.log(jnp.abs(jnp.diag(jacobian)))
    log_jac_grad = jax.grad(lambda u: jnp.sum(log_abs_det_jacobian(constraint, u)))(unconstrained)

    _assert_finite_array(constrained)
    _assert_finite_array(round_trip)
    _assert_finite_array(log_jac)
    _assert_finite_array(expected_log_jac)
    _assert_finite_array(log_jac_grad)
    assert jnp.allclose(round_trip, unconstrained, rtol=2e-3, atol=2e-3)
    assert jnp.allclose(log_jac, expected_log_jac, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    ("constraint", "unconstrained"),
    [
        (
            ResolvedVectorBounds(
                lower=jnp.asarray([-4.0, -1.0, 0.5, 2.0, 10.0]),
                upper=None,
            ),
            jnp.asarray([-10.0, -5.0, 0.0, 5.0, 10.0]),
        ),
        (
            ResolvedVectorBounds(
                lower=None,
                upper=jnp.asarray([-2.0, 0.0, 3.0, 10.0, 20.0]),
            ),
            jnp.asarray([-10.0, -5.0, 0.0, 5.0, 10.0]),
        ),
        (
            ResolvedVectorBounds(
                lower=jnp.asarray([-5.0, -1.0, 0.25, 2.0, 4.0]),
                upper=jnp.asarray([-1.0, 0.5, 2.25, 10.0, 20.0]),
            ),
            jnp.asarray([-10.0, -5.0, 0.0, 5.0, 10.0]),
        ),
    ],
)
def test_vector_bounds_round_trip_and_jacobian_are_stable(
    constraint: ResolvedVectorBounds,
    unconstrained: jax.Array,
) -> None:
    _assert_branch_matches_autodiff(constraint, unconstrained)


def test_vector_bounds_rejects_missing_sides() -> None:
    with pytest.raises(TypeError, match="at least one"):
        ResolvedVectorBounds(lower=None, upper=None)
