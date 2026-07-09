"""JAX implementations of constraint operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
from bayeswire.constraints.core import ConstrainedValue, Constraint, UnconstrainedValue
from bayeswire.constraints.interval import Interval, UnitInterval
from bayeswire.constraints.ordered import Ordered
from bayeswire.constraints.positive import Positive


@dataclass(frozen=True, eq=False)
class ResolvedVectorBounds:
    """Concrete per-coordinate bounds resolved from VectorBounds data refs."""

    lower: jax.Array | None = None
    upper: jax.Array | None = None

    def __post_init__(self) -> None:
        """Validate that at least one side and compatible shapes are present."""
        if self.lower is None and self.upper is None:
            raise TypeError("ResolvedVectorBounds requires at least one bound side")
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower.shape != self.upper.shape
        ):
            raise ValueError("ResolvedVectorBounds lower and upper shapes must match")


type JaxConstraint = Constraint | ResolvedVectorBounds


@runtime_checkable
class _PythonConstraint(Protocol):
    """Compatibility protocol for Python-defined JAX constraints."""

    def transform(self, x: ConstrainedValue) -> UnconstrainedValue:
        """Map constrained values to unconstrained values."""
        ...

    def inverse_transform(self, y: UnconstrainedValue) -> ConstrainedValue:
        """Map unconstrained values to constrained values."""
        ...

    def log_abs_det_jacobian(self, y: UnconstrainedValue) -> object:
        """Return inverse-transform log absolute determinant."""
        ...


def transform(constraint: JaxConstraint, x: ConstrainedValue) -> jax.Array:
    """Map constrained values to unconstrained values with the JAX backend."""
    if isinstance(constraint, Positive):
        return jnp.log(jnp.asarray(x))
    if isinstance(constraint, Interval):
        unit_value = (jnp.asarray(x) - constraint.lower) / constraint.width
        return jnp.log(unit_value) - jnp.log1p(-unit_value)
    if isinstance(constraint, UnitInterval):
        return transform(Interval(0.0, 1.0), x)
    if isinstance(constraint, ResolvedVectorBounds):
        constrained = jnp.asarray(x)
        if constraint.lower is None:
            if constraint.upper is None:
                raise TypeError("ResolvedVectorBounds requires at least one bound side")
            return jnp.log(constraint.upper - constrained)
        if constraint.upper is None:
            return jnp.log(constrained - constraint.lower)
        unit_value = (constrained - constraint.lower) / (constraint.upper - constraint.lower)
        return jnp.log(unit_value) - jnp.log1p(-unit_value)
    if isinstance(constraint, Ordered):
        constrained = jnp.asarray(x)
        if constrained.ndim == 0:
            raise ValueError("Ordered constraint requires vector values")
        first = constrained[..., :1]
        log_differences = jnp.log(constrained[..., 1:] - constrained[..., :-1])
        return jnp.concatenate((first, log_differences), axis=-1)
    if isinstance(constraint, _PythonConstraint):
        return cast(jax.Array, constraint.transform(x))
    raise TypeError(f"Unsupported constraint: {type(constraint).__name__}")


def inverse_transform(constraint: JaxConstraint, y: UnconstrainedValue) -> jax.Array:
    """Map unconstrained values to constrained values with the JAX backend."""
    if isinstance(constraint, Positive):
        return jnp.exp(jnp.asarray(y))
    if isinstance(constraint, Interval):
        return constraint.lower + constraint.width * jax.nn.sigmoid(jnp.asarray(y))
    if isinstance(constraint, UnitInterval):
        return inverse_transform(Interval(0.0, 1.0), y)
    if isinstance(constraint, ResolvedVectorBounds):
        unconstrained = jnp.asarray(y)
        if constraint.lower is None:
            if constraint.upper is None:
                raise TypeError("ResolvedVectorBounds requires at least one bound side")
            return constraint.upper - jnp.exp(unconstrained)
        if constraint.upper is None:
            return constraint.lower + jnp.exp(unconstrained)
        return constraint.lower + (constraint.upper - constraint.lower) * jax.nn.sigmoid(
            unconstrained
        )
    if isinstance(constraint, Ordered):
        unconstrained = jnp.asarray(y)
        if unconstrained.ndim == 0:
            raise ValueError("Ordered constraint requires vector values")
        first = unconstrained[..., :1]
        increments = jnp.exp(unconstrained[..., 1:])
        tail = first + jnp.cumsum(increments, axis=-1)
        return jnp.concatenate((first, tail), axis=-1)
    if isinstance(constraint, _PythonConstraint):
        return cast(jax.Array, constraint.inverse_transform(y))
    raise TypeError(f"Unsupported constraint: {type(constraint).__name__}")


def log_abs_det_jacobian(constraint: JaxConstraint, y: UnconstrainedValue) -> jax.Array:
    """Return inverse-transform log absolute determinant with the JAX backend."""
    if isinstance(constraint, Positive):
        return jnp.asarray(y)
    if isinstance(constraint, Interval):
        unconstrained = jnp.asarray(y)
        return (
            jnp.log(constraint.width)
            - jax.nn.softplus(-unconstrained)
            - jax.nn.softplus(unconstrained)
        )
    if isinstance(constraint, UnitInterval):
        return log_abs_det_jacobian(Interval(0.0, 1.0), y)
    if isinstance(constraint, ResolvedVectorBounds):
        unconstrained = jnp.asarray(y)
        if constraint.lower is None:
            if constraint.upper is None:
                raise TypeError("ResolvedVectorBounds requires at least one bound side")
            return unconstrained
        if constraint.upper is None:
            return unconstrained
        return (
            jnp.log(constraint.upper - constraint.lower)
            - jax.nn.softplus(-unconstrained)
            - jax.nn.softplus(unconstrained)
        )
    if isinstance(constraint, Ordered):
        unconstrained = jnp.asarray(y)
        if unconstrained.ndim == 0:
            raise ValueError("Ordered constraint requires vector values")
        return jnp.sum(unconstrained[..., 1:], axis=-1)
    if isinstance(constraint, _PythonConstraint):
        return cast(jax.Array, constraint.log_abs_det_jacobian(y))
    raise TypeError(f"Unsupported constraint: {type(constraint).__name__}")
