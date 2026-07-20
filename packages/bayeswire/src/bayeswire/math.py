"""Symbolic math helpers for model declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bayeswire.model._deferred import DeferredMatVecOp, DeferredUnaryOp

__all__ = ["LinearMap", "exp", "linear", "sigmoid"]


class LinearMap(Protocol):
    """Authoring capability for applying one rank-2 linear map."""

    def apply(self, vector: object, /) -> DeferredMatVecOp:
        """Apply this map to one rank-1 vector expression."""
        ...


@dataclass(frozen=True)
class _LinearMap:
    """Immutable authoring value holding an unresolved matrix operand."""

    matrix: object

    def apply(self, vector: object, /) -> DeferredMatVecOp:
        """Lower this application to deferred matrix-vector syntax."""
        return DeferredMatVecOp(matrix=self.matrix, vector=vector)


def linear(matrix: object, /) -> LinearMap:
    """Return a reusable authoring-side linear map for ``matrix``."""
    return _LinearMap(matrix=matrix)


def exp(value: object) -> DeferredUnaryOp:
    """Return a declaration-time symbolic exponential expression."""
    return DeferredUnaryOp("exp", value)


def sigmoid(value: object) -> DeferredUnaryOp:
    """Return a declaration-time symbolic logistic sigmoid expression."""
    return DeferredUnaryOp("sigmoid", value)
