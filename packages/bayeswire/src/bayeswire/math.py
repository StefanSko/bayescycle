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
class _DeferredLinearMap:
    """Unfinished authoring state returned by experimental ``linear``."""

    matrix: object

    def apply(self, vector: object, /) -> DeferredMatVecOp:
        """Apply this map to one rank-1 vector expression."""
        raise NotImplementedError("linear(...).apply(...) is an interface-only experiment")


def linear(matrix: object, /) -> LinearMap:
    """Interpret an expression as a rank-2 linear map awaiting application."""
    return _DeferredLinearMap(matrix=matrix)


def exp(value: object) -> DeferredUnaryOp:
    """Return a declaration-time symbolic exponential expression."""
    return DeferredUnaryOp("exp", value)


def sigmoid(value: object) -> DeferredUnaryOp:
    """Return a declaration-time symbolic logistic sigmoid expression."""
    return DeferredUnaryOp("sigmoid", value)
