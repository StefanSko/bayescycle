"""Immutable authoring-time prior composition."""

from __future__ import annotations


def with_prior(target: object, *, prior: object) -> type[object]:
    """Return a new closed model using ``prior`` for ``target`` parameters."""
    del target, prior
    raise NotImplementedError("with_prior(...) is not implemented")
