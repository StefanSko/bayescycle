"""Shared helpers for integration tests."""

from bayesjax.model import bind_model as _bind_model
from bayesjax.model.bound import BoundModel


def bind_model(model_cls: object, **values: object) -> BoundModel:
    """Bind model data through the public explicit-transition API."""
    return _bind_model(model_cls, values)
