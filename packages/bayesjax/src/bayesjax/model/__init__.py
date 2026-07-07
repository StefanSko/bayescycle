"""Model binding: the backend transition from bayeswire metadata to bound state."""

from bayesjax.model.binding import bind_model
from bayesjax.model.bound import BoundModel

__all__ = [
    "BoundModel",
    "bind_model",
]
