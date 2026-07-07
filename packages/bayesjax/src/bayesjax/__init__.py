"""bayesjax — the JAX/BlackJAX sampling backend for bayeswire models."""

from bayesjax.model import BoundModel, bind_model

__all__ = [
    "BoundModel",
    "bind_model",
]
