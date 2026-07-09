"""Per-coordinate vector bound constraint metadata."""

from __future__ import annotations

from dataclasses import dataclass

from bayeswire.model.expr import DataRef


@dataclass(frozen=True)
class VectorBounds:
    """Constraint metadata for per-coordinate vector lower and/or upper bounds."""

    lower: DataRef | None = None
    upper: DataRef | None = None

    def __post_init__(self) -> None:
        """Validate that at least one side of the vector bounds is present."""
        if self.lower is None and self.upper is None:
            raise TypeError("VectorBounds requires at least one of lower or upper")
