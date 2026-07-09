"""Unit tests for vector-bound constraint metadata."""

from __future__ import annotations

import pytest

from bayeswire.constraints import VectorBounds
from bayeswire.model.expr import DataRef


def test_vector_bounds_requires_at_least_one_bound() -> None:
    with pytest.raises(TypeError, match="at least one"):
        VectorBounds()


def test_vector_bounds_accepts_lower_only() -> None:
    bounds = VectorBounds(lower=DataRef("lower"))

    assert bounds.lower == DataRef("lower")
    assert bounds.upper is None


def test_vector_bounds_accepts_upper_only() -> None:
    bounds = VectorBounds(upper=DataRef("upper"))

    assert bounds.lower is None
    assert bounds.upper == DataRef("upper")


def test_vector_bounds_accepts_lower_and_upper() -> None:
    bounds = VectorBounds(lower=DataRef("lower"), upper=DataRef("upper"))

    assert bounds.lower == DataRef("lower")
    assert bounds.upper == DataRef("upper")
