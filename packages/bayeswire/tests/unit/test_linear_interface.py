"""Interface checks for the non-executing linear-map authoring experiment."""

from __future__ import annotations

import pytest

from bayeswire.math import LinearMap, linear


def test_linear_returns_typed_application_capability() -> None:
    matrix = object()

    mapping: LinearMap = linear(matrix)

    with pytest.raises(NotImplementedError, match=r"linear\(\.\.\.\)\.apply"):
        mapping.apply(object())
