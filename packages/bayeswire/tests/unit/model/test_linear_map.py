"""Focused tests for explicit reusable linear-map authoring."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bayeswire import Data, Param, model
from bayeswire.distributions import Normal
from bayeswire.math import LinearMap, linear
from bayeswire.model import model_meta
from bayeswire.model._deferred import DeferredMatVecOp
from bayeswire.model.decorator import _resolve_declaration_expr
from bayeswire.model.expr import BinOp, DataRef, MatVecOp, ParamRef


def apply_linear(mapping: LinearMap, vector: object) -> DeferredMatVecOp:
    """Exercise the public typing contract without depending on its implementation."""
    return mapping.apply(vector)


def test_linear_returns_immutable_typed_application_capability() -> None:
    matrix = object()
    vector = object()

    mapping: LinearMap = linear(matrix)
    product = apply_linear(mapping, vector)

    assert product.matrix is matrix
    assert product.vector is vector
    with pytest.raises(FrozenInstanceError):
        mapping.__setattr__("matrix", object())


def test_named_linear_map_can_be_reused_without_entering_model_metadata() -> None:
    @model
    class ReusedLinearMap:
        matrix = Data.matrix(2, 2)
        first = Param(Normal(0.0, 1.0), size=2)
        second = Param(Normal(0.0, 1.0), size=2)

        transform = linear(matrix)
        transformed_first = transform.apply(first)
        transformed_second = transform.apply(second)

    meta = model_meta(ReusedLinearMap)

    assert tuple(meta.expressions) == ("transformed_first", "transformed_second")
    assert meta.expressions["transformed_first"] == MatVecOp(DataRef("matrix"), ParamRef("first"))
    assert meta.expressions["transformed_second"] == MatVecOp(DataRef("matrix"), ParamRef("second"))
    assert "transform" not in meta.params
    assert "transform" not in meta.data


def test_linear_resolves_composed_matrix_operand_recursively() -> None:
    scale = Param(Normal(0.0, 1.0))
    matrix = Data.matrix()
    vector = Param(Normal(0.0, 1.0), size=3)

    resolved = _resolve_declaration_expr(
        linear(scale * matrix).apply(vector),
        {
            scale.symbol: "scale",
            matrix.symbol: "matrix",
            vector.symbol: "vector",
        },
    )

    assert resolved == MatVecOp(
        BinOp("*", ParamRef("scale"), DataRef("matrix")),
        ParamRef("vector"),
    )


def test_declarations_do_not_support_direct_matrix_vector_operator() -> None:
    matrix = Data.matrix()
    vector = Param(Normal(0.0, 1.0), size=3)

    with pytest.raises(TypeError, match="unsupported operand type"):
        eval("matrix @ vector", {}, {"matrix": matrix, "vector": vector})


def test_final_expressions_do_not_support_direct_matrix_vector_operator() -> None:
    matrix = DataRef("matrix")
    vector = ParamRef("vector")

    with pytest.raises(TypeError, match="unsupported operand type"):
        eval("matrix @ vector", {}, {"matrix": matrix, "vector": vector})
