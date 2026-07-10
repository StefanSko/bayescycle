from __future__ import annotations

from typing import cast

from bayeswire import (
    Data,
    Dim,
    Observed,
    Param,
    Submodel,
    dimension_metadata_to_dict,
    model,
    model_dimensions,
)
from bayeswire.distributions import Normal
from bayeswire.ir import meta_to_dict
from bayeswire.model.decorator import ModelMeta


def test_declared_dimension_metadata_is_exposed_for_linear_regression() -> None:
    predictor = Dim("predictor", coords=("x1", "x2", "x3"))
    obs = Dim("obs")

    @model
    class LinearRegression:
        x = Data.matrix(5, 3, dims=(obs, predictor))
        beta = Param(Normal(0.0, 1.0), size=3, dims=(predictor,))
        alpha = Param(Normal(0.0, 1.0))
        y = Observed(Normal(alpha, 1.0), dims=(obs,))

    assert dimension_metadata_to_dict(model_dimensions(LinearRegression)) == {
        "dims": {
            "x": ["obs", "predictor"],
            "beta": ["predictor"],
            "alpha": [],
            "y": ["obs"],
        },
        "coords": {"predictor": ["x1", "x2", "x3"]},
    }
    meta = cast(ModelMeta, getattr(LinearRegression, "_model_meta"))  # noqa: B009
    model_node = meta_to_dict(meta)["model"]
    assert isinstance(model_node, dict)
    assert "dims" not in model_node


def test_submodel_prefixes_dimension_variables_labels_and_coordinates() -> None:
    group = Dim("group", coords=("a", "b"))

    @model
    class Effects:
        n = Data.scalar()
        theta = Param(Normal(0.0, 1.0), size=2, dims=(group,))
        y = Observed(Normal(theta, 1.0), dims=(group,))

    @model
    class PairedEffects:
        treatment = Submodel(Effects)
        control = Submodel(Effects)

    assert dimension_metadata_to_dict(model_dimensions(PairedEffects)) == {
        "dims": {
            "treatment.theta": ["treatment.group"],
            "treatment.y": ["treatment.group"],
            "control.theta": ["control.group"],
            "control.y": ["control.group"],
        },
        "coords": {
            "treatment.group": ["a", "b"],
            "control.group": ["a", "b"],
        },
    }
