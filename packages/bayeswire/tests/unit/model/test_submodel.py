"""Public behavior for closed, namespaced model composition."""

from __future__ import annotations

from typing import Protocol, cast

import pytest

from bayeswire import Data, Observed, Param, PartiallyObserved, Submodel, model
from bayeswire.constraints import Positive, VectorBounds
from bayeswire.distributions import HalfNormal, Normal
from bayeswire.distributions.core import DistributionParameter
from bayeswire.model import model_meta
from bayeswire.model._data_schema import DataDimRef, ResolvedDataShapeSchema
from bayeswire.model.expr import BinOp, DataRef, ParamRef, VectorScatterOp


class NormalFields(Protocol):
    loc: object
    scale: object


def dist_param(value: object) -> DistributionParameter:
    return value


def normal_fields(value: object) -> NormalFields:
    assert isinstance(value, Normal)
    return cast(NormalFields, value)


def test_submodel_flattens_complete_model_under_closed_namespace() -> None:
    @model
    class GroupEffects:
        n_groups = Data.scalar()
        mu = Param(Normal(0.0, 1.0))
        sigma = Param(HalfNormal(0.5), constraint=Positive())
        z = Param(Normal(0.0, 1.0), size=n_groups)
        theta = mu + sigma * z
        measurements = Observed(Normal(theta, 1.0))

    @model
    class OutcomeModel:
        effects = Submodel(GroupEffects)
        copied = Param(Normal(0.0, 1.0), size=effects.n_groups)
        outcome = Observed(Normal(dist_param(effects.theta + copied), 1.0))

    meta = model_meta(OutcomeModel)

    assert tuple(meta.params) == (
        "effects.mu",
        "effects.sigma",
        "effects.z",
        "copied",
    )
    assert tuple(meta.data) == ("effects.n_groups",)
    assert tuple(node.name for node in meta.observed_nodes) == (
        "effects.measurements",
        "outcome",
    )
    assert tuple(meta.expressions) == ("effects.theta",)
    assert tuple(meta.free_values) == (
        "effects.mu",
        "effects.sigma",
        "effects.z",
        "copied",
    )
    assert tuple(site.name for site in meta.stochastic_sites) == (
        "effects.mu",
        "effects.sigma",
        "effects.z",
        "effects.measurements",
        "copied",
        "outcome",
    )
    assert meta.params["effects.z"].size == DataRef("effects.n_groups")
    assert meta.params["copied"].size == DataRef("effects.n_groups")
    assert meta.expressions["effects.theta"] == BinOp(
        "+",
        ParamRef("effects.mu"),
        BinOp("*", ParamRef("effects.sigma"), ParamRef("effects.z")),
    )
    assert (
        normal_fields(meta.observed_nodes[0].distribution).loc == meta.expressions["effects.theta"]
    )
    assert normal_fields(meta.observed_nodes[1].distribution).loc == BinOp(
        "+",
        meta.expressions["effects.theta"],
        ParamRef("copied"),
    )


def test_submodel_instances_are_independent_and_may_form_the_complete_parent() -> None:
    @model
    class Measurement:
        location = Param(Normal(0.0, 1.0))
        values = Observed(Normal(location, 1.0))

    @model
    class PairedMeasurements:
        first = Submodel(Measurement)
        second = Submodel(Measurement)

    meta = model_meta(PairedMeasurements)

    assert tuple(meta.params) == ("first.location", "second.location")
    assert tuple(node.name for node in meta.observed_nodes) == (
        "first.values",
        "second.values",
    )
    assert tuple(site.name for site in meta.stochastic_sites) == (
        "first.location",
        "first.values",
        "second.location",
        "second.values",
    )


def test_submodel_data_can_shape_parent_declarations() -> None:
    @model
    class SizedPrior:
        n = Data.scalar()
        values = Param(Normal(0.0, 1.0), size=n)

    @model
    class Parent:
        prior = Submodel(SizedPrior)
        copied_data = Data.vector(prior.n)
        copied_param = Param(Normal(prior.values, 1.0), size=prior.n)

    meta = model_meta(Parent)

    assert meta.data["copied_data"].schema == ResolvedDataShapeSchema((DataDimRef("prior.n"),))
    assert meta.params["copied_param"].size == DataRef("prior.n")
    assert normal_fields(meta.params["copied_param"].distribution).loc == ParamRef("prior.values")


def test_submodel_exposes_nested_and_partially_observed_values() -> None:
    @model
    class Partial:
        observed_values = Data.vector(1)
        observed_idx = Data.vector(1)
        missing_idx = Data.vector(1)
        missing_lower = Data.vector(1)
        y = PartiallyObserved.vector(
            Normal(0.0, 1.0),
            length=2,
            observed=observed_values,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
            missing_lower=missing_lower,
        )

    @model
    class Inner:
        partial = Submodel(Partial)

    @model
    class Outer:
        inner = Submodel(Inner)
        z = Observed(Normal(inner.partial.y, 1.0))

    meta = model_meta(Outer)

    assert tuple(meta.free_values) == ("inner.partial.y",)
    assert meta.free_values["inner.partial.y"].constraint == VectorBounds(
        lower=DataRef("inner.partial.missing_lower"),
        upper=None,
    )
    partial_site = meta.stochastic_sites[0]
    assert partial_site.name == "inner.partial.y"
    assert isinstance(partial_site.value, VectorScatterOp)
    assert partial_site.value.observed_values == DataRef("inner.partial.observed_values")
    assert normal_fields(meta.observed_nodes[-1].distribution).loc == partial_site.value


def test_submodel_data_can_feed_parent_partially_observed_declaration() -> None:
    @model
    class PartialInputs:
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed_values = Data.vector(n_obs)
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        missing_lower = Data.vector(n_mis)
        anchor = Param(Normal(0.0, 1.0))

    @model
    class Parent:
        inputs = Submodel(PartialInputs)
        missing_upper = Data.vector(inputs.n_mis)
        y = PartiallyObserved.vector(
            Normal(inputs.anchor, 1.0),
            length=inputs.n,
            observed=inputs.observed_values,
            observed_idx=inputs.observed_idx,
            missing_idx=inputs.missing_idx,
            missing_lower=inputs.missing_lower,
            missing_upper=missing_upper,
        )

    meta = model_meta(Parent)

    assert meta.data["inputs.missing_idx"].schema == ResolvedDataShapeSchema(
        (DataDimRef("inputs.n_mis"),)
    )
    assert meta.free_values["y"].size == DataRef("inputs.n_mis")
    assert meta.free_values["y"].constraint == VectorBounds(
        lower=DataRef("inputs.missing_lower"),
        upper=DataRef("missing_upper"),
    )
    site = meta.stochastic_sites[-1]
    assert isinstance(site.value, VectorScatterOp)
    assert site.value.length == DataRef("inputs.n")
    assert site.value.observed_idx == DataRef("inputs.observed_idx")
    assert site.value.observed_values == DataRef("inputs.observed_values")
    assert site.value.missing_idx == DataRef("inputs.missing_idx")


def test_submodel_rejects_invalid_classes_aliases_and_observed_references() -> None:
    class Undecorated:
        pass

    with pytest.raises(TypeError, match="decorated with @model"):
        Submodel(Undecorated)

    @model
    class Component:
        mu = Param(Normal(0.0, 1.0))
        y = Observed(Normal(mu, 1.0))

    with pytest.raises(TypeError, match="unexpected keyword"):
        Submodel(Component, n=3)  # ty: ignore[unknown-argument]

    shared = Submodel(Component)

    with pytest.raises(ValueError, match="Declaration aliases are not supported"):

        @model
        class Aliased:
            first = shared
            second = shared

    with pytest.raises(AttributeError, match="likelihood factor"):
        _ = Submodel(Component).y
