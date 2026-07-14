"""Target factorization retains every non-Param value and factor."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bayeswire import (
    Data,
    Observed,
    Param,
    PartiallyObserved,
    Submodel,
    model,
    model_dimensions,
    with_prior,
)
from bayeswire.distributions import Normal
from bayeswire.ir import bindable_from_meta
from bayeswire.model import ModelMeta, model_meta
from bayeswire.model.decorator import ResolvedStochasticSite
from bayeswire.model.dimensions import ResolvedModelDimensions, ResolvedVariableDims
from bayeswire.model.expr import ConstNode, DataRef, ParamRef


@model
class ThetaPrior:
    theta = Param(Normal(2.0, 0.5))


def _rebuilt_like(model_cls: type[object], meta: ModelMeta) -> type[object]:
    return bindable_from_meta(meta, dimensions=model_dimensions(model_cls))


def test_target_keeps_derived_expressions_observed_nodes_and_factors() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        x = Data.vector()
        mean = theta * x
        y = Observed(Normal(mean, 1.0))

    meta = model_meta(Target)
    owner, observed = meta.stochastic_sites
    factor = ResolvedStochasticSite(
        name="regularizer",
        distribution=Normal(ConstNode(0.0), ConstNode(2.0)),
        value=ParamRef("theta"),
    )
    target = _rebuilt_like(
        Target,
        replace(meta, stochastic_sites=(factor, owner, observed)),
    )

    composed = with_prior(target, prior=ThetaPrior)
    result = model_meta(composed)

    assert tuple(result.expressions) == ("mean",)
    assert tuple(node.name for node in result.observed_nodes) == ("y",)
    assert tuple(site.name for site in result.stochastic_sites) == (
        "theta",
        "regularizer",
        "y",
    )


def test_target_removes_a_renamed_param_owner_by_structure() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        y = Observed(Normal(theta, 1.0))

    meta = model_meta(Target)
    renamed_owner = replace(meta.stochastic_sites[0], name="authored_theta_prior")
    target = _rebuilt_like(
        Target,
        replace(meta, stochastic_sites=(renamed_owner, meta.stochastic_sites[1])),
    )

    composed = with_prior(target, prior=ThetaPrior)

    assert tuple(site.name for site in model_meta(composed).stochastic_sites) == ("theta", "y")


def test_target_retains_partially_observed_free_value_and_owner() -> None:
    @model
    class PartialTarget:
        theta = Param(Normal(0.0, 1.0))
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed = Data.vector(n_obs)
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        value = PartiallyObserved.vector(
            Normal(theta, 1.0),
            length=n,
            observed=observed,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
        )

    composed = with_prior(PartialTarget, prior=ThetaPrior)
    result = model_meta(composed)

    assert tuple(result.free_values) == ("theta", "value")
    assert tuple(site.name for site in result.stochastic_sites) == ("theta", "value")


def test_target_retains_closed_submodel_contents() -> None:
    @model
    class Child:
        child_theta = Param(Normal(0.0, 1.0))
        child_y = Observed(Normal(child_theta, 1.0))

    @model
    class Parent:
        child = Submodel(Child)
        parent_y = Observed(Normal(child.child_theta, 1.0))

    target_meta = model_meta(Parent)
    child_param = target_meta.params["child.child_theta"]
    child_free = target_meta.free_values["child.child_theta"]
    child_owner = target_meta.stochastic_sites[0]
    source_meta = ModelMeta(
        params={"child.child_theta": replace(child_param, distribution=Normal(2.0, 0.5))},
        data={},
        observed_nodes=(),
        expressions={},
        free_values={"child.child_theta": child_free},
        stochastic_sites=(replace(child_owner, distribution=Normal(2.0, 0.5)),),
    )
    source_dimensions = ResolvedModelDimensions(
        variables={"child.child_theta": ResolvedVariableDims(())},
        coords={},
    )
    source = bindable_from_meta(source_meta, dimensions=source_dimensions)

    composed = with_prior(Parent, prior=source)
    result = model_meta(composed)

    assert tuple(result.params) == ("child.child_theta",)
    assert tuple(node.name for node in result.observed_nodes) == (
        "child.child_y",
        "parent_y",
    )
    assert tuple(site.name for site in result.stochastic_sites) == (
        "child.child_theta",
        "child.child_y",
        "parent_y",
    )


def test_target_retains_data_dependent_parameter_sizes() -> None:
    @model
    class SizedTarget:
        n = Data.scalar()
        theta = Param(Normal(0.0, 1.0), size=n)
        y = Observed(Normal(theta, 1.0))

    @model
    class SizedPrior:
        n = Data.scalar()
        theta = Param(Normal(2.0, 0.5), size=n)

    composed = with_prior(SizedTarget, prior=SizedPrior)
    result = model_meta(composed)

    assert tuple(result.data) == ("n",)
    assert result.params["theta"].size == DataRef("n")
    assert result.free_values["theta"].size == DataRef("n")


@pytest.mark.parametrize("owner_kind", ["missing", "duplicate", "indirect"])
def test_target_rejects_invalid_param_owners(owner_kind: str) -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        y = Observed(Normal(theta, 1.0))

    meta = model_meta(Target)
    owner, observed = meta.stochastic_sites
    if owner_kind == "missing":
        sites = (observed,)
    elif owner_kind == "duplicate":
        sites = (owner, replace(owner, name="second_prior"), observed)
    else:
        sites = (replace(owner, value=ConstNode(0.0)), observed)
    target = _rebuilt_like(Target, replace(meta, stochastic_sites=sites))

    with pytest.raises(ValueError, match="target parameter 'theta'.*exactly one direct owner"):
        with_prior(target, prior=ThetaPrior)


@pytest.mark.parametrize("free_slot_kind", ["missing", "constraint", "size"])
def test_target_rejects_inconsistent_param_free_slots(free_slot_kind: str) -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        y = Observed(Normal(theta, 1.0))

    meta = model_meta(Target)
    if free_slot_kind == "missing":
        free_values = {"other": meta.free_values["theta"]}
    elif free_slot_kind == "constraint":
        from bayeswire.constraints import Positive

        free_values = {"theta": replace(meta.free_values["theta"], constraint=Positive())}
    else:
        free_values = {"theta": replace(meta.free_values["theta"], size=2)}
    target = _rebuilt_like(Target, replace(meta, free_values=free_values))

    with pytest.raises(ValueError, match="target.*free slot.*Param"):
        with_prior(target, prior=ThetaPrior)


@pytest.mark.parametrize("owner_kind", ["missing", "duplicate"])
def test_target_rejects_invalid_partially_observed_owner(owner_kind: str) -> None:
    @model
    class PartialTarget:
        theta = Param(Normal(0.0, 1.0))
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed = Data.vector(n_obs)
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        value = PartiallyObserved.vector(
            Normal(theta, 1.0),
            length=n,
            observed=observed,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
        )

    meta = model_meta(PartialTarget)
    param_owner, partial_owner = meta.stochastic_sites
    if owner_kind == "missing":
        sites = (param_owner,)
    else:
        sites = (param_owner, partial_owner, replace(partial_owner))
    target = _rebuilt_like(PartialTarget, replace(meta, stochastic_sites=sites))

    with pytest.raises(ValueError, match="target free value 'value'.*canonical owner"):
        with_prior(target, prior=ThetaPrior)


def test_target_rejects_missing_observed_owner() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        y = Observed(Normal(theta, 1.0))

    meta = model_meta(Target)
    target = _rebuilt_like(Target, replace(meta, stochastic_sites=(meta.stochastic_sites[0],)))

    with pytest.raises(ValueError, match="target Observed 'y'.*exactly one direct owner"):
        with_prior(target, prior=ThetaPrior)
