"""Source-prior classification for immutable prior composition."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bayeswire import Data, Observed, Param, PartiallyObserved, model, model_dimensions, with_prior
from bayeswire.distributions import Normal
from bayeswire.ir import bindable_from_meta
from bayeswire.model import model_meta
from bayeswire.model.decorator import ResolvedFreeValue, ResolvedStochasticSite
from bayeswire.model.expr import BinOp, ConstNode, ParamRef


@model
class ScalarTarget:
    theta = Param(Normal(0.0, 1.0))


@model
class ScalarPrior:
    theta = Param(Normal(1.0, 0.5))


def _rebuilt_like(model_cls: type[object], meta: object) -> type[object]:
    from bayeswire.model.decorator import ModelMeta

    assert isinstance(meta, ModelMeta)
    return bindable_from_meta(meta, dimensions=model_dimensions(model_cls))


def test_source_param_owner_is_structural_not_site_label() -> None:
    meta = model_meta(ScalarPrior)
    renamed = replace(meta.stochastic_sites[0], name="theta_prior")
    source = _rebuilt_like(ScalarPrior, replace(meta, stochastic_sites=(renamed,)))

    composed = with_prior(ScalarTarget, prior=source)

    assert tuple(site.name for site in model_meta(composed).stochastic_sites) == ("theta_prior",)


def test_source_rejects_observed_declarations() -> None:
    @model
    class PriorWithObserved:
        theta = Param(Normal(0.0, 1.0))
        y = Observed(Normal(theta, 1.0))

    with pytest.raises(ValueError, match="prior.*Observed"):
        with_prior(ScalarTarget, prior=PriorWithObserved)


def test_source_rejects_partially_observed_values() -> None:
    @model
    class PartialPrior:
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed = Data.vector(n_obs)
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        value = PartiallyObserved.vector(
            Normal(0.0, 1.0),
            length=n,
            observed=observed,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
        )

    with pytest.raises(ValueError, match="prior.*PartiallyObserved"):
        with_prior(ScalarTarget, prior=PartialPrior)


def test_source_rejects_non_param_free_values() -> None:
    meta = model_meta(ScalarPrior)
    free_values = dict(meta.free_values)
    free_values["aux"] = ResolvedFreeValue(constraint=None, size=None)
    sites = meta.stochastic_sites + (
        ResolvedStochasticSite(
            name="aux",
            distribution=Normal(0.0, 1.0),
            value=ParamRef("aux"),
        ),
    )
    source = _rebuilt_like(
        ScalarPrior,
        replace(meta, free_values=free_values, stochastic_sites=sites),
    )

    with pytest.raises(ValueError, match="prior free value 'aux'.*declared Param"):
        with_prior(ScalarTarget, prior=source)


def test_source_rejects_additional_density_factors() -> None:
    meta = model_meta(ScalarPrior)
    factor = ResolvedStochasticSite(
        name="penalty",
        distribution=Normal(0.0, 2.0),
        value=ParamRef("theta"),
    )
    source = _rebuilt_like(
        ScalarPrior,
        replace(meta, stochastic_sites=meta.stochastic_sites + (factor,)),
    )

    with pytest.raises(ValueError, match="prior density factor 'penalty'.*not allowed"):
        with_prior(ScalarTarget, prior=source)


@pytest.mark.parametrize("site_kind", ["missing", "duplicate", "indirect"])
def test_source_rejects_missing_duplicate_or_indirect_param_sites(site_kind: str) -> None:
    meta = model_meta(ScalarPrior)
    owner = meta.stochastic_sites[0]
    if site_kind == "missing":
        sites: tuple[ResolvedStochasticSite, ...] = (
            replace(owner, name="unrelated", value=ParamRef("not_theta")),
        )
    elif site_kind == "duplicate":
        sites = (owner, replace(owner, name="second_theta_prior"))
    else:
        sites = (
            replace(
                owner,
                value=BinOp("+", ParamRef("theta"), ConstNode(0.0)),
            ),
        )
    source = _rebuilt_like(ScalarPrior, replace(meta, stochastic_sites=sites))

    with pytest.raises(ValueError, match="prior parameter 'theta'.*exactly one direct owner"):
        with_prior(ScalarTarget, prior=source)


@pytest.mark.parametrize("free_value_kind", ["partial", "constraint", "size"])
def test_source_rejects_inconsistent_param_free_slots(free_value_kind: str) -> None:
    meta = model_meta(ScalarPrior)
    if free_value_kind == "partial":
        free_values: dict[str, ResolvedFreeValue] = {"other": ResolvedFreeValue(None, None)}
    elif free_value_kind == "constraint":
        from bayeswire.constraints import Positive

        free_values = {"theta": ResolvedFreeValue(Positive(), None)}
    else:
        free_values = {"theta": ResolvedFreeValue(None, 2)}
    source = _rebuilt_like(ScalarPrior, replace(meta, free_values=free_values))

    with pytest.raises(ValueError, match="prior.*free slot.*Param"):
        with_prior(ScalarTarget, prior=source)


def test_source_accepts_legacy_empty_free_value_map() -> None:
    meta = model_meta(ScalarPrior)
    source = _rebuilt_like(ScalarPrior, replace(meta, free_values={}))

    composed = with_prior(ScalarTarget, prior=source)

    assert tuple(model_meta(composed).free_values) == ("theta",)


def test_source_preserves_site_order_independently_for_independent_params() -> None:
    @model
    class Target:
        first = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        first = Param(Normal(1.0, 0.5))
        second = Param(Normal(-1.0, 0.25))

    meta = model_meta(Prior)
    source = _rebuilt_like(
        Prior,
        replace(meta, stochastic_sites=tuple(reversed(meta.stochastic_sites))),
    )

    composed = with_prior(Target, prior=source)
    result = model_meta(composed)

    assert tuple(result.params) == ("first", "second")
    assert tuple(result.free_values) == ("first", "second")
    assert tuple(site.name for site in result.stochastic_sites) == ("second", "first")


def test_source_accepts_ordered_hierarchical_extra_params() -> None:
    @model
    class HierarchicalPrior:
        location = Param(Normal(0.0, 1.0))
        theta = Param(Normal(location, 0.2))

    composed = with_prior(ScalarTarget, prior=HierarchicalPrior)

    assert tuple(model_meta(composed).params) == ("location", "theta")
    assert tuple(model_meta(composed).free_values) == ("location", "theta")


@pytest.mark.parametrize("dependency_kind", ["forward", "self", "cycle"])
def test_source_rejects_non_ancestral_param_dependencies(dependency_kind: str) -> None:
    @model
    class TwoParamPrior:
        theta = Param(Normal(0.0, 1.0))
        later = Param(Normal(0.0, 1.0))

    meta = model_meta(TwoParamPrior)
    params = dict(meta.params)
    sites = list(meta.stochastic_sites)
    if dependency_kind == "forward":
        theta_distribution = Normal(ParamRef("later"), ConstNode(1.0))
        params["theta"] = replace(params["theta"], distribution=theta_distribution)
        sites[0] = replace(sites[0], distribution=theta_distribution)
    elif dependency_kind == "self":
        theta_distribution = Normal(ParamRef("theta"), ConstNode(1.0))
        params["theta"] = replace(params["theta"], distribution=theta_distribution)
        sites[0] = replace(sites[0], distribution=theta_distribution)
    else:
        theta_distribution = Normal(ParamRef("later"), ConstNode(1.0))
        later_distribution = Normal(ParamRef("theta"), ConstNode(1.0))
        params["theta"] = replace(params["theta"], distribution=theta_distribution)
        params["later"] = replace(params["later"], distribution=later_distribution)
        sites[0] = replace(sites[0], distribution=theta_distribution)
        sites[1] = replace(sites[1], distribution=later_distribution)
    source = _rebuilt_like(
        TwoParamPrior,
        replace(meta, params=params, stochastic_sites=tuple(sites)),
    )

    with pytest.raises(ValueError, match="prior parameter 'theta'.*earlier Param"):
        with_prior(ScalarTarget, prior=source)


def test_source_rejects_undecorated_classes() -> None:
    class NotAModel:
        pass

    with pytest.raises(TypeError, match="prior.*bayeswire model class"):
        with_prior(ScalarTarget, prior=NotAModel)
