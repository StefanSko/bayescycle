"""Public model hooks: metadata access and code-free model reconstruction."""

from __future__ import annotations

import pytest

from bayeswire.distributions import Normal
from bayeswire.ir import bindable_from_meta, meta_from_dict, meta_to_dict
from bayeswire.model import (
    Dim,
    ModelMeta,
    Observed,
    Param,
    Submodel,
    attached_model_dimensions,
    dimension_metadata_from_dict,
    dimension_metadata_to_dict,
    is_model_class,
    model,
    model_dependencies,
    model_dimensions,
    model_meta,
    resolved_free_values,
    resolved_stochastic_sites,
    with_prior,
)

obs = Dim("obs", coords=("a", "b"))


@model
class PublicHookModel:
    theta = Param(Normal(0.0, 1.0))
    y = Observed(Normal(theta, 1.0), dims=(obs,))


def test_model_meta_returns_public_metadata_for_decorated_model() -> None:
    meta = model_meta(PublicHookModel)

    assert isinstance(meta, ModelMeta)
    assert tuple(meta.params) == ("theta",)
    assert tuple(observed.name for observed in meta.observed_nodes) == ("y",)


def test_is_model_class_identifies_decorated_model_classes() -> None:
    assert is_model_class(PublicHookModel)
    assert not is_model_class(object)
    assert not is_model_class(PublicHookModel())
    assert not is_model_class(object())


def test_model_meta_rejects_non_model_objects() -> None:
    with pytest.raises(TypeError, match="@model"):
        model_meta(object)

    with pytest.raises(TypeError, match="@model"):
        model_meta(PublicHookModel())


def test_public_hooks_reject_subclasses_with_inherited_model_metadata() -> None:
    class ChildModel(PublicHookModel):
        extra = Param(Normal(0.0, 1.0))

    assert not is_model_class(ChildModel)
    with pytest.raises(TypeError, match="@model"):
        model_meta(ChildModel)


def test_resolved_free_values_returns_flat_nuts_layout() -> None:
    free_values = resolved_free_values(model_meta(PublicHookModel))

    assert tuple(free_values) == ("theta",)
    assert free_values["theta"].constraint is None
    assert free_values["theta"].size is None


def test_resolved_stochastic_sites_returns_log_density_factors() -> None:
    sites = resolved_stochastic_sites(model_meta(PublicHookModel))

    assert tuple(site.name for site in sites) == ("theta", "y")


def test_resolved_accessors_derive_layout_from_params_when_absent() -> None:
    meta = model_meta(PublicHookModel)
    legacy = ModelMeta(
        params=meta.params,
        data=meta.data,
        observed_nodes=meta.observed_nodes,
        expressions=meta.expressions,
    )

    assert tuple(resolved_free_values(legacy)) == ("theta",)
    assert tuple(site.name for site in resolved_stochastic_sites(legacy)) == ("theta", "y")


def test_model_classes_carry_no_runtime_bind_attribute() -> None:
    assert "bind" not in PublicHookModel.__dict__

    rebuilt = bindable_from_meta(model_meta(PublicHookModel))
    assert not hasattr(rebuilt, "bind")


def test_reconstructed_model_is_indistinguishable_by_the_public_hooks() -> None:
    meta = model_meta(PublicHookModel)
    rebuilt = bindable_from_meta(meta)

    assert is_model_class(rebuilt)
    assert model_meta(rebuilt) is meta
    # Without the decoded dimension sidecar there is nothing to reattach.
    assert attached_model_dimensions(rebuilt) is None


def test_round_tripped_document_reconstructs_equal_metadata() -> None:
    meta = model_meta(PublicHookModel)
    rebuilt = bindable_from_meta(meta_from_dict(meta_to_dict(meta)))

    assert model_meta(rebuilt) == meta


def test_reconstructed_model_round_trips_dimension_metadata() -> None:
    meta = model_meta(PublicHookModel)
    declared_dimensions = model_dimensions(PublicHookModel)
    document = dimension_metadata_to_dict(declared_dimensions)
    rebuilt = bindable_from_meta(meta, dimensions=dimension_metadata_from_dict(document))

    assert is_model_class(rebuilt)
    assert model_dimensions(rebuilt) == declared_dimensions
    assert attached_model_dimensions(rebuilt) == declared_dimensions
    assert dimension_metadata_to_dict(model_dimensions(rebuilt)) == document


def test_attached_model_dimensions_rejects_non_model_objects() -> None:
    with pytest.raises(TypeError, match="@model"):
        attached_model_dimensions(object)


def test_model_dependencies_reports_composition_inputs_in_target_prior_order() -> None:
    @model
    class AlternativePrior:
        theta = Param(Normal(2.0, 0.5))

    composed = with_prior(PublicHookModel, prior=AlternativePrior)

    assert model_dependencies(composed) == (PublicHookModel, AlternativePrior)
    assert isinstance(model_dependencies(composed), tuple)


def test_model_dependencies_reports_direct_submodel_targets_in_declaration_order() -> None:
    @model
    class FirstComponent:
        first = Param(Normal(0.0, 1.0))

    @model
    class SecondComponent:
        second = Param(Normal(0.0, 1.0))

    @model
    class Root:
        first = Submodel(FirstComponent)
        second = Submodel(SecondComponent)

    assert model_dependencies(Root) == (FirstComponent, SecondComponent)
    assert model_dependencies(FirstComponent) == ()


def test_model_dependencies_preserves_nested_composition_as_direct_graph() -> None:
    @model
    class FirstPrior:
        theta = Param(Normal(2.0, 0.5))

    @model
    class SecondPrior:
        theta = Param(Normal(-1.0, 0.25))

    first = with_prior(PublicHookModel, prior=FirstPrior)
    second = with_prior(first, prior=SecondPrior)

    assert model_dependencies(first) == (PublicHookModel, FirstPrior)
    assert model_dependencies(second) == (first, SecondPrior)


def test_model_dependencies_preserves_duplicate_target_and_prior_roles() -> None:
    @model
    class PriorOnly:
        theta = Param(Normal(0.0, 1.0))

    composed = with_prior(PriorOnly, prior=PriorOnly)

    assert model_dependencies(composed) == (PriorOnly, PriorOnly)


def test_model_dependencies_reports_no_graph_for_decoded_ir() -> None:
    rebuilt = bindable_from_meta(model_meta(PublicHookModel))

    assert model_dependencies(rebuilt) == ()


def test_model_dependencies_rejects_cycles() -> None:
    @model
    class First:
        first = Param(Normal(0.0, 1.0))

    @model
    class Second:
        second = Param(Normal(0.0, 1.0))

    type.__setattr__(First, "_model_dependencies", (Second,))
    type.__setattr__(Second, "_model_dependencies", (First,))

    with pytest.raises(ValueError, match="model dependency cycle.*First.*Second.*First"):
        model_dependencies(First)


def test_model_dependencies_rejects_malformed_private_state() -> None:
    @model
    class Root:
        theta = Param(Normal(0.0, 1.0))

    type.__setattr__(Root, "_model_dependencies", (object,))

    with pytest.raises(TypeError, match="dependency.*bayeswire model class"):
        model_dependencies(Root)


def test_bindable_from_meta_rejects_dimensions_for_undeclared_variables() -> None:
    meta = model_meta(PublicHookModel)
    dimensions = dimension_metadata_from_dict(
        {"dims": {"not_declared": ["obs"]}, "coords": {"obs": ["a", "b"]}}
    )

    with pytest.raises(ValueError, match="not declared by the model"):
        bindable_from_meta(meta, dimensions=dimensions)
