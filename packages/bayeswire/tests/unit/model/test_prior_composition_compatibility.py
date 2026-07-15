"""Exact interfaces, deterministic data sharing, and role collisions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from bayeswire import (
    Data,
    Dim,
    Observed,
    Param,
    PartiallyObserved,
    model,
    model_dimensions,
    with_prior,
)
from bayeswire.constraints import Positive
from bayeswire.distributions import HalfNormal, Normal
from bayeswire.ir import bindable_from_meta
from bayeswire.model import model_meta
from bayeswire.model._data_schema import ResolvedDataShapeSchema
from bayeswire.model.decorator import ResolvedData


def test_prior_must_supply_every_target_param() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class EmptyOfTheta:
        other = Param(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="prior is missing target parameter 'theta'"):
        with_prior(Target, prior=EmptyOfTheta)


def test_matching_params_require_equal_constraints() -> None:
    @model
    class Target:
        theta = Param(HalfNormal(1.0), constraint=Positive())

    @model
    class Prior:
        theta = Param(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="parameter 'theta'.*constraint"):
        with_prior(Target, prior=Prior)


def test_matching_params_require_equal_static_sizes() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0), size=2)

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5), size=3)

    with pytest.raises(ValueError, match="parameter 'theta'.*size"):
        with_prior(Target, prior=Prior)


def test_matching_params_require_equal_data_dependent_size_names() -> None:
    @model
    class Target:
        target_n = Data.scalar()
        theta = Param(Normal(0.0, 1.0), size=target_n)

    @model
    class Prior:
        source_n = Data.scalar()
        theta = Param(Normal(1.0, 0.5), size=source_n)

    with pytest.raises(ValueError, match="parameter 'theta'.*data-dependent size"):
        with_prior(Target, prior=Prior)


def test_shared_data_requires_equal_resolved_schemas() -> None:
    @model
    class Target:
        context = Data.vector()
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        context = Data.scalar()
        theta = Param(Normal(1.0, 0.5))

    with pytest.raises(ValueError, match="shared data 'context'.*schema"):
        with_prior(Target, prior=Prior)


def test_matching_params_require_exact_dimension_names_and_coordinates() -> None:
    target_axis = Dim("target_axis", coords=("a", "b"))
    source_axis = Dim("source_axis", coords=("a", "b"))

    @model
    class Target:
        theta = Param(Normal(0.0, 1.0), size=2, dims=(target_axis,))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5), size=2, dims=(source_axis,))

    with pytest.raises(ValueError, match="parameter 'theta'.*dimensions"):
        with_prior(Target, prior=Prior)

    source_axis_same_name = Dim("target_axis", coords=("x", "y"))

    @model
    class CoordinatePrior:
        theta = Param(Normal(1.0, 0.5), size=2, dims=(source_axis_same_name,))

    with pytest.raises(ValueError, match="parameter 'theta'.*dimensions"):
        with_prior(Target, prior=CoordinatePrior)


def test_matching_param_rejects_absent_versus_attached_empty_dimensions() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    source_without_sidecar = bindable_from_meta(model_meta(Prior))

    with pytest.raises(ValueError, match="parameter 'theta'.*dimensions"):
        with_prior(Target, prior=source_without_sidecar)


def test_shared_data_requires_exact_optional_dimensions() -> None:
    row = Dim("row", coords=("a", "b"))

    @model
    class Target:
        context = Data.vector(2, dims=(row,))
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        context = Data.vector(2)
        theta = Param(Normal(1.0, 0.5))

    with pytest.raises(ValueError, match="shared data 'context'.*dimensions"):
        with_prior(Target, prior=Prior)


def test_coordinate_name_reuse_requires_equal_values_across_retained_variables() -> None:
    source_row = Dim("row", coords=("a", "b"))
    target_row = Dim("row", coords=("x", "y"))

    @model
    class Target:
        target_context = Data.vector(2, dims=(target_row,))
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        source_context = Data.vector(2, dims=(source_row,))
        theta = Param(Normal(1.0, 0.5))

    with pytest.raises(ValueError, match="dimension 'row'.*coordinate"):
        with_prior(Target, prior=Prior)


def test_source_only_param_cannot_collide_with_target_data() -> None:
    @model
    class Target:
        auxiliary = Data.scalar()
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))
        auxiliary = Param(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="name 'auxiliary'.*Param.*target data"):
        with_prior(Target, prior=Prior)


def test_source_only_param_cannot_collide_with_target_observed_value() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        y = Observed(Normal(theta, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))
        y = Param(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="name 'y'.*Param.*target Observed"):
        with_prior(Target, prior=Prior)


def test_source_expression_cannot_collide_with_target_data() -> None:
    @model
    class Target:
        summary = Data.scalar()
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))
        summary = theta + 1.0

    with pytest.raises(ValueError, match="name 'summary'.*source expression.*target data"):
        with_prior(Target, prior=Prior)


def test_source_data_cannot_collide_with_target_expression() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        context = theta + 1.0

    @model
    class Prior:
        context = Data.scalar()
        theta = Param(Normal(1.0, 0.5))

    with pytest.raises(ValueError, match="name 'context'.*source data.*target expression"):
        with_prior(Target, prior=Prior)


def test_source_and_target_expressions_never_deduplicate() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        summary = theta + 1.0

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))
        summary = theta + 1.0

    with pytest.raises(ValueError, match="expressions collide at 'summary'"):
        with_prior(Target, prior=Prior)


def test_source_internal_param_data_collision_is_rejected() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    meta = model_meta(Prior)
    malformed = replace(
        meta,
        data={"theta": ResolvedData(ResolvedDataShapeSchema(()))},
    )
    source = bindable_from_meta(malformed, dimensions=model_dimensions(Prior))

    with pytest.raises(ValueError, match="source name 'theta'.*Param.*data"):
        with_prior(Target, prior=source)


def test_source_only_param_cannot_collide_with_target_non_param_free_value() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed = Data.vector(n_obs)
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        latent = PartiallyObserved.vector(
            Normal(theta, 1.0),
            length=n,
            observed=observed,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
        )

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))
        latent = Param(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="name 'latent'.*Param.*target free value"):
        with_prior(Target, prior=Prior)


def test_compatible_shared_data_is_emitted_once_at_source_position() -> None:
    @model
    class Target:
        shared = Data.scalar()
        target_only = Data.scalar()
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        source_only = Data.scalar()
        shared = Data.scalar()
        theta = Param(Normal(1.0, 0.5))

    composed = with_prior(Target, prior=Prior)

    assert tuple(model_meta(composed).data) == ("source_only", "shared", "target_only")
