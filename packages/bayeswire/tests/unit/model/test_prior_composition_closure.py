"""Ordering, closure, dimensions, determinism, and input immutability."""

from __future__ import annotations

import copy
import hashlib
import math
import subprocess
import sys
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import cast

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
from bayeswire.constraints import Positive, VectorBounds
from bayeswire.distributions import Bernoulli, Normal, StudentT, Truncated
from bayeswire.distributions.core import Distribution
from bayeswire.ir import (
    UnserializableValue,
    bindable_from_meta,
    meta_to_dict,
    register_distribution,
)
from bayeswire.model import ModelMeta, model_meta
from bayeswire.model._data_schema import (
    DataDimRef,
    ResolvedDataRankSchema,
    ResolvedDataShapeSchema,
)
from bayeswire.model.decorator import ResolvedData, ResolvedStochasticSite
from bayeswire.model.dimensions import CoordValue, ResolvedModelDimensions, ResolvedVariableDims
from bayeswire.model.expr import (
    BinOp,
    ConstNode,
    DataRef,
    ExprNode,
    IndexOp,
    IndexSpec,
    ParamRef,
    UnaryOp,
    VectorScatterOp,
)


@dataclass(frozen=True)
class MappedTestDistribution:
    parameters: dict[str, object]


register_distribution(MappedTestDistribution, tag="MappedTestDistribution")


@dataclass(frozen=True)
class AlwaysEqualTestDistribution:
    marker: int

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AlwaysEqualTestDistribution)


register_distribution(AlwaysEqualTestDistribution, tag="AlwaysEqualTestDistribution")


def test_composition_freezes_all_ordered_merges() -> None:
    @model
    class Target:
        shared = Data.scalar()
        target_only = Data.scalar()
        theta = Param(Normal(0.0, 1.0))
        target_summary = theta + target_only
        y = Observed(Normal(target_summary, 1.0))

    @model
    class Prior:
        source_only = Data.scalar()
        shared = Data.scalar()
        hyper = Param(Normal(0.0, 1.0))
        theta = Param(Normal(hyper, 0.5))
        source_summary = hyper + source_only

    composed = with_prior(Target, prior=Prior)
    result = model_meta(composed)

    assert tuple(result.params) == ("hyper", "theta")
    assert tuple(result.data) == ("source_only", "shared", "target_only")
    assert tuple(result.expressions) == ("source_summary", "target_summary")
    assert tuple(result.free_values) == ("hyper", "theta")
    assert tuple(site.name for site in result.stochastic_sites) == ("hyper", "theta", "y")
    assert tuple(node.name for node in result.observed_nodes) == ("y",)


def test_composition_never_mutates_input_metadata_or_sidecars() -> None:
    axis = Dim("axis", coords=("a", "b"))

    @model
    class Target:
        theta = Param(Normal(0.0, 1.0), size=2, dims=(axis,))
        y = Observed(Normal(theta, 1.0), dims=(axis,))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5), size=2, dims=(axis,))

    target_meta = model_meta(Target)
    source_meta = model_meta(Prior)
    target_dimensions = model_dimensions(Target)
    source_dimensions = model_dimensions(Prior)
    target_before = copy.deepcopy(target_meta)
    source_before = copy.deepcopy(source_meta)
    target_dimensions_before = copy.deepcopy(target_dimensions)
    source_dimensions_before = copy.deepcopy(source_dimensions)

    with_prior(Target, prior=Prior)

    assert model_meta(Target) is target_meta
    assert model_meta(Prior) is source_meta
    assert model_meta(Target) == target_before
    assert model_meta(Prior) == source_before
    assert model_dimensions(Target) is target_dimensions
    assert model_dimensions(Prior) is source_dimensions
    assert model_dimensions(Target) == target_dimensions_before
    assert model_dimensions(Prior) == source_dimensions_before


def test_canonical_bytes_are_reproducible_across_independent_processes() -> None:
    script = """
import hashlib
from bayeswire import Data, Observed, Param, model, with_prior
from bayeswire.distributions import Normal
from bayeswire.ir import canonical_bytes
from bayeswire.model import model_meta

@model
class Target:
    theta = Param(Normal(0.0, 1.0))
    x = Data.vector()
    y = Observed(Normal(theta * x, 1.0))

@model
class Prior:
    theta = Param(Normal(2.0, 0.5))

Composed = with_prior(Target, prior=Prior)
print(hashlib.sha256(canonical_bytes(model_meta(Composed))).hexdigest())
"""

    first = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    second = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert first == second
    assert len(first) == hashlib.sha256().digest_size * 2


def test_source_must_be_closed_before_target_names_are_available() -> None:
    @model
    class Target:
        shared = Data.scalar()
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    source_meta = model_meta(Prior)
    malformed = replace(
        source_meta,
        expressions={"leaked": DataRef("shared")},
    )
    source = bindable_from_meta(malformed, dimensions=model_dimensions(Prior))

    with pytest.raises(ValueError, match="source expression 'leaked'.*unknown data 'shared'"):
        with_prior(Target, prior=source)


def test_target_must_be_closed_before_source_names_are_available() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        source_only = Param(Normal(0.0, 1.0))
        theta = Param(Normal(1.0, 0.5))

    target_meta = model_meta(Target)
    malformed = replace(
        target_meta,
        expressions={"leaked": ParamRef("source_only")},
    )
    target = bindable_from_meta(malformed, dimensions=model_dimensions(Target))

    with pytest.raises(
        ValueError, match="target expression 'leaked'.*unknown free value 'source_only'"
    ):
        with_prior(target, prior=Prior)


def test_data_dimension_references_must_name_declared_scalar_data() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        context = Data.vector()
        theta = Param(Normal(1.0, 0.5))

    source_meta = model_meta(Prior)
    malformed_data = dict(source_meta.data)
    malformed_data["context"] = ResolvedData(ResolvedDataShapeSchema((DataDimRef("missing"),)))
    source = bindable_from_meta(
        replace(source_meta, data=malformed_data),
        dimensions=model_dimensions(Prior),
    )

    with pytest.raises(ValueError, match="source data 'context'.*unknown scalar data 'missing'"):
        with_prior(Target, prior=source)


def test_source_owner_matching_ignores_custom_distribution_equality() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5))

    source_meta = model_meta(PriorDeclaration)
    source = bindable_from_meta(
        replace(
            source_meta,
            params={
                "theta": replace(
                    source_meta.params["theta"],
                    distribution=AlwaysEqualTestDistribution(1),
                ),
            },
            stochastic_sites=(
                replace(
                    source_meta.stochastic_sites[0],
                    distribution=AlwaysEqualTestDistribution(2),
                ),
            ),
        ),
        dimensions=model_dimensions(PriorDeclaration),
    )

    with pytest.raises(ValueError, match="parameter 'theta'.*direct owner.*found 0"):
        with_prior(Target, prior=source)


def test_source_closure_traverses_registered_distribution_map_fields() -> None:
    @model
    class Target:
        shared = Data.scalar()
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    source_meta = model_meta(Prior)
    distribution = MappedTestDistribution({"location": DataRef("shared")})
    source = bindable_from_meta(
        replace(
            source_meta,
            params={"theta": replace(source_meta.params["theta"], distribution=distribution)},
            stochastic_sites=(replace(source_meta.stochastic_sites[0], distribution=distribution),),
        ),
        dimensions=model_dimensions(Prior),
    )

    with pytest.raises(
        ValueError,
        match="source parameter 'theta' distribution.*unknown data 'shared'",
    ):
        with_prior(Target, prior=source)


def test_source_ancestry_traverses_registered_distribution_map_fields() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))
        later = Param(Normal(0.0, 1.0))

    source_meta = model_meta(Prior)
    distribution = MappedTestDistribution({"location": ParamRef("later")})
    params = dict(source_meta.params)
    params["theta"] = replace(params["theta"], distribution=distribution)
    sites = list(source_meta.stochastic_sites)
    sites[0] = replace(sites[0], distribution=distribution)
    source = bindable_from_meta(
        replace(source_meta, params=params, stochastic_sites=tuple(sites)),
        dimensions=model_dimensions(Prior),
    )

    with pytest.raises(ValueError, match="prior parameter 'theta'.*earlier Param"):
        with_prior(Target, prior=source)


def test_additional_factor_cannot_use_observed_bind_input_as_a_value() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        x = Data.scalar()
        y = Observed(Normal(theta, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    target_meta = model_meta(Target)
    observed_factor = ResolvedStochasticSite(
        name="penalty",
        distribution=Normal(ConstNode(0.0), ConstNode(2.0)),
        value=DataRef("y"),
    )
    malformed = bindable_from_meta(
        replace(
            target_meta,
            stochastic_sites=target_meta.stochastic_sites + (observed_factor,),
        ),
        dimensions=model_dimensions(Target),
    )

    with pytest.raises(
        ValueError,
        match="target stochastic site 'penalty' value.*unknown data 'y'",
    ):
        with_prior(malformed, prior=Prior)

    declared_data_factor = replace(observed_factor, value=DataRef("x"))
    valid = bindable_from_meta(
        replace(
            target_meta,
            stochastic_sites=target_meta.stochastic_sites + (declared_data_factor,),
        ),
        dimensions=model_dimensions(Target),
    )
    assert model_meta(with_prior(valid, prior=Prior)).stochastic_sites[-1] == declared_data_factor


def test_final_closure_rejects_vector_bounds_owner_collision_from_source_site_label() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed = Data.vector(n_obs)
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        missing_upper = Data.vector(n_mis)
        latent = PartiallyObserved.vector(
            Normal(theta, 1.0),
            length=n,
            observed=observed,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
            missing_upper=missing_upper,
        )

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    source_meta = model_meta(Prior)
    source = bindable_from_meta(
        replace(
            source_meta,
            stochastic_sites=(replace(source_meta.stochastic_sites[0], name="latent"),),
        ),
        dimensions=model_dimensions(Prior),
    )

    with pytest.raises(
        ValueError,
        match="composed model free value 'latent'.*exactly one canonical owner",
    ):
        with_prior(Target, prior=source)


def test_final_closure_rejects_retained_factor_that_becomes_second_param_owner() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(2.0, 0.5))

    target_meta = model_meta(Target)
    factor = ResolvedStochasticSite(
        name="penalty",
        distribution=Normal(ConstNode(2.0), ConstNode(0.5)),
        value=ParamRef("theta"),
    )
    target = bindable_from_meta(
        replace(target_meta, stochastic_sites=target_meta.stochastic_sites + (factor,)),
        dimensions=model_dimensions(Target),
    )

    with pytest.raises(
        ValueError,
        match="composed model parameter 'theta'.*exactly one structural owner",
    ):
        with_prior(target, prior=Prior)


def test_final_closure_rejects_source_vector_bounds_param_site_label_collision() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    target_meta = model_meta(Target)
    factor = ResolvedStochasticSite(
        name="hyper",
        distribution=Normal(ConstNode(0.0), ConstNode(2.0)),
        value=ParamRef("theta"),
    )
    target = bindable_from_meta(
        replace(target_meta, stochastic_sites=target_meta.stochastic_sites + (factor,)),
        dimensions=model_dimensions(Target),
    )

    @model
    class PriorDeclaration:
        lower = Data.vector(2)
        theta = Param(Normal(2.0, 0.5))
        hyper = Param(Normal(0.0, 1.0), size=2)

    source_meta = model_meta(PriorDeclaration)
    constraint = VectorBounds(lower=DataRef("lower"), upper=None)
    params = dict(source_meta.params)
    params["hyper"] = replace(params["hyper"], constraint=constraint)
    free_values = dict(source_meta.free_values)
    free_values["hyper"] = replace(free_values["hyper"], constraint=constraint)
    source = bindable_from_meta(
        replace(source_meta, params=params, free_values=free_values),
        dimensions=model_dimensions(PriorDeclaration),
    )

    with pytest.raises(
        ValueError,
        match="composed model VectorBounds parameter 'hyper'.*unique same-name owner",
    ):
        with_prior(target, prior=source)


@pytest.mark.parametrize("invalid_size", [ParamRef("theta"), True, -1, 1.5])
def test_composition_rejects_invalid_resolved_size_node_kinds(invalid_size: object) -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5))

    def with_size(model_cls: type[object]) -> type[object]:
        meta = model_meta(model_cls)
        return bindable_from_meta(
            replace(
                meta,
                params={"theta": replace(meta.params["theta"], size=invalid_size)},
                free_values={
                    "theta": replace(meta.free_values["theta"], size=invalid_size),
                },
            )
        )

    with pytest.raises(
        (TypeError, ValueError),
        match="size must be DataRef, a non-negative integer, or None",
    ):
        with_prior(with_size(TargetDeclaration), prior=with_size(PriorDeclaration))


@pytest.mark.parametrize(
    "malformed",
    [
        cast(ExprNode, 7),
        BinOp("unsupported", ParamRef("theta"), ConstNode(1.0)),
        UnaryOp("unsupported", ParamRef("theta")),
        IndexOp(ParamRef("theta"), cast(IndexSpec, 7)),
    ],
)
def test_composition_rejects_malformed_named_expression_ir(malformed: ExprNode) -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    meta = model_meta(TargetDeclaration)
    target = bindable_from_meta(
        replace(meta, expressions={"malformed": malformed}),
        dimensions=model_dimensions(TargetDeclaration),
    )

    with pytest.raises(
        (TypeError, ValueError),
        match="expression 'malformed'.*(expression IR|binary operator|unary function|index spec)",
    ):
        with_prior(target, prior=Prior)


def test_composition_rejects_non_expression_stochastic_site_value() -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    meta = model_meta(TargetDeclaration)
    factor = ResolvedStochasticSite(
        name="malformed",
        distribution=Normal(0.0, 1.0),
        value=cast(ExprNode, 7),
    )
    target = bindable_from_meta(
        replace(meta, stochastic_sites=meta.stochastic_sites + (factor,)),
        dimensions=model_dimensions(TargetDeclaration),
    )

    with pytest.raises(TypeError, match="stochastic site 'malformed' value.*expression IR"):
        with_prior(target, prior=Prior)


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (ResolvedDataRankSchema(-1), "rank must be a non-negative integer"),
        (ResolvedDataRankSchema(True), "rank must be a non-negative integer"),
        (ResolvedDataShapeSchema((cast(int, 1.5),)), "shape dimensions.*non-negative integers"),
        (ResolvedDataShapeSchema((-1,)), "shape dimensions.*non-negative integers"),
        (ResolvedDataShapeSchema((True,)), "shape dimensions.*non-negative integers"),
    ],
)
def test_composition_rejects_malformed_resolved_data_schemas(
    schema: ResolvedDataRankSchema | ResolvedDataShapeSchema,
    message: str,
) -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        context = Data.vector()
        theta = Param(Normal(1.0, 0.5))

    meta = model_meta(PriorDeclaration)
    data = dict(meta.data)
    data["context"] = ResolvedData(schema)
    source = bindable_from_meta(
        replace(meta, data=data),
        dimensions=model_dimensions(PriorDeclaration),
    )

    with pytest.raises((TypeError, ValueError), match=message):
        with_prior(Target, prior=source)


def test_matching_coordinates_distinguish_json_scalar_types() -> None:
    target_axis = Dim("axis", coords=(True,))
    source_axis = Dim("axis", coords=(1,))

    @model
    class Target:
        theta = Param(Normal(0.0, 1.0), size=1, dims=(target_axis,))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5), size=1, dims=(source_axis,))

    with pytest.raises(ValueError, match="parameter 'theta'.*dimensions"):
        with_prior(Target, prior=Prior)


def test_retained_coordinate_collisions_distinguish_json_scalar_types() -> None:
    target_axis = Dim("axis", coords=(True,))
    source_axis = Dim("axis", coords=(1,))

    @model
    class Target:
        target_data = Data.vector(1, dims=(target_axis,))
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        source_data = Data.vector(1, dims=(source_axis,))
        theta = Param(Normal(1.0, 0.5))

    with pytest.raises(ValueError, match="dimension 'axis'.*coordinate"):
        with_prior(Target, prior=Prior)


def test_composition_rejects_mutable_sidecar_containers() -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0), size=2)

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5), size=2)

    names = cast(tuple[str, ...], ["axis"])
    coordinates = cast(tuple[CoordValue, ...], ["a", "b"])
    malformed = ResolvedModelDimensions(
        variables={"theta": ResolvedVariableDims(names)},
        coords={"axis": coordinates},
    )
    target = bindable_from_meta(model_meta(TargetDeclaration), dimensions=malformed)
    source = bindable_from_meta(model_meta(PriorDeclaration), dimensions=malformed)

    with pytest.raises(TypeError, match="dimension names and coordinates must be tuples"):
        with_prior(target, prior=source)


def test_composition_rejects_duck_typed_dimension_variable_records() -> None:
    @dataclass(frozen=True)
    class FakeDims:
        names: tuple[str, ...]

    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    malformed = ResolvedModelDimensions(
        variables=cast(dict[str, ResolvedVariableDims], {"theta": FakeDims(())}),
        coords={},
    )
    target = bindable_from_meta(model_meta(TargetDeclaration), dimensions=malformed)

    with pytest.raises(TypeError, match="dimension variables must contain ResolvedVariableDims"):
        with_prior(target, prior=Prior)


def test_composition_rejects_empty_source_target_and_factor_only_models() -> None:
    empty_meta = ModelMeta(params={}, data={}, observed_nodes=(), expressions={})
    empty = bindable_from_meta(empty_meta)

    @model
    class ObservedTarget:
        y = Observed(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(0.0, 1.0))

    factor_only = bindable_from_meta(
        replace(
            empty_meta,
            stochastic_sites=(
                ResolvedStochasticSite(
                    name="factor",
                    distribution=Normal(0.0, 1.0),
                    value=ConstNode(0.0),
                ),
            ),
        )
    )

    for target, source in (
        (ObservedTarget, empty),
        (empty, Prior),
        (empty, empty),
        (factor_only, Prior),
    ):
        with pytest.raises(ValueError, match="at least one stochastic declaration"):
            with_prior(target, prior=source)


def test_composition_rejects_resolved_role_type_mismatches() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5))

    source_meta = model_meta(PriorDeclaration)
    wrong_free_value = bindable_from_meta(
        replace(source_meta, free_values={"theta": source_meta.params["theta"]}),
        dimensions=model_dimensions(PriorDeclaration),
    )
    with pytest.raises(TypeError, match="free value 'theta'.*ResolvedFreeValue"):
        with_prior(Target, prior=wrong_free_value)

    wrong_distribution = bindable_from_meta(
        replace(
            source_meta,
            params={
                "theta": replace(source_meta.params["theta"], distribution=Positive()),
            },
            stochastic_sites=(replace(source_meta.stochastic_sites[0], distribution=Positive()),),
        ),
        dimensions=model_dimensions(PriorDeclaration),
    )
    with pytest.raises(TypeError, match="parameter 'theta' distribution.*registered distribution"):
        with_prior(Target, prior=wrong_distribution)


def test_composition_rejects_malformed_nested_constraint_and_distribution_roles() -> None:
    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5))

    source_meta = model_meta(PriorDeclaration)
    malformed_bounds = VectorBounds(lower=cast(DataRef, 1))
    bounds_source = bindable_from_meta(
        replace(
            source_meta,
            params={
                "theta": replace(source_meta.params["theta"], constraint=malformed_bounds),
            },
            free_values={
                "theta": replace(source_meta.free_values["theta"], constraint=malformed_bounds),
            },
        ),
        dimensions=model_dimensions(PriorDeclaration),
    )
    with pytest.raises(TypeError, match=r"VectorBounds.*lower.*DataRef"):
        with_prior(Target, prior=bounds_source)

    malformed_distribution = Truncated(cast(Distribution, Positive()), lower=0.0)
    distribution_source = bindable_from_meta(
        replace(
            source_meta,
            params={
                "theta": replace(
                    source_meta.params["theta"],
                    distribution=malformed_distribution,
                ),
            },
            stochastic_sites=(
                replace(
                    source_meta.stochastic_sites[0],
                    distribution=malformed_distribution,
                ),
            ),
        ),
        dimensions=model_dimensions(PriorDeclaration),
    )
    with pytest.raises(TypeError, match=r"Truncated.*base.*registered distribution"):
        with_prior(Target, prior=distribution_source)


def test_composition_rejects_mutable_observed_and_site_sequences() -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))
        y = Observed(Normal(theta, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    meta = model_meta(TargetDeclaration)
    for malformed in (
        replace(meta, observed_nodes=cast(tuple, list(meta.observed_nodes))),
        replace(meta, stochastic_sites=cast(tuple, list(meta.stochastic_sites))),
    ):
        target = bindable_from_meta(malformed, dimensions=model_dimensions(TargetDeclaration))
        with pytest.raises(TypeError, match="observed nodes and stochastic sites must be tuples"):
            with_prior(target, prior=Prior)


@pytest.mark.parametrize("role", ["source", "target"])
@pytest.mark.parametrize("field", ["free_values", "stochastic_sites"])
def test_composition_rejects_empty_lists_before_legacy_fallback(role: str, field: str) -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5))

    declaration = PriorDeclaration if role == "source" else TargetDeclaration
    meta = model_meta(declaration)
    if field == "free_values":
        malformed_meta = replace(meta, free_values=cast(dict, []))
        match = "free_values must be a dict"
    else:
        malformed_meta = replace(meta, stochastic_sites=cast(tuple, []))
        match = "observed nodes and stochastic sites must be tuples"
    malformed = bindable_from_meta(malformed_meta, dimensions=model_dimensions(declaration))

    target = malformed if role == "target" else TargetDeclaration
    source = malformed if role == "source" else PriorDeclaration
    with pytest.raises(TypeError, match=match):
        with_prior(target, prior=source)


def test_composition_rejects_unexecutable_retained_factor_roles() -> None:
    @model
    class TargetDeclaration:
        x = Data.scalar()
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    meta = model_meta(TargetDeclaration)
    discrete_factor = ResolvedStochasticSite(
        "discrete_factor",
        Bernoulli(ConstNode(0.5)),
        ParamRef("theta"),
    )
    malformed = bindable_from_meta(
        replace(meta, stochastic_sites=meta.stochastic_sites + (discrete_factor,)),
        dimensions=model_dimensions(TargetDeclaration),
    )
    with pytest.raises(TypeError, match="Discrete stochastic site.*data or constants"):
        with_prior(malformed, prior=Prior)

    unsupported_factor = ResolvedStochasticSite(
        "truncated_factor",
        Truncated(
            StudentT(ConstNode(3.0), ConstNode(0.0), ConstNode(1.0)),
            lower=ConstNode(0.0),
        ),
        ParamRef("theta"),
    )
    malformed = bindable_from_meta(
        replace(meta, stochastic_sites=meta.stochastic_sites + (unsupported_factor,)),
        dimensions=model_dimensions(TargetDeclaration),
    )
    with pytest.raises(TypeError, match=r"Truncated.*StudentT.*CDF/ICDF"):
        with_prior(malformed, prior=Prior)

    valid_data_factor = replace(discrete_factor, value=DataRef("x"))
    valid = bindable_from_meta(
        replace(meta, stochastic_sites=meta.stochastic_sites + (valid_data_factor,)),
        dimensions=model_dimensions(TargetDeclaration),
    )
    assert model_meta(with_prior(valid, prior=Prior)).stochastic_sites[-1] == valid_data_factor


@pytest.mark.parametrize("role", ["source", "target"])
def test_composition_rejects_discrete_param_metadata(role: str) -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5))

    declaration = PriorDeclaration if role == "source" else TargetDeclaration
    meta = model_meta(declaration)
    discrete = Bernoulli(ConstNode(0.5))
    malformed = bindable_from_meta(
        replace(
            meta,
            params={"theta": replace(meta.params["theta"], distribution=discrete)},
            stochastic_sites=(replace(meta.stochastic_sites[0], distribution=discrete),),
        ),
        dimensions=model_dimensions(declaration),
    )

    target = malformed if role == "target" else TargetDeclaration
    source = malformed if role == "source" else PriorDeclaration
    with pytest.raises(TypeError, match="Discrete distributions cannot be used as Param"):
        with_prior(target, prior=source)


def test_composition_rejects_discrete_partially_observed_metadata() -> None:
    @model
    class TargetDeclaration:
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed = Data.vector(n_obs)
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        theta = Param(Normal(0.0, 1.0))
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

    meta = model_meta(TargetDeclaration)
    sites = tuple(
        replace(site, distribution=Bernoulli(ConstNode(0.5))) if site.name == "latent" else site
        for site in meta.stochastic_sites
    )
    malformed = bindable_from_meta(
        replace(meta, stochastic_sites=sites),
        dimensions=model_dimensions(TargetDeclaration),
    )

    with pytest.raises(
        TypeError,
        match="Discrete distributions cannot be partially observed NUTS values",
    ):
        with_prior(malformed, prior=Prior)


def test_composition_rejects_malformed_vector_scatter_static_roles() -> None:
    @model
    class TargetDeclaration:
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed = Data.vector(n_obs)
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        missing_upper = Data.vector(n_mis)
        theta = Param(Normal(0.0, 1.0))
        latent = PartiallyObserved.vector(
            Normal(theta, 1.0),
            length=n,
            observed=observed,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
            missing_upper=missing_upper,
        )

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    meta = model_meta(TargetDeclaration)
    latent_index = next(
        index for index, site in enumerate(meta.stochastic_sites) if site.name == "latent"
    )
    latent_site = meta.stochastic_sites[latent_index]
    assert isinstance(latent_site.value, VectorScatterOp)

    malformed_scatters = (
        replace(latent_site.value, length=ParamRef("theta")),
        replace(latent_site.value, observed_idx=ParamRef("theta")),
        replace(latent_site.value, missing_idx=ParamRef("theta")),
    )
    for scatter in malformed_scatters:
        sites = list(meta.stochastic_sites)
        sites[latent_index] = replace(latent_site, value=scatter)
        malformed = bindable_from_meta(
            replace(meta, stochastic_sites=tuple(sites)),
            dimensions=model_dimensions(TargetDeclaration),
        )
        with pytest.raises(TypeError, match=r"VectorScatter.*(length|idx).*(data|DataRef)"):
            with_prior(malformed, prior=Prior)

    mismatched_observed = replace(
        latent_site.value,
        observed_values=DataRef("missing_idx"),
    )
    sites = list(meta.stochastic_sites)
    sites[latent_index] = replace(latent_site, value=mismatched_observed)
    malformed = bindable_from_meta(
        replace(meta, stochastic_sites=tuple(sites)),
        dimensions=model_dimensions(TargetDeclaration),
    )
    with pytest.raises(
        ValueError,
        match="VectorScatter observed_values and observed_idx must use the same dimension",
    ):
        with_prior(malformed, prior=Prior)

    mismatched_missing = replace(
        latent_site.value,
        missing_idx=DataRef("observed_idx"),
    )
    same_label_nonowner = ResolvedStochasticSite(
        name="latent",
        distribution=Normal(ConstNode(0.0), ConstNode(2.0)),
        value=ParamRef("theta"),
    )
    sites = list(meta.stochastic_sites)
    sites.insert(latent_index, same_label_nonowner)
    sites[latent_index + 1] = replace(latent_site, value=mismatched_missing)
    malformed = bindable_from_meta(
        replace(
            meta,
            free_values={
                **meta.free_values,
                "latent": replace(meta.free_values["latent"], constraint=None),
            },
            stochastic_sites=tuple(sites),
        ),
        dimensions=model_dimensions(TargetDeclaration),
    )
    with pytest.raises(
        ValueError,
        match="VectorScatter missing_idx must match the free-value size dimension",
    ):
        with_prior(malformed, prior=Prior)

    scalar_bounds = VectorBounds(lower=DataRef("n"))
    malformed = bindable_from_meta(
        replace(
            meta,
            free_values={
                **meta.free_values,
                "latent": replace(meta.free_values["latent"], constraint=scalar_bounds),
            },
        ),
        dimensions=model_dimensions(TargetDeclaration),
    )
    with pytest.raises(TypeError, match=r"VectorBounds.*'n'.*exact vector"):
        with_prior(malformed, prior=Prior)


def test_composition_rejects_invalid_builtin_constructor_states() -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        lower = Data.vector(2)
        theta = Param(Normal(1.0, 0.5))

    malformed_bounds = VectorBounds(lower=DataRef("lower"))
    object.__setattr__(malformed_bounds, "lower", None)
    target_meta = model_meta(TargetDeclaration)
    source_meta = model_meta(PriorDeclaration)

    def with_constraint(meta: ModelMeta) -> ModelMeta:
        return replace(
            meta,
            params={"theta": replace(meta.params["theta"], constraint=malformed_bounds)},
            free_values={
                "theta": replace(meta.free_values["theta"], constraint=malformed_bounds),
            },
        )

    target = bindable_from_meta(with_constraint(target_meta))
    source = bindable_from_meta(with_constraint(source_meta))
    with pytest.raises(TypeError, match="VectorBounds requires at least one"):
        with_prior(target, prior=source)

    malformed_truncated = Truncated(Normal(0.0, 1.0), lower=0.0)
    object.__setattr__(malformed_truncated, "lower", None)
    malformed_source = bindable_from_meta(
        replace(
            source_meta,
            params={
                "theta": replace(
                    source_meta.params["theta"],
                    distribution=malformed_truncated,
                ),
            },
            stochastic_sites=(
                replace(
                    source_meta.stochastic_sites[0],
                    distribution=malformed_truncated,
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="Truncated distributions require at least one bound"):
        with_prior(TargetDeclaration, prior=malformed_source)


def test_composition_validates_discarded_target_prior_codec() -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    meta = model_meta(TargetDeclaration)
    malformed_distribution = MappedTestDistribution({"nested": (1, 2)})
    target = bindable_from_meta(
        replace(
            meta,
            params={
                "theta": replace(
                    meta.params["theta"],
                    distribution=malformed_distribution,
                )
            },
            stochastic_sites=(
                replace(
                    meta.stochastic_sites[0],
                    distribution=malformed_distribution,
                ),
            ),
        ),
        dimensions=model_dimensions(TargetDeclaration),
    )

    with pytest.raises(UnserializableValue, match="bare tuple"):
        with_prior(target, prior=Prior)


def test_composition_round_trip_detaches_nested_extension_maps() -> None:
    distribution = MappedTestDistribution({"location": 1.0})

    @model
    class Target:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        theta = Param(distribution)

    composed = with_prior(Target, prior=PriorDeclaration)
    source_distribution = cast(
        MappedTestDistribution,
        model_meta(PriorDeclaration).params["theta"].distribution,
    )
    composed_distribution = cast(
        MappedTestDistribution,
        model_meta(composed).params["theta"].distribution,
    )

    assert composed_distribution is not source_distribution
    source_distribution.parameters["location"] = 9.0
    assert composed_distribution.parameters["location"] == 1.0


@pytest.mark.parametrize("index", [1.5, "parameter"])
def test_composition_rejects_non_integer_or_parameter_dependent_indexes(index: object) -> None:
    @model
    class Prior:
        theta = Param(Normal(1.0, 0.5))

    if index == "parameter":

        @model
        class Target:
            x = Data.vector()
            theta = Param(Normal(0.0, 1.0))
            selected = x[theta]
            y = Observed(Normal(selected, 1.0))
    else:

        @model
        class Target:
            x = Data.vector()
            theta = Param(Normal(0.0, 1.0))
            selected = x[index]
            y = Observed(Normal(selected + theta, 1.0))

    with pytest.raises(TypeError, match="index expressions must use integer data or constants"):
        with_prior(Target, prior=Prior)


def test_final_metadata_contains_only_resolved_public_ir_nodes() -> None:
    @model
    class Target:
        x = Data.vector()
        theta = Param(Normal(0.0, 1.0))
        y = Observed(Normal(theta * x, 1.0))

    @model
    class Prior:
        hyper = Param(Normal(0.0, 1.0))
        theta = Param(Normal(hyper, 0.5))

    composed = with_prior(Target, prior=Prior)
    document = meta_to_dict(model_meta(composed))

    assert document["bayeswire_ir"] == 1
    assert "_ParameterInput" not in repr(document)
    assert "_ParameterExport" not in repr(document)
    assert "_Wire" not in repr(document)


def test_dimensions_merge_source_then_retained_target_variables() -> None:
    parameter_axis = Dim("parameter_axis", coords=("a", "b"))
    source_axis = Dim("source_axis", coords=("s1", "s2"))
    target_axis = Dim("target_axis", coords=("t1", "t2"))

    @model
    class Target:
        target_data = Data.vector(2, dims=(target_axis,))
        theta = Param(Normal(0.0, 1.0), size=2, dims=(parameter_axis,))
        y = Observed(Normal(theta + target_data, 1.0), dims=(target_axis,))

    @model
    class Prior:
        source_data = Data.vector(2, dims=(source_axis,))
        theta = Param(Normal(1.0, 0.5), size=2, dims=(parameter_axis,))

    composed = with_prior(Target, prior=Prior)
    dimensions = model_dimensions(composed)

    assert tuple(dimensions.variables) == ("source_data", "theta", "target_data", "y")
    assert tuple(dimensions.coords) == ("source_axis", "parameter_axis", "target_axis")
    assert dimensions.variables["theta"] == ResolvedVariableDims(("parameter_axis",))


def _malformed_dimensions(
    *,
    variable_names: tuple[str, ...],
    coords: dict[str, tuple[CoordValue, ...]],
) -> ResolvedModelDimensions:
    return ResolvedModelDimensions(
        variables={"theta": ResolvedVariableDims(variable_names)},
        coords=coords,
    )


def test_composition_rejects_dimension_rank_incompatible_with_static_param_shape() -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5))

    malformed = _malformed_dimensions(variable_names=("axis",), coords={"axis": ("only",)})
    target = bindable_from_meta(model_meta(TargetDeclaration), dimensions=malformed)
    source = bindable_from_meta(model_meta(PriorDeclaration), dimensions=malformed)

    with pytest.raises(ValueError, match="dimension rank.*theta.*static rank 0"):
        with_prior(target, prior=source)


def test_composition_rejects_unused_dimension_coordinates() -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5))

    malformed = _malformed_dimensions(variable_names=(), coords={"unused": ("x",)})
    target = bindable_from_meta(model_meta(TargetDeclaration), dimensions=malformed)
    source = bindable_from_meta(model_meta(PriorDeclaration), dimensions=malformed)

    with pytest.raises(ValueError, match="unused dimension coordinates.*unused"):
        with_prior(target, prior=source)


@pytest.mark.parametrize("owner_role", ["source", "target"])
def test_vector_bounds_param_requires_its_structural_owner_to_be_same_name(
    owner_role: str,
) -> None:
    @model
    class TargetDeclaration:
        lower = Data.vector(2)
        theta = Param(Normal(0.0, 1.0), size=2)

    @model
    class PriorDeclaration:
        lower = Data.vector(2)
        theta = Param(Normal(1.0, 0.5), size=2)

    def with_vector_bounds(model_cls: type[object], *, rename_owner: bool) -> type[object]:
        meta = model_meta(model_cls)
        constraint = VectorBounds(lower=DataRef("lower"), upper=None)
        params = {"theta": replace(meta.params["theta"], constraint=constraint)}
        free_values = {"theta": replace(meta.free_values["theta"], constraint=constraint)}
        owner = meta.stochastic_sites[0]
        if rename_owner:
            owner = replace(owner, name="renamed_theta_prior")
        return bindable_from_meta(
            replace(meta, params=params, free_values=free_values, stochastic_sites=(owner,)),
            dimensions=model_dimensions(model_cls),
        )

    target = with_vector_bounds(TargetDeclaration, rename_owner=owner_role == "target")
    source = with_vector_bounds(PriorDeclaration, rename_owner=owner_role == "source")

    with pytest.raises(ValueError, match="VectorBounds parameter 'theta'.*same-name owner"):
        with_prior(target, prior=source)


@pytest.mark.parametrize(
    ("variable_names", "coords", "message"),
    [
        (("",), {"": ("a", "b")}, "dimension names must be non-empty"),
        (("axis",), {"axis": (math.nan, math.nan)}, "coordinate floats must be finite"),
        (
            ("axis",),
            {"axis": (cast(CoordValue, object()), cast(CoordValue, object()))},
            "coordinates must be JSON scalar values",
        ),
    ],
)
def test_composition_rejects_malformed_dimension_names_and_coordinates(
    variable_names: tuple[str, ...],
    coords: dict[str, tuple[CoordValue, ...]],
    message: str,
) -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0), size=2)

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5), size=2)

    malformed = _malformed_dimensions(variable_names=variable_names, coords=coords)
    target = bindable_from_meta(model_meta(TargetDeclaration), dimensions=malformed)
    source = bindable_from_meta(model_meta(PriorDeclaration), dimensions=malformed)

    with pytest.raises((TypeError, ValueError), match=message):
        with_prior(target, prior=source)


def test_composition_rejects_empty_dimension_variable_names() -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0))

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5))

    def with_empty_param_name(model_cls: type[object]) -> type[object]:
        meta = model_meta(model_cls)
        renamed = replace(meta.stochastic_sites[0], name="", value=ParamRef(""))
        renamed_meta = replace(
            meta,
            params={"": meta.params["theta"]},
            free_values={"": meta.free_values["theta"]},
            stochastic_sites=(renamed,),
        )
        dimensions = ResolvedModelDimensions(
            variables={"": ResolvedVariableDims(())},
            coords={},
        )
        return bindable_from_meta(renamed_meta, dimensions=dimensions)

    with pytest.raises(ValueError, match="dimension variable names must be non-empty"):
        with_prior(
            with_empty_param_name(TargetDeclaration),
            prior=with_empty_param_name(PriorDeclaration),
        )


def test_composition_rejects_coordinate_length_incompatible_with_static_size() -> None:
    @model
    class TargetDeclaration:
        theta = Param(Normal(0.0, 1.0), size=2)

    @model
    class PriorDeclaration:
        theta = Param(Normal(1.0, 0.5), size=2)

    malformed = _malformed_dimensions(
        variable_names=("axis",),
        coords={"axis": ("a", "b", "c")},
    )
    target = bindable_from_meta(model_meta(TargetDeclaration), dimensions=malformed)
    source = bindable_from_meta(model_meta(PriorDeclaration), dimensions=malformed)

    with pytest.raises(ValueError, match="coordinate length 3.*theta.*static axis size 2"):
        with_prior(target, prior=source)


def _all_nested_values(value: object) -> tuple[object, ...]:
    if isinstance(value, dict):
        return tuple(
            nested for item in value.values() for nested in (item, *_all_nested_values(item))
        )
    if isinstance(value, tuple):
        return tuple(nested for item in value for nested in (item, *_all_nested_values(item)))
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            nested
            for value_field in fields(value)
            for item in (getattr(value, value_field.name),)
            for nested in (item, *_all_nested_values(item))
        )
    return ()


def test_valid_composition_has_no_dangling_param_or_data_references() -> None:
    @model
    class Target:
        x = Data.vector()
        theta = Param(Normal(0.0, 1.0))
        y = Observed(Normal(theta * x, 1.0))

    @model
    class Prior:
        hyper = Param(Normal(0.0, 1.0))
        theta = Param(Normal(hyper, 0.5))

    result = model_meta(with_prior(Target, prior=Prior))
    nested = _all_nested_values(result)
    free_names = set(result.free_values)
    data_names = set(result.data) | {node.name for node in result.observed_nodes}

    assert all(value.name in free_names for value in nested if isinstance(value, ParamRef))
    assert all(value.name in data_names for value in nested if isinstance(value, DataRef))
