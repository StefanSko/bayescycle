"""Ordering, closure, dimensions, determinism, and input immutability."""

from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from dataclasses import fields, is_dataclass, replace

import pytest

from bayeswire import Data, Dim, Observed, Param, model, model_dimensions, with_prior
from bayeswire.distributions import Normal
from bayeswire.ir import bindable_from_meta, meta_to_dict
from bayeswire.model import model_meta
from bayeswire.model._data_schema import DataDimRef, ResolvedDataShapeSchema
from bayeswire.model.decorator import ResolvedData
from bayeswire.model.dimensions import ResolvedModelDimensions, ResolvedVariableDims
from bayeswire.model.expr import DataRef, ParamRef


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
    coords: dict[str, tuple[str, ...]],
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
