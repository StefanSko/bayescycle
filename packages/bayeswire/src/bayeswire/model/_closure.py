"""Typed closure and static dimension validation for composed metadata."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass

from bayeswire.constraints import VectorBounds
from bayeswire.model._components import _DimensionSnapshot, _ModelSnapshot
from bayeswire.model._data_schema import (
    DataDimRef,
    ResolvedDataRankSchema,
    ResolvedDataSchema,
    ResolvedDataShapeSchema,
)
from bayeswire.model.expr import DataRef, ParamRef, VectorScatterOp


def _is_scalar_schema(schema: ResolvedDataSchema) -> bool:
    if isinstance(schema, ResolvedDataShapeSchema):
        return schema.dims == ()
    return schema.rank == 0


def _validate_size_reference(
    size: object,
    *,
    scalar_data: set[str],
    label: str,
) -> None:
    if size is None:
        return
    if isinstance(size, DataRef):
        if size.name not in scalar_data:
            raise ValueError(f"{label} references unknown scalar data {size.name!r}")
        return
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        error = f"{label} must be DataRef, a non-negative integer, or None"
        if isinstance(size, int) and not isinstance(size, bool):
            raise ValueError(error)
        raise TypeError(error)


def _validate_value_references(
    value: object,
    *,
    free_values: set[str],
    data: set[str],
    observed_data: set[str],
    label: str,
) -> None:
    """Validate typed references recursively through registered dataclass-shaped values."""
    if isinstance(value, ParamRef):
        if value.name not in free_values:
            raise ValueError(f"{label} references unknown free value {value.name!r}")
        return
    if isinstance(value, DataRef):
        if value.name not in data and value.name not in observed_data:
            raise ValueError(f"{label} references unknown data {value.name!r}")
        return
    if isinstance(value, dict):
        for item in value.values():
            _validate_value_references(
                item,
                free_values=free_values,
                data=data,
                observed_data=observed_data,
                label=label,
            )
        return
    if isinstance(value, tuple):
        for item in value:
            _validate_value_references(
                item,
                free_values=free_values,
                data=data,
                observed_data=observed_data,
                label=label,
            )
        return
    if is_dataclass(value) and not isinstance(value, type):
        for value_field in fields(value):
            _validate_value_references(
                getattr(value, value_field.name),
                free_values=free_values,
                data=data,
                observed_data=observed_data,
                label=label,
            )


def _is_canonical_free_value_owner(site: object, name: str) -> bool:
    """Return whether one site has a same-name direct or scatter owner form."""
    from bayeswire.model.decorator import ResolvedStochasticSite

    if not isinstance(site, ResolvedStochasticSite) or site.name != name:
        return False
    if site.value == ParamRef(name):
        return True
    return isinstance(site.value, VectorScatterOp) and site.value.missing_values == ParamRef(name)


def _validate_param_owners(model: _ModelSnapshot, *, role: str) -> None:
    """Revalidate structural and VectorBounds Param ownership after every merge."""
    free_values = dict(model.free_values)
    for name, param in model.params:
        owners = tuple(
            site
            for site in model.stochastic_sites
            if site.value == ParamRef(name) and site.distribution == param.distribution
        )
        if len(owners) != 1:
            raise ValueError(
                f"{role} parameter {name!r} must have exactly one structural owner site, "
                f"found {len(owners)}"
            )
        free_value = free_values.get(name)
        if free_value is None or not isinstance(free_value.constraint, VectorBounds):
            continue
        named_sites = tuple(site for site in model.stochastic_sites if site.name == name)
        if len(named_sites) != 1 or named_sites[0] is not owners[0]:
            raise ValueError(
                f"{role} VectorBounds parameter {name!r} must use its structural Param "
                "site as the unique same-name owner"
            )


def _validate_non_param_free_value_owners(
    model: _ModelSnapshot,
    *,
    role: str,
) -> None:
    params = {name for name, _value in model.params}
    for name, free_value in model.free_values:
        if name in params:
            continue
        candidates = tuple(
            site
            for site in model.stochastic_sites
            if (
                site.name == name
                if isinstance(free_value.constraint, VectorBounds)
                else _is_canonical_free_value_owner(site, name)
            )
        )
        if len(candidates) != 1 or not _is_canonical_free_value_owner(candidates[0], name):
            raise ValueError(
                f"{role} free value {name!r} must have exactly one canonical owner site, "
                f"found {len(candidates)}"
            )


def _validate_model_closure(
    model: _ModelSnapshot,
    dimensions: _DimensionSnapshot | None,
    *,
    role: str,
) -> None:
    """Require one input or final snapshot to be independently closed."""
    free_values = {name for name, _value in model.free_values}
    data = {name for name, _value in model.data}
    scalar_data = {name for name, resolved in model.data if _is_scalar_schema(resolved.schema)}
    _validate_param_owners(model, role=role)
    _validate_non_param_free_value_owners(model, role=role)

    for name, resolved in model.data:
        if isinstance(resolved.schema, ResolvedDataShapeSchema):
            for dimension in resolved.schema.dims:
                if isinstance(dimension, DataDimRef) and dimension.name not in scalar_data:
                    raise ValueError(
                        f"{role} data {name!r} references unknown scalar data {dimension.name!r}"
                    )

    for name, param in model.params:
        _validate_size_reference(
            param.size,
            scalar_data=scalar_data,
            label=f"{role} parameter {name!r} size",
        )
        _validate_value_references(
            param.distribution,
            free_values=free_values,
            data=data,
            observed_data=set(),
            label=f"{role} parameter {name!r} distribution",
        )

    for name, free_value in model.free_values:
        _validate_size_reference(
            free_value.size,
            scalar_data=scalar_data,
            label=f"{role} free value {name!r} size",
        )
        _validate_value_references(
            free_value.constraint,
            free_values=free_values,
            data=data,
            observed_data=set(),
            label=f"{role} free value {name!r} constraint",
        )

    for name, expression in model.expressions:
        _validate_value_references(
            expression,
            free_values=free_values,
            data=data,
            observed_data=set(),
            label=f"{role} expression {name!r}",
        )

    for node in model.observed_nodes:
        _validate_value_references(
            node.distribution,
            free_values=free_values,
            data=data,
            observed_data=set(),
            label=f"{role} Observed {node.name!r} distribution",
        )

    for site in model.stochastic_sites:
        _validate_value_references(
            site.distribution,
            free_values=free_values,
            data=data,
            observed_data=set(),
            label=f"{role} stochastic site {site.name!r} distribution",
        )
        owned_observed = {
            node.name
            for node in model.observed_nodes
            if site.value == DataRef(node.name) and site.distribution == node.distribution
        }
        _validate_value_references(
            site.value,
            free_values=free_values,
            data=data,
            observed_data=owned_observed,
            label=f"{role} stochastic site {site.name!r} value",
        )

    _validate_static_dimensions(model, dimensions, role=role)


def _static_shape(
    name: str,
    model: _ModelSnapshot,
) -> tuple[int | None, ...] | None:
    params = dict(model.params)
    param = params.get(name)
    if param is not None:
        if param.size is None:
            return ()
        if isinstance(param.size, int):
            return (param.size,)
        return (None,)

    data = dict(model.data).get(name)
    if data is None:
        return None
    if isinstance(data.schema, ResolvedDataRankSchema):
        return tuple(None for _ in range(data.schema.rank))
    return tuple(
        dimension if isinstance(dimension, int) else None for dimension in data.schema.dims
    )


def _validate_static_dimensions(
    model: _ModelSnapshot,
    dimensions: _DimensionSnapshot | None,
    *,
    role: str,
) -> None:
    """Validate sidecar syntax, roles, and all shapes known without binding."""
    if dimensions is None:
        return

    for variable, names in dimensions.variables:
        if not isinstance(variable, str) or variable == "":
            raise ValueError(f"{role} dimension variable names must be non-empty strings")
        for name in names:
            if not isinstance(name, str) or name == "":
                raise ValueError(f"{role} dimension names must be non-empty strings")
    for name, values in dimensions.coords:
        if not isinstance(name, str) or name == "":
            raise ValueError(f"{role} dimension coordinate names must be non-empty strings")
        for value in values:
            if value is None or isinstance(value, bool | int | str):
                continue
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(f"{role} dimension coordinate floats must be finite")
                continue
            raise TypeError(
                f"{role} dimension coordinates must be JSON scalar values: "
                "str, int, float, bool, or None"
            )

    params = {name for name, _value in model.params}
    data = {name for name, _value in model.data}
    observed = {node.name for node in model.observed_nodes}
    allowed = params | data | observed
    variable_map = dict(dimensions.variables)
    unknown_variables = tuple(name for name in variable_map if name not in allowed)
    if unknown_variables:
        raise ValueError(
            f"{role} dimensions reference unsupported variables: {list(unknown_variables)}"
        )

    used_dimensions = {dimension for names in variable_map.values() for dimension in names}
    coordinate_map = dict(dimensions.coords)
    unused_coords = tuple(name for name in coordinate_map if name not in used_dimensions)
    if unused_coords:
        raise ValueError(f"{role} has unused dimension coordinates: {list(unused_coords)}")

    for variable, names in dimensions.variables:
        static_shape = _static_shape(variable, model)
        if static_shape is None:
            continue
        if len(names) != len(static_shape):
            raise ValueError(
                f"{role} dimension rank for {variable!r} is {len(names)}, "
                f"but its static rank {len(static_shape)} was required"
            )
        for dimension, static_axis_size in zip(names, static_shape, strict=True):
            coords = coordinate_map.get(dimension)
            if coords is None or static_axis_size is None:
                continue
            if len(coords) != static_axis_size:
                raise ValueError(
                    f"{role} coordinate length {len(coords)} for {variable!r} dimension "
                    f"{dimension!r} does not match static axis size {static_axis_size}"
                )
