"""Typed closure and static dimension validation for composed metadata."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass

from bayeswire._ir_registry import BUILTIN_DISTRIBUTION_CLASSES, NODE_SPECS_BY_CLASS
from bayeswire.constraints import Interval, Ordered, Positive, UnitInterval, VectorBounds
from bayeswire.distributions import Truncated
from bayeswire.model._components import _DimensionSnapshot, _ModelSnapshot
from bayeswire.model._data_schema import (
    DataDimRef,
    ResolvedDataRankSchema,
    ResolvedDataSchema,
    ResolvedDataShapeSchema,
)
from bayeswire.model.decorator import (
    ResolvedData,
    ResolvedFreeValue,
    ResolvedObserved,
    ResolvedParam,
    ResolvedStochasticSite,
)
from bayeswire.model.expr import (
    BinOp,
    ConstNode,
    DataRef,
    FullSlice,
    IndexOp,
    IndexTuple,
    ParamRef,
    ScalarIndex,
    UnaryOp,
    VectorScatterOp,
    is_final_expr_node,
)

_BINARY_OPERATORS = frozenset({"+", "-", "*", "/"})
_UNARY_FUNCTIONS = frozenset({"exp", "neg", "sigmoid"})
_CONSTRAINT_TYPES = (Positive, Interval, UnitInterval, Ordered, VectorBounds)


def _validate_distribution_parameter(value: object, *, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        and not is_final_expr_node(value)
    ):
        raise TypeError(f"{label} must be a finite numeric value or resolved expression")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if is_final_expr_node(value):
        _validate_expression_structure(value, label=label)


def _validate_distribution(value: object, *, label: str) -> None:
    from bayeswire.ir import _is_registered_distribution

    if not _is_registered_distribution(value):
        raise TypeError(f"{label} must be a registered distribution")
    if isinstance(value, Truncated):
        _validate_distribution(value.base, label=f"{label} Truncated base")
        for name, bound in (("lower", value.lower), ("upper", value.upper)):
            if bound is not None:
                _validate_distribution_parameter(
                    bound,
                    label=f"{label} Truncated {name}",
                )
        return
    if type(value) in BUILTIN_DISTRIBUTION_CLASSES:
        for field_name, _kind in NODE_SPECS_BY_CLASS[type(value)].field_kinds:
            _validate_distribution_parameter(
                getattr(value, field_name),
                label=f"{label} field {field_name!r}",
            )


def _validate_constraint(value: object, *, label: str) -> None:
    if value is not None and not isinstance(value, _CONSTRAINT_TYPES):
        raise TypeError(f"{label} must be a supported constraint or None")
    if isinstance(value, VectorBounds):
        for name, bound in (("lower", value.lower), ("upper", value.upper)):
            if bound is not None and not isinstance(bound, DataRef):
                raise TypeError(f"{label} VectorBounds {name} must be DataRef or None")


def _validate_model_role_types(model: _ModelSnapshot, *, role: str) -> None:
    """Require each ModelMeta field to contain its registered semantic role."""
    if not isinstance(model.observed_nodes, tuple) or not isinstance(model.stochastic_sites, tuple):
        raise TypeError(f"{role} observed nodes and stochastic sites must be tuples")
    for name, value in model.params:
        if not isinstance(name, str) or not isinstance(value, ResolvedParam):
            raise TypeError(f"{role} parameter {name!r} must be ResolvedParam")
        _validate_distribution(value.distribution, label=f"{role} parameter {name!r} distribution")
        _validate_constraint(value.constraint, label=f"{role} parameter {name!r} constraint")
    for name, value in model.data:
        if not isinstance(name, str) or not isinstance(value, ResolvedData):
            raise TypeError(f"{role} data {name!r} must be ResolvedData")
    for value in model.observed_nodes:
        if not isinstance(value, ResolvedObserved):
            raise TypeError(f"{role} observed nodes must contain ResolvedObserved values")
        if not isinstance(value.name, str):
            raise TypeError(f"{role} Observed names must be strings")
        _validate_distribution(
            value.distribution,
            label=f"{role} Observed {value.name!r} distribution",
        )
    for name, _value in model.expressions:
        if not isinstance(name, str):
            raise TypeError(f"{role} expression names must be strings")
    for name, value in model.free_values:
        if not isinstance(name, str) or not isinstance(value, ResolvedFreeValue):
            raise TypeError(f"{role} free value {name!r} must be ResolvedFreeValue")
        _validate_constraint(value.constraint, label=f"{role} free value {name!r} constraint")
    for value in model.stochastic_sites:
        if not isinstance(value, ResolvedStochasticSite):
            raise TypeError(f"{role} stochastic sites must contain ResolvedStochasticSite values")
        if not isinstance(value.name, str):
            raise TypeError(f"{role} stochastic site names must be strings")
        _validate_distribution(
            value.distribution,
            label=f"{role} stochastic site {value.name!r} distribution",
        )


def _validate_index_expression(value: object, *, label: str) -> None:
    error = f"{label} index expressions must use integer data or constants"
    if isinstance(value, DataRef):
        return
    if isinstance(value, ConstNode):
        if isinstance(value.value, bool) or not isinstance(value.value, int):
            raise TypeError(error)
        return
    if isinstance(value, BinOp):
        if value.op not in {"+", "-", "*"}:
            raise TypeError(error)
        _validate_index_expression(value.left, label=label)
        _validate_index_expression(value.right, label=label)
        return
    if isinstance(value, UnaryOp):
        if value.function != "neg":
            raise TypeError(error)
        _validate_index_expression(value.operand, label=label)
        return
    if isinstance(value, IndexOp):
        _validate_index_expression(value.base, label=label)
        _validate_index_spec(value.index, label=label)
        return
    raise TypeError(error)


def _validate_index_spec(spec: object, *, label: str) -> None:
    if isinstance(spec, ScalarIndex):
        _validate_expression_structure(spec.expr, label=label)
        _validate_index_expression(spec.expr, label=label)
        return
    if isinstance(spec, FullSlice):
        return
    if isinstance(spec, IndexTuple):
        if not isinstance(spec.items, tuple) or not spec.items:
            raise TypeError(f"{label} index spec must contain a non-empty tuple")
        for item in spec.items:
            if isinstance(item, IndexTuple):
                raise TypeError(f"{label} index spec must not contain nested index tuples")
            _validate_index_spec(item, label=label)
        return
    raise TypeError(f"{label} index spec must be ScalarIndex, FullSlice, or IndexTuple")


def _validate_expression_structure(value: object, *, label: str) -> None:
    """Require one tree to use only executable resolved expression IR."""
    if not is_final_expr_node(value):
        raise TypeError(f"{label} must contain resolved expression IR")
    if isinstance(value, ParamRef | DataRef):
        if not isinstance(value.name, str):
            raise TypeError(f"{label} reference names must be strings")
        return
    if isinstance(value, ConstNode):
        if isinstance(value.value, bool) or not isinstance(value.value, int | float):
            raise TypeError(f"{label} constants must be int or float values")
        if isinstance(value.value, float) and not math.isfinite(value.value):
            raise ValueError(f"{label} constants must be finite")
        return
    if isinstance(value, BinOp):
        if value.op not in _BINARY_OPERATORS:
            raise ValueError(f"{label} uses unknown binary operator {value.op!r}")
        _validate_expression_structure(value.left, label=label)
        _validate_expression_structure(value.right, label=label)
        return
    if isinstance(value, UnaryOp):
        if value.function not in _UNARY_FUNCTIONS:
            raise ValueError(f"{label} uses unknown unary function {value.function!r}")
        _validate_expression_structure(value.operand, label=label)
        return
    if isinstance(value, IndexOp):
        _validate_expression_structure(value.base, label=label)
        _validate_index_spec(value.index, label=label)
        return
    if isinstance(value, VectorScatterOp):
        for item in (
            value.length,
            value.observed_idx,
            value.observed_values,
            value.missing_idx,
            value.missing_values,
        ):
            _validate_expression_structure(item, label=label)


def _validate_data_schema(schema: object, *, label: str) -> None:
    if isinstance(schema, ResolvedDataRankSchema):
        if isinstance(schema.rank, bool) or not isinstance(schema.rank, int) or schema.rank < 0:
            error = f"{label} rank must be a non-negative integer"
            if isinstance(schema.rank, int) and not isinstance(schema.rank, bool):
                raise ValueError(error)
            raise TypeError(error)
        return
    if isinstance(schema, ResolvedDataShapeSchema):
        if not isinstance(schema.dims, tuple):
            raise TypeError(f"{label} shape dimensions must be a tuple")
        for dimension in schema.dims:
            if isinstance(dimension, DataDimRef):
                if not isinstance(dimension.name, str):
                    raise TypeError(f"{label} shape data-reference names must be strings")
                continue
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                error = f"{label} shape dimensions must be non-negative integers or DataDimRef"
                if isinstance(dimension, int) and not isinstance(dimension, bool):
                    raise ValueError(error)
                raise TypeError(error)
        return
    raise TypeError(f"{label} must use a resolved data rank or shape schema")


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
    if is_final_expr_node(value):
        _validate_expression_structure(value, label=label)
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
    _validate_model_role_types(model, role=role)
    param_names = {name for name, _value in model.params}
    free_values = {name for name, _value in model.free_values}
    non_param_free_values = free_values - param_names
    if not param_names and not model.observed_nodes and not non_param_free_values:
        raise ValueError(f"{role} must contain at least one stochastic declaration")

    data = {name for name, _value in model.data}
    for name, resolved in model.data:
        _validate_data_schema(resolved.schema, label=f"{role} data {name!r}")
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
        _validate_expression_structure(expression, label=f"{role} expression {name!r}")
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
        _validate_expression_structure(
            site.value,
            label=f"{role} stochastic site {site.name!r} value",
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
        if not isinstance(names, tuple):
            raise TypeError(f"{role} dimension names and coordinates must be tuples")
        if not isinstance(variable, str) or variable == "":
            raise ValueError(f"{role} dimension variable names must be non-empty strings")
        for name in names:
            if not isinstance(name, str) or name == "":
                raise ValueError(f"{role} dimension names must be non-empty strings")
    for name, values in dimensions.coords:
        if not isinstance(values, tuple):
            raise TypeError(f"{role} dimension names and coordinates must be tuples")
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
