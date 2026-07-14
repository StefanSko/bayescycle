"""Wire, compose, and close private prior/outcome kernels."""

from __future__ import annotations

from bayeswire.model._closure import _validate_model_closure
from bayeswire.model._components import (
    _ClosedComposition,
    _ComposedKernel,
    _OutcomeKernel,
    _ParameterExport,
    _ParameterInput,
    _ParameterKernel,
    _snapshot_dimensions,
    _snapshot_model,
    _variable_dimensions,
    _VariableDimensions,
    _Wire,
    _Wiring,
)
from bayeswire.model._structural import _structurally_equal
from bayeswire.model.decorator import ModelMeta
from bayeswire.model.dimensions import CoordValue, ResolvedModelDimensions, ResolvedVariableDims
from bayeswire.model.expr import DataRef


def _build_same_name_wiring(
    source: _ParameterKernel,
    outcomes: _OutcomeKernel,
) -> _Wiring:
    """Build complete target-order same-name wiring."""
    exports = {export.interface.name: export for export in source.exports}
    wires: list[_Wire] = []
    for target_input in outcomes.inputs:
        name = target_input.interface.name
        source_export = exports.get(name)
        if source_export is None:
            raise ValueError(f"prior is missing target parameter {name!r}")
        _validate_interface(target_input, source_export)
        wires.append(_Wire(target=target_input, source=source_export))
    return _Wiring(tuple(wires))


def _coordinate_values_equal(
    left: tuple[CoordValue, ...],
    right: tuple[CoordValue, ...],
) -> bool:
    """Compare JSON scalar coordinates without Python bool/int coercion."""
    return len(left) == len(right) and all(
        type(left_value) is type(right_value) and left_value == right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def _variable_dimensions_equal(
    left: _VariableDimensions,
    right: _VariableDimensions,
) -> bool:
    if left.names != right.names or len(left.coords) != len(right.coords):
        return False
    for (left_name, left_values), (right_name, right_values) in zip(
        left.coords,
        right.coords,
        strict=True,
    ):
        if left_name != right_name or (left_values is None) != (right_values is None):
            return False
        if (
            left_values is not None
            and right_values is not None
            and not _coordinate_values_equal(left_values, right_values)
        ):
            return False
    return True


def _validate_interface(
    target: _ParameterInput,
    source: _ParameterExport,
) -> None:
    """Require exact v0 static compatibility except for distributions."""
    name = source.interface.name
    if not _structurally_equal(source.interface.constraint, target.interface.constraint):
        raise ValueError(f"prior parameter {name!r} must match the target constraint exactly")
    if not _structurally_equal(source.interface.size, target.interface.size):
        if isinstance(source.interface.size, DataRef) or isinstance(target.interface.size, DataRef):
            raise ValueError(
                f"prior parameter {name!r} must use the same data-dependent size name as the target"
            )
        raise ValueError(f"prior parameter {name!r} must match the target size exactly")
    if not _variable_dimensions_equal(
        source.interface.dimensions,
        target.interface.dimensions,
    ):
        raise ValueError(f"prior parameter {name!r} must match the target dimensions exactly")


def _first_overlap(left: tuple[str, ...], right: set[str]) -> str | None:
    return next((name for name in left if name in right), None)


def _reject_cross_role_overlap(
    left: tuple[str, ...],
    right: set[str],
    *,
    source_role: str,
    target_role: str,
) -> None:
    name = _first_overlap(left, right)
    if name is not None:
        raise ValueError(
            f"name {name!r} from {source_role} collides with {target_role}; "
            "rename one declaration before composing"
        )


def _validate_cross_role_collisions(
    source: _ParameterKernel,
    outcomes: _OutcomeKernel,
) -> None:
    """Reject every cross-input value-role overlap except Params and shared data."""
    source_params = tuple(name for name, _value in source.model.params)
    source_data = tuple(name for name, _value in source.model.data)
    source_expressions = tuple(name for name, _value in source.model.expressions)
    target_data = {name for name, _value in outcomes.model.data}
    target_expressions = {name for name, _value in outcomes.model.expressions}
    target_observed = {observed.name for observed in outcomes.model.observed_nodes}
    target_non_param_free = {name for name, _value in outcomes.retained_free_values}

    for target_names, target_role in (
        (target_data, "target data"),
        (target_expressions, "target expression"),
        (target_observed, "target Observed value"),
        (target_non_param_free, "target free value"),
    ):
        _reject_cross_role_overlap(
            source_params,
            target_names,
            source_role="source Param",
            target_role=target_role,
        )
    for target_names, target_role in (
        (target_expressions, "target expression"),
        (target_observed, "target Observed value"),
        (target_non_param_free, "target free value"),
    ):
        _reject_cross_role_overlap(
            source_data,
            target_names,
            source_role="source data",
            target_role=target_role,
        )
    for target_names, target_role in (
        (target_data, "target data"),
        (target_observed, "target Observed value"),
        (target_non_param_free, "target free value"),
    ):
        _reject_cross_role_overlap(
            source_expressions,
            target_names,
            source_role="source expression",
            target_role=target_role,
        )


def _compose_kernels(
    source: _ParameterKernel,
    outcomes: _OutcomeKernel,
    wiring: _Wiring,
) -> _ComposedKernel:
    """Validate that wiring closes every target input."""
    wired_inputs = tuple(wire.target for wire in wiring.wires)
    if wired_inputs != outcomes.inputs:
        raise ValueError("prior composition wiring must close every target parameter exactly once")
    _validate_cross_role_collisions(source, outcomes)
    return _ComposedKernel(source=source, outcomes=outcomes, wiring=wiring)


def _close_composition(composed: _ComposedKernel) -> _ClosedComposition:
    """Materialize ordinary flat metadata from a validated composition."""
    source = composed.source
    outcomes = composed.outcomes

    data = dict(source.model.data)
    for name, value in outcomes.model.data:
        existing = data.get(name)
        if existing is None:
            data[name] = value
            continue
        if not _structurally_equal(existing, value):
            raise ValueError(f"shared data {name!r} must have identical resolved schemas")
        if not _variable_dimensions_equal(
            _variable_dimensions(source.dimensions, name),
            _variable_dimensions(outcomes.dimensions, name),
        ):
            raise ValueError(f"shared data {name!r} must have identical dimensions")

    expressions = dict(source.model.expressions)
    for name, value in outcomes.model.expressions:
        if name in expressions:
            raise ValueError(f"source and target expressions collide at {name!r}")
        expressions[name] = value

    free_values = {export.interface.name: export.free_value for export in source.exports}
    free_values.update(outcomes.retained_free_values)
    meta = ModelMeta(
        params=dict(source.model.params),
        data=data,
        observed_nodes=outcomes.model.observed_nodes,
        expressions=expressions,
        free_values=free_values,
        stochastic_sites=source.model.stochastic_sites + outcomes.retained_sites,
    )
    dimensions = _merge_dimensions(source, outcomes)
    _validate_model_closure(
        _snapshot_model(meta),
        _snapshot_dimensions(dimensions),
        role="composed model",
    )
    return _ClosedComposition(
        meta=meta,
        dimensions=dimensions,
        dependencies=(outcomes.model_cls, source.model_cls),
    )


def _merge_dimensions(
    source: _ParameterKernel,
    outcomes: _OutcomeKernel,
) -> ResolvedModelDimensions | None:
    """Merge source-owned and retained target sidecar entries deterministically."""
    if source.dimensions is None and outcomes.dimensions is None:
        return None

    source_params = {name for name, _param in source.model.params}
    source_data = {name for name, _data in source.model.data}
    target_params = {name for name, _param in outcomes.model.params}
    target_data = {name for name, _data in outcomes.model.data}
    target_observed = {observed.name for observed in outcomes.model.observed_nodes}

    variables: dict[str, ResolvedVariableDims] = {}
    if source.dimensions is not None:
        for name, names in source.dimensions.variables:
            if name in source_params or name in source_data:
                variables[name] = ResolvedVariableDims(names)

    if outcomes.dimensions is not None:
        for name, names in outcomes.dimensions.variables:
            if name in target_params:
                continue
            if name not in target_data and name not in target_observed:
                continue
            existing = variables.get(name)
            candidate = ResolvedVariableDims(names)
            if existing is not None and existing != candidate:
                raise ValueError(f"shared dimension variable {name!r} is incompatible")
            if existing is None:
                variables[name] = candidate

    used_dimensions = {
        dimension
        for variable_dimensions in variables.values()
        for dimension in variable_dimensions.names
    }
    coords: dict[str, tuple[CoordValue, ...]] = {}
    for snapshot in (source.dimensions, outcomes.dimensions):
        if snapshot is None:
            continue
        for name, values in snapshot.coords:
            if name not in used_dimensions:
                continue
            existing = coords.get(name)
            if existing is not None and not _coordinate_values_equal(existing, values):
                raise ValueError(f"dimension {name!r} has conflicting coordinate values")
            coords[name] = values

    return ResolvedModelDimensions(variables=variables, coords=coords)
