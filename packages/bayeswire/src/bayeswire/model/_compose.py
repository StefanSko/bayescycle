"""Wire, compose, and close private prior/outcome kernels."""

from __future__ import annotations

from bayeswire.model._components import (
    _ClosedComposition,
    _ComposedKernel,
    _OutcomeKernel,
    _ParameterExport,
    _ParameterInput,
    _ParameterKernel,
    _variable_dimensions,
    _Wire,
    _Wiring,
)
from bayeswire.model.decorator import ModelMeta
from bayeswire.model.dimensions import CoordValue, ResolvedModelDimensions, ResolvedVariableDims


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


def _validate_interface(
    target: _ParameterInput,
    source: _ParameterExport,
) -> None:
    """Require exact v0 static compatibility except for distributions."""
    if source.interface != target.interface:
        raise ValueError(
            f"prior parameter {source.interface.name!r} must match the target constraint, "
            "size, data-dependent size name, and dimensions"
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
        if existing != value:
            raise ValueError(f"shared data {name!r} must have identical resolved schemas")
        if _variable_dimensions(source.dimensions, name) != _variable_dimensions(
            outcomes.dimensions,
            name,
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
        stochastic_sites=tuple(export.site for export in source.exports) + outcomes.retained_sites,
    )
    dimensions = _merge_dimensions(source, outcomes)
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
            if existing is not None and existing != values:
                raise ValueError(f"dimension {name!r} has conflicting coordinate values")
            coords[name] = values

    return ResolvedModelDimensions(variables=variables, coords=coords)
