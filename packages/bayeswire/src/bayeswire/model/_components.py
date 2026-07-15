"""Private immutable values for prior-composition phases."""

from __future__ import annotations

from dataclasses import dataclass

from bayeswire.constraints.core import Constraint
from bayeswire.model._data_schema import ResolvedDataSchema
from bayeswire.model.decorator import (
    ModelMeta,
    ResolvedData,
    ResolvedFreeValue,
    ResolvedObserved,
    ResolvedParam,
    ResolvedStochasticSite,
)
from bayeswire.model.dimensions import CoordValue, ResolvedModelDimensions, ResolvedVariableDims
from bayeswire.model.expr import DataRef, ExprNode

type ModelClass = type[object]
type ResolvedSize = DataRef | int | None
type ParamEntries = tuple[tuple[str, ResolvedParam], ...]
type DataEntries = tuple[tuple[str, ResolvedData], ...]
type ExpressionEntries = tuple[tuple[str, ExprNode], ...]
type FreeValueEntries = tuple[tuple[str, ResolvedFreeValue], ...]


@dataclass(frozen=True)
class _VariableDimensions:
    """Optional sidecar metadata for one variable and its named coordinates."""

    names: tuple[str, ...] | None
    coords: tuple[tuple[str, tuple[CoordValue, ...] | None], ...]


@dataclass(frozen=True)
class _ParameterInterface:
    """Static same-name interface of one resolved Param."""

    name: str
    constraint: Constraint | None
    size: ResolvedSize
    dimensions: _VariableDimensions


@dataclass(frozen=True)
class _ParameterExport:
    """One source Param and its declaration-backed prior site."""

    interface: _ParameterInterface
    param: ResolvedParam
    free_value: ResolvedFreeValue
    site: ResolvedStochasticSite


@dataclass(frozen=True)
class _ParameterInput:
    """One target Param whose authored prior will be replaced."""

    interface: _ParameterInterface
    param: ResolvedParam
    free_value: ResolvedFreeValue
    site: ResolvedStochasticSite


@dataclass(frozen=True)
class _ModelSnapshot:
    """An insertion-ordered immutable snapshot of resolved model metadata."""

    params: ParamEntries
    data: DataEntries
    observed_nodes: tuple[ResolvedObserved, ...]
    expressions: ExpressionEntries
    free_values: FreeValueEntries
    stochastic_sites: tuple[ResolvedStochasticSite, ...]


@dataclass(frozen=True)
class _DimensionSnapshot:
    """An immutable snapshot of an optional dimension sidecar."""

    variables: tuple[tuple[str, tuple[str, ...]], ...]
    coords: tuple[tuple[str, tuple[CoordValue, ...]], ...]


@dataclass(frozen=True)
class _ParameterKernel:
    """Validated prior-producing view of one closed source model."""

    model_cls: ModelClass
    model: _ModelSnapshot
    dimensions: _DimensionSnapshot | None
    exports: tuple[_ParameterExport, ...]


@dataclass(frozen=True)
class _OutcomeKernel:
    """Target view with Param prior slots separated from retained outcomes."""

    model_cls: ModelClass
    model: _ModelSnapshot
    dimensions: _DimensionSnapshot | None
    inputs: tuple[_ParameterInput, ...]
    retained_free_values: FreeValueEntries
    retained_sites: tuple[ResolvedStochasticSite, ...]


@dataclass(frozen=True)
class _Wire:
    """One exact same-name target-input to source-export connection."""

    target: _ParameterInput
    source: _ParameterExport


@dataclass(frozen=True)
class _Wiring:
    """Complete ordered wiring for one source/target pair."""

    wires: tuple[_Wire, ...]


@dataclass(frozen=True)
class _ComposedKernel:
    """Validated closed-kernel composition before metadata materialization."""

    source: _ParameterKernel
    outcomes: _OutcomeKernel
    wiring: _Wiring


@dataclass(frozen=True)
class _ClosedComposition:
    """Ordinary flat metadata after all private inputs have been closed."""

    meta: ModelMeta
    dimensions: ResolvedModelDimensions | None
    dependencies: tuple[ModelClass, ...]


def _snapshot_model(meta: ModelMeta) -> _ModelSnapshot:
    """Copy mutable ordered maps into an immutable phase value."""
    from bayeswire.model.decorator import resolved_free_values, resolved_stochastic_sites

    for field_name in ("params", "data", "expressions", "free_values"):
        if not isinstance(getattr(meta, field_name), dict):
            raise TypeError(f"ModelMeta {field_name} must be a dict")
    if not isinstance(meta.observed_nodes, tuple) or not isinstance(
        meta.stochastic_sites,
        tuple,
    ):
        raise TypeError("ModelMeta observed nodes and stochastic sites must be tuples")

    return _ModelSnapshot(
        params=tuple(meta.params.items()),
        data=tuple(meta.data.items()),
        observed_nodes=meta.observed_nodes,
        expressions=tuple(meta.expressions.items()),
        free_values=tuple(resolved_free_values(meta).items()),
        stochastic_sites=resolved_stochastic_sites(meta),
    )


def _snapshot_dimensions(dimensions: ResolvedModelDimensions | None) -> _DimensionSnapshot | None:
    """Copy an optional mutable sidecar into an immutable phase value."""
    if dimensions is None:
        return None
    if not isinstance(dimensions.variables, dict) or not isinstance(dimensions.coords, dict):
        raise TypeError("dimension variables and coordinates must be dicts")
    if any(
        not isinstance(variable_dimensions, ResolvedVariableDims)
        for variable_dimensions in dimensions.variables.values()
    ):
        raise TypeError("dimension variables must contain ResolvedVariableDims values")
    return _DimensionSnapshot(
        variables=tuple(
            (name, variable_dimensions.names)
            for name, variable_dimensions in dimensions.variables.items()
        ),
        coords=tuple(dimensions.coords.items()),
    )


def _variable_dimensions(
    dimensions: _DimensionSnapshot | None,
    name: str,
) -> _VariableDimensions:
    """Return exact optional dimensions and coordinates for one variable."""
    if dimensions is None:
        return _VariableDimensions(names=None, coords=())
    variable_map = dict(dimensions.variables)
    names = variable_map.get(name)
    if names is None:
        return _VariableDimensions(names=None, coords=())
    coordinate_map = dict(dimensions.coords)
    return _VariableDimensions(
        names=names,
        coords=tuple((dimension, coordinate_map.get(dimension)) for dimension in names),
    )


def _data_schema_entries(model: _ModelSnapshot) -> tuple[tuple[str, ResolvedDataSchema], ...]:
    """Return immutable named schemas for diagnostics and validation."""
    return tuple((name, resolved.schema) for name, resolved in model.data)
