"""Core model declaration types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import count

from bayeswire.constraints.core import Constraint
from bayeswire.distributions.core import Distribution, SymbolicDistributionParameter
from bayeswire.model._data_schema import (
    DataDimRef,
    DataDimSymbol,
    DataRankSchema,
    DataSchema,
    DataShapeDim,
    DataShapeSchema,
    ResolvedDataRankSchema,
    ResolvedDataSchema,
    ResolvedDataShapeSchema,
    SubmodelDataDimSymbol,
    is_scalar_data_schema,
)
from bayeswire.model._deferred import (
    DeclarationSymbol,
    DeferredBinOp,
    DeferredIndexOp,
    DeferredUnaryOp,
)
from bayeswire.model.dimensions import Dim, normalize_dims

_SYMBOL_IDS = count()


def _next_symbol() -> DeclarationSymbol:
    """Return a fresh declaration symbol."""
    return DeclarationSymbol(next(_SYMBOL_IDS))


def _validate_static_data_dimension(value: int, *, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _validate_data_rank(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("Data rank must be an integer, not bool")
    if value < 0:
        raise ValueError("Data rank must be non-negative")
    return value


def _validate_dims_rank(
    dims: tuple[Dim, ...] | None,
    *,
    rank: int,
    label: str,
) -> tuple[Dim, ...] | None:
    if dims is None:
        return None
    if len(dims) != rank:
        raise ValueError(f"{label} dims length must match rank {rank}, got {len(dims)}")
    return dims


def _data_schema_rank(schema: DataSchema) -> int:
    if isinstance(schema, DataShapeSchema):
        return len(schema.dims)
    return schema.rank


def _is_scalar_resolved_data_schema(schema: ResolvedDataSchema) -> bool:
    if isinstance(schema, ResolvedDataShapeSchema):
        return schema.dims == ()
    return isinstance(schema, ResolvedDataRankSchema) and schema.rank == 0


class _SubmodelMemberKind(Enum):
    """Referenceable member categories exposed by a composed model."""

    PARAM = "param"
    DATA = "data"
    EXPRESSION = "expression"
    PARTIALLY_OBSERVED = "partially_observed"
    NAMESPACE = "namespace"


@dataclass(frozen=True)
class _SubmodelMemberState:
    """Internal state kept behind one non-semantic reserved attribute."""

    submodel_symbol: DeclarationSymbol
    model_cls: type[object]
    member_path: str
    kind: _SubmodelMemberKind
    schema: ResolvedDataSchema | None = None


@dataclass(frozen=True)
class _SubmodelMember(SymbolicDistributionParameter):
    """Typed declaration proxy for one member of a closed submodel namespace."""

    _bayeswire_state: _SubmodelMemberState

    def __getattr__(self, name: str) -> _SubmodelMember:
        state = _submodel_member_state(self)
        if state.kind is not _SubmodelMemberKind.NAMESPACE:
            raise AttributeError(f"Submodel member {state.member_path!r} has no child members")
        path = f"{state.member_path}.{name}"
        return _resolve_submodel_member(state.submodel_symbol, state.model_cls, path)

    def __add__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", self, other)

    def __radd__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", other, self)

    def __sub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", self, other)

    def __rsub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", other, self)

    def __mul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", self, other)

    def __rmul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", other, self)

    def __truediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", self, other)

    def __rtruediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", other, self)

    def __neg__(self) -> DeferredUnaryOp:
        return DeferredUnaryOp("neg", self)

    def __getitem__(self, index: object) -> DeferredIndexOp:
        return DeferredIndexOp(self, index)


@dataclass(frozen=True)
class _SubmodelState:
    """Internal target and declaration identity for one submodel instance."""

    model_cls: type[object]
    symbol: DeclarationSymbol


@dataclass(frozen=True, init=False)
class Submodel:
    """Closed, namespaced composition of an already-resolved model declaration."""

    _bayeswire_state: _SubmodelState

    def __init__(self, model_cls: object) -> None:
        from bayeswire.model.decorator import is_model_class, model_meta, resolved_free_values

        if not isinstance(model_cls, type) or not is_model_class(model_cls):
            raise TypeError("Submodel requires a bayeswire model class decorated with @model")
        meta = model_meta(model_cls)
        names = (
            *meta.params,
            *meta.data,
            *meta.expressions,
            *resolved_free_values(meta),
            *(observed.name for observed in meta.observed_nodes),
        )
        if any("_bayeswire_state" in name.split(".") for name in names):
            raise ValueError(
                "Submodel member name '_bayeswire_state' is reserved for declaration state"
            )
        object.__setattr__(self, "_bayeswire_state", _SubmodelState(model_cls, _next_symbol()))

    def __getattr__(self, name: str) -> _SubmodelMember:
        state = _submodel_state(self)
        return _resolve_submodel_member(state.symbol, state.model_cls, name)


def submodel_target(value: Submodel) -> type[object]:
    """Return the model class targeted by a Submodel declaration."""
    return _submodel_state(value).model_cls


def _submodel_state(value: Submodel) -> _SubmodelState:
    return object.__getattribute__(value, "_bayeswire_state")


def _submodel_symbol(value: Submodel) -> DeclarationSymbol:
    return _submodel_state(value).symbol


def _submodel_member_state(value: _SubmodelMember) -> _SubmodelMemberState:
    return object.__getattribute__(value, "_bayeswire_state")


def _submodel_member(
    submodel_symbol: DeclarationSymbol,
    model_cls: type[object],
    member_path: str,
    kind: _SubmodelMemberKind,
    schema: ResolvedDataSchema | None = None,
) -> _SubmodelMember:
    return _SubmodelMember(
        _SubmodelMemberState(
            submodel_symbol=submodel_symbol,
            model_cls=model_cls,
            member_path=member_path,
            kind=kind,
            schema=schema,
        )
    )


def _resolve_submodel_member(
    submodel_symbol: DeclarationSymbol,
    model_cls: type[object],
    member_path: str,
) -> _SubmodelMember:
    """Return a deferred member reference from resolved child metadata."""
    from bayeswire.model.decorator import model_meta, resolved_free_values

    meta = model_meta(model_cls)
    if member_path in meta.params:
        return _submodel_member(
            submodel_symbol,
            model_cls,
            member_path,
            _SubmodelMemberKind.PARAM,
        )
    data = meta.data.get(member_path)
    if data is not None:
        return _submodel_member(
            submodel_symbol,
            model_cls,
            member_path,
            _SubmodelMemberKind.DATA,
            data.schema,
        )
    if member_path in meta.expressions:
        return _submodel_member(
            submodel_symbol,
            model_cls,
            member_path,
            _SubmodelMemberKind.EXPRESSION,
        )
    if member_path in resolved_free_values(meta):
        return _submodel_member(
            submodel_symbol,
            model_cls,
            member_path,
            _SubmodelMemberKind.PARTIALLY_OBSERVED,
        )
    if any(observed.name == member_path for observed in meta.observed_nodes):
        raise AttributeError(
            f"Observed submodel member {member_path!r} contributes a likelihood factor "
            "but is not a declaration expression"
        )

    prefix = f"{member_path}."
    names = (
        *meta.params,
        *meta.data,
        *meta.expressions,
        *resolved_free_values(meta),
        *(observed.name for observed in meta.observed_nodes),
    )
    if any(name.startswith(prefix) for name in names):
        return _submodel_member(
            submodel_symbol,
            model_cls,
            member_path,
            _SubmodelMemberKind.NAMESPACE,
        )
    raise AttributeError(f"Submodel has no referenceable member {member_path!r}")


@dataclass(frozen=True, init=False)
class Param(SymbolicDistributionParameter):
    """Parameter declaration used inside ``@model`` class bodies."""

    distribution: Distribution
    constraint: Constraint | None
    size: Data | _SubmodelMember | int | None
    dims: tuple[Dim, ...] | None
    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)

    def __init__(
        self,
        distribution: Distribution,
        constraint: Constraint | None = None,
        size: Data | _SubmodelMember | int | None = None,
        *,
        dims: Sequence[Dim] | None = None,
    ) -> None:
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "constraint", constraint)
        object.__setattr__(self, "size", size)
        object.__setattr__(
            self,
            "dims",
            _validate_dims_rank(
                normalize_dims(dims),
                rank=0 if size is None else 1,
                label="Param",
            ),
        )
        object.__setattr__(self, "symbol", _next_symbol())

    def __add__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", self, other)

    def __radd__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", other, self)

    def __sub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", self, other)

    def __rsub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", other, self)

    def __mul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", self, other)

    def __rmul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", other, self)

    def __truediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", self, other)

    def __rtruediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", other, self)

    def __neg__(self) -> DeferredUnaryOp:
        return DeferredUnaryOp("neg", self)

    def __getitem__(self, index: object) -> DeferredIndexOp:
        return DeferredIndexOp(self, index)


@dataclass(frozen=True, init=False)
class Data(SymbolicDistributionParameter):
    """Data declaration used inside ``@model`` class bodies."""

    schema: DataSchema
    dims: tuple[Dim, ...] | None
    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)

    def __init__(
        self,
        *,
        shape: Sequence[object] | None = None,
        rank: int | None = None,
        dims: Sequence[Dim] | None = None,
    ) -> None:
        if shape is None and rank is None:
            raise TypeError(
                "Data declarations require a schema. Use Data.scalar(), Data.vector(...), "
                "Data.matrix(...), or Data.array(...)."
            )
        if shape is not None and rank is not None:
            raise TypeError("Data declarations must specify either shape or rank, not both")

        if shape is not None:
            schema: DataSchema = DataShapeSchema(
                tuple(self._normalize_shape_dim(dim) for dim in shape)
            )
        else:
            if rank is None:
                raise TypeError("Data rank is required")
            schema = DataRankSchema(_validate_data_rank(rank))

        object.__setattr__(self, "schema", schema)
        object.__setattr__(
            self,
            "dims",
            _validate_dims_rank(
                normalize_dims(dims),
                rank=_data_schema_rank(schema),
                label="Data",
            ),
        )
        object.__setattr__(self, "symbol", _next_symbol())

    @classmethod
    def scalar(cls, *, dims: Sequence[Dim] | None = None) -> Data:
        """Declare scalar data."""
        return cls(shape=(), dims=dims)

    @classmethod
    def vector(cls, length: object | None = None, *, dims: Sequence[Dim] | None = None) -> Data:
        """Declare rank-1 data, optionally with an exact length."""
        if length is None:
            return cls(rank=1, dims=dims)
        return cls(shape=(length,), dims=dims)

    @classmethod
    def matrix(
        cls,
        rows: object | None = None,
        cols: object | None = None,
        *,
        dims: Sequence[Dim] | None = None,
    ) -> Data:
        """Declare rank-2 data, optionally with an exact shape."""
        if rows is None and cols is None:
            return cls(rank=2, dims=dims)
        if rows is None or cols is None:
            raise TypeError("Data.matrix requires both rows and cols, or neither")
        return cls(shape=(rows, cols), dims=dims)

    @classmethod
    def array(
        cls,
        *,
        shape: Sequence[object] | None = None,
        rank: int | None = None,
        dims: Sequence[Dim] | None = None,
    ) -> Data:
        """Declare generic array data by exact shape or rank."""
        if shape is None and rank is None:
            raise TypeError("Data.array requires shape or rank")
        return cls(shape=shape, rank=rank, dims=dims)

    @staticmethod
    def _normalize_shape_dim(dim: object) -> DataShapeDim:
        if isinstance(dim, bool):
            raise TypeError("Data shape dimensions must be integers, not bool")
        if isinstance(dim, int):
            return _validate_static_data_dimension(dim, label="Data shape dimension")
        if isinstance(dim, Data):
            if not is_scalar_data_schema(dim.schema):
                raise TypeError("Data shape dimensions must reference scalar data declarations")
            return DataDimSymbol(dim.symbol)
        if isinstance(dim, _SubmodelMember):
            state = _submodel_member_state(dim)
            if (
                state.kind is not _SubmodelMemberKind.DATA
                or state.schema is None
                or not _is_scalar_resolved_data_schema(state.schema)
            ):
                raise TypeError("Data shape dimensions must reference scalar data declarations")
            return SubmodelDataDimSymbol(state.submodel_symbol, state.member_path)
        raise TypeError("Data shape dimensions must be integers or scalar data declarations")

    def __add__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", self, other)

    def __radd__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", other, self)

    def __sub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", self, other)

    def __rsub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", other, self)

    def __mul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", self, other)

    def __rmul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", other, self)

    def __truediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", self, other)

    def __rtruediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", other, self)

    def __neg__(self) -> DeferredUnaryOp:
        return DeferredUnaryOp("neg", self)

    def __getitem__(self, index: object) -> DeferredIndexOp:
        return DeferredIndexOp(self, index)


@dataclass(frozen=True, init=False)
class Observed:
    """Observed variable declaration."""

    distribution: Distribution
    dims: tuple[Dim, ...] | None
    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)

    def __init__(
        self,
        distribution: Distribution,
        *,
        dims: Sequence[Dim] | None = None,
    ) -> None:
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "dims", normalize_dims(dims))
        object.__setattr__(self, "symbol", _next_symbol())


@dataclass(frozen=True, init=False)
class PartiallyObserved(SymbolicDistributionParameter):
    """Partially observed vector declaration.

    The declaration contributes one log-density factor evaluated at the full
    assembled vector. Coordinates listed by ``missing_idx`` are free NUTS
    coordinates; coordinates listed by ``observed_idx`` are fixed data.
    """

    distribution: Distribution
    length: Data | _SubmodelMember | int
    observed: Data | _SubmodelMember
    observed_idx: Data | _SubmodelMember
    missing_idx: Data | _SubmodelMember
    missing_lower: Data | _SubmodelMember | None
    missing_upper: Data | _SubmodelMember | None
    symbol: DeclarationSymbol = field(default_factory=_next_symbol, init=False, repr=False)

    def __init__(
        self,
        distribution: Distribution,
        *,
        length: Data | _SubmodelMember | int,
        observed: Data | _SubmodelMember,
        observed_idx: Data | _SubmodelMember,
        missing_idx: Data | _SubmodelMember,
        missing_lower: Data | _SubmodelMember | None = None,
        missing_upper: Data | _SubmodelMember | None = None,
    ) -> None:
        validated_missing_idx = _validate_exact_vector_data(
            missing_idx,
            label="PartiallyObserved missing_idx",
        )
        object.__setattr__(self, "distribution", distribution)
        object.__setattr__(self, "length", _validate_partial_vector_length(length))
        object.__setattr__(
            self,
            "observed",
            _validate_exact_vector_data(observed, label="PartiallyObserved observed values"),
        )
        object.__setattr__(
            self,
            "observed_idx",
            _validate_exact_vector_data(observed_idx, label="PartiallyObserved observed_idx"),
        )
        object.__setattr__(self, "missing_idx", validated_missing_idx)
        object.__setattr__(
            self,
            "missing_lower",
            _validate_partially_observed_bound_data(
                missing_lower,
                missing_idx=validated_missing_idx,
                label="PartiallyObserved missing_lower",
            ),
        )
        object.__setattr__(
            self,
            "missing_upper",
            _validate_partially_observed_bound_data(
                missing_upper,
                missing_idx=validated_missing_idx,
                label="PartiallyObserved missing_upper",
            ),
        )
        object.__setattr__(self, "symbol", _next_symbol())

    @classmethod
    def vector(
        cls,
        distribution: Distribution,
        *,
        length: Data | _SubmodelMember | int,
        observed: Data | _SubmodelMember,
        observed_idx: Data | _SubmodelMember,
        missing_idx: Data | _SubmodelMember,
        missing_lower: Data | _SubmodelMember | None = None,
        missing_upper: Data | _SubmodelMember | None = None,
    ) -> PartiallyObserved:
        """Declare a rank-1 random vector with explicit observed/missing coordinates."""
        return cls(
            distribution,
            length=length,
            observed=observed,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
            missing_lower=missing_lower,
            missing_upper=missing_upper,
        )

    def __add__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", self, other)

    def __radd__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("+", other, self)

    def __sub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", self, other)

    def __rsub__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("-", other, self)

    def __mul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", self, other)

    def __rmul__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("*", other, self)

    def __truediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", self, other)

    def __rtruediv__(self, other: object) -> DeferredBinOp:
        return DeferredBinOp("/", other, self)

    def __neg__(self) -> DeferredUnaryOp:
        return DeferredUnaryOp("neg", self)

    def __getitem__(self, index: object) -> DeferredIndexOp:
        return DeferredIndexOp(self, index)


def _validate_partial_vector_length(
    length: Data | _SubmodelMember | int,
) -> Data | _SubmodelMember | int:
    if isinstance(length, bool):
        raise TypeError("PartiallyObserved.vector length must be an integer, not bool")
    if isinstance(length, int):
        return _validate_static_data_dimension(length, label="PartiallyObserved.vector length")
    if isinstance(length, Data):
        if not is_scalar_data_schema(length.schema):
            raise TypeError("PartiallyObserved.vector length must reference scalar data")
        return length
    if isinstance(length, _SubmodelMember):
        state = _submodel_member_state(length)
        if (
            state.kind is not _SubmodelMemberKind.DATA
            or state.schema is None
            or not _is_scalar_resolved_data_schema(state.schema)
        ):
            raise TypeError("PartiallyObserved.vector length must reference scalar data")
        return length
    raise TypeError("PartiallyObserved.vector length must be an integer or scalar data declaration")


def _validate_exact_vector_data(
    value: Data | _SubmodelMember,
    *,
    label: str,
) -> Data | _SubmodelMember:
    if isinstance(value, Data):
        if isinstance(value.schema, DataShapeSchema) and len(value.schema.dims) == 1:
            return value
    else:
        state = _submodel_member_state(value)
        if (
            state.kind is _SubmodelMemberKind.DATA
            and isinstance(state.schema, ResolvedDataShapeSchema)
            and len(state.schema.dims) == 1
        ):
            return value
    raise TypeError(f"{label} must be a Data.vector(length) declaration")


def _validate_partially_observed_bound_data(
    value: Data | _SubmodelMember | None,
    *,
    missing_idx: Data | _SubmodelMember,
    label: str,
) -> Data | _SubmodelMember | None:
    if value is None:
        return None
    bound = _validate_exact_vector_data(value, label=label)
    if _exact_vector_dim(bound) != _exact_vector_dim(missing_idx):
        raise TypeError(f"{label} must use the same length dimension as missing_idx")
    return bound


def _exact_vector_dim(value: Data | _SubmodelMember) -> object:
    if isinstance(value, Data):
        schema = value.schema
        if not isinstance(schema, DataShapeSchema) or len(schema.dims) != 1:
            raise TypeError("Data declaration must be an exact vector")
        dim = schema.dims[0]
        if isinstance(dim, SubmodelDataDimSymbol):
            return (dim.submodel_symbol, dim.member_path)
        return dim

    state = _submodel_member_state(value)
    schema = state.schema
    if not isinstance(schema, ResolvedDataShapeSchema) or len(schema.dims) != 1:
        raise TypeError("Data declaration must be an exact vector")
    dim = schema.dims[0]
    if isinstance(dim, DataDimRef):
        return (state.submodel_symbol, dim.name)
    return dim
