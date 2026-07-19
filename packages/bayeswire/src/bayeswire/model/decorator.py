"""Model declaration resolution."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import SupportsFloat, cast

from bayeswire.constraints import Interval, Positive, UnitInterval, VectorBounds
from bayeswire.constraints.core import Constraint
from bayeswire.distributions._capabilities import has_scalar_inverse_cdf
from bayeswire.distributions._symbolic_validation import reject_opaque_symbolic_distribution
from bayeswire.distributions.continuous import (
    Beta,
    Exponential,
    HalfNormal,
    Normal,
    StudentT,
    Uniform,
)
from bayeswire.distributions.core import DiscreteDistribution, Distribution
from bayeswire.distributions.truncated import Truncated
from bayeswire.model._data_schema import (
    DataDimRef,
    DataDimSymbol,
    DataRankSchema,
    DataShapeSchema,
    ResolvedDataRankSchema,
    ResolvedDataSchema,
    ResolvedDataShapeDim,
    ResolvedDataShapeSchema,
    SubmodelDataDimSymbol,
)
from bayeswire.model._deferred import (
    DeclarationSymbol,
    DeferredBinOp,
    DeferredIndexOp,
    DeferredMatVecOp,
    DeferredUnaryOp,
    is_deferred_expr,
)
from bayeswire.model._expression_errors import (
    array_like_constant_error,
    is_array_like_constant,
    is_non_scalar_array_like_constant,
    non_scalar_distribution_parameter_error,
)
from bayeswire.model.core import (
    Data,
    Observed,
    Param,
    PartiallyObserved,
    Submodel,
    _submodel_member_state,
    _submodel_symbol,
    _SubmodelMember,
    _SubmodelMemberKind,
    submodel_target,
)
from bayeswire.model.dimensions import (
    CoordValue,
    Dim,
    ResolvedModelDimensions,
    ResolvedVariableDims,
)
from bayeswire.model.expr import (
    BinOp,
    ConstNode,
    DataRef,
    ExprNode,
    FullSlice,
    IndexOp,
    IndexSpec,
    IndexTuple,
    MatVecOp,
    ParamRef,
    ScalarIndex,
    UnaryOp,
    VectorScatterOp,
    is_final_expr_node,
)

type ModelClass = type[object]
type SymbolTable = dict[DeclarationSymbol, str]


@dataclass(frozen=True)
class ResolvedData:
    """Data metadata after declaration symbols are resolved to names."""

    schema: ResolvedDataSchema


@dataclass(frozen=True)
class ResolvedFreeValue:
    """Free NUTS coordinate metadata after declaration symbols are resolved."""

    constraint: Constraint | None
    size: DataRef | int | None


@dataclass(frozen=True)
class ResolvedParam:
    """Parameter declaration metadata after declaration symbols are resolved."""

    distribution: Distribution
    constraint: Constraint | None
    size: DataRef | int | None


@dataclass(frozen=True)
class ResolvedStochasticSite:
    """One log-density factor evaluated at a resolved model value expression."""

    name: str
    distribution: Distribution
    value: ExprNode


@dataclass(frozen=True)
class ResolvedObserved:
    """Observed likelihood metadata after declaration symbols are resolved to names."""

    name: str
    distribution: Distribution


@dataclass(frozen=True)
class ModelMeta:
    """Final model metadata attached by ``@model``."""

    params: dict[str, ResolvedParam]
    data: dict[str, ResolvedData]
    observed_nodes: tuple[ResolvedObserved, ...]
    expressions: dict[str, ExprNode]
    free_values: dict[str, ResolvedFreeValue] = field(default_factory=dict)
    stochastic_sites: tuple[ResolvedStochasticSite, ...] = ()


@dataclass(frozen=True)
class _ResolvedDeclarations:
    """Resolved top-level declarations from a declaration class."""

    params: dict[str, ResolvedParam]
    data: dict[str, ResolvedData]
    observed_nodes: tuple[ResolvedObserved, ...]
    free_values: dict[str, ResolvedFreeValue]
    stochastic_sites: tuple[ResolvedStochasticSite, ...]


def _qualified_name(prefix: str, name: str) -> str:
    """Return one opaque, dotted name in a composed model namespace."""
    return f"{prefix}.{name}"


def _prefix_model_meta(meta: ModelMeta, prefix: str) -> ModelMeta:
    """Flatten resolved child metadata under one opaque name prefix."""
    return ModelMeta(
        params={
            _qualified_name(prefix, name): ResolvedParam(
                distribution=_prefix_distribution(value.distribution, prefix),
                constraint=_prefix_constraint(value.constraint, prefix),
                size=_prefix_size(value.size, prefix),
            )
            for name, value in meta.params.items()
        },
        data={
            _qualified_name(prefix, name): ResolvedData(_prefix_data_schema(value.schema, prefix))
            for name, value in meta.data.items()
        },
        observed_nodes=tuple(
            ResolvedObserved(
                name=_qualified_name(prefix, value.name),
                distribution=_prefix_distribution(value.distribution, prefix),
            )
            for value in meta.observed_nodes
        ),
        expressions={
            _qualified_name(prefix, name): _prefix_expr(value, prefix)
            for name, value in meta.expressions.items()
        },
        free_values={
            _qualified_name(prefix, name): ResolvedFreeValue(
                constraint=_prefix_constraint(value.constraint, prefix),
                size=_prefix_size(value.size, prefix),
            )
            for name, value in resolved_free_values(meta).items()
        },
        stochastic_sites=tuple(
            ResolvedStochasticSite(
                name=_qualified_name(prefix, site.name),
                distribution=_prefix_distribution(site.distribution, prefix),
                value=_prefix_expr(site.value, prefix),
            )
            for site in resolved_stochastic_sites(meta)
        ),
    )


def _prefix_distribution(distribution: Distribution, prefix: str) -> Distribution:
    """Prefix every resolved declaration reference inside a distribution."""
    if not is_dataclass(distribution) or isinstance(distribution, type):
        return distribution
    resolved = {
        distribution_field.name: _prefix_resolved_value(
            getattr(distribution, distribution_field.name),
            prefix,
        )
        for distribution_field in fields(distribution)
    }
    return type(distribution)(**resolved)


def _prefix_constraint(constraint: Constraint | None, prefix: str) -> Constraint | None:
    """Prefix data references carried by a resolved constraint."""
    if constraint is None or not is_dataclass(constraint) or isinstance(constraint, type):
        return constraint
    resolved = {
        constraint_field.name: _prefix_resolved_value(
            getattr(constraint, constraint_field.name),
            prefix,
        )
        for constraint_field in fields(constraint)
    }
    return type(constraint)(**resolved)


def _prefix_resolved_value(value: object, prefix: str) -> object:
    """Prefix references recursively inside one resolved dataclass field."""
    if is_final_expr_node(value):
        return _prefix_expr(cast(ExprNode, value), prefix)
    if isinstance(value, DataDimRef):
        return DataDimRef(_qualified_name(prefix, value.name))
    if isinstance(value, dict):
        return {name: _prefix_resolved_value(item, prefix) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_prefix_resolved_value(item, prefix) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        resolved = {
            value_field.name: _prefix_resolved_value(
                getattr(value, value_field.name),
                prefix,
            )
            for value_field in fields(value)
        }
        return type(value)(**resolved)
    return value


def _prefix_expr(value: ExprNode, prefix: str) -> ExprNode:
    """Prefix every parameter and data reference in one final expression tree."""
    if isinstance(value, ParamRef):
        return ParamRef(_qualified_name(prefix, value.name))
    if isinstance(value, DataRef):
        return DataRef(_qualified_name(prefix, value.name))
    if isinstance(value, ConstNode):
        return value
    if isinstance(value, BinOp):
        return BinOp(
            value.op,
            _prefix_expr(value.left, prefix),
            _prefix_expr(value.right, prefix),
        )
    if isinstance(value, UnaryOp):
        return UnaryOp(value.function, _prefix_expr(value.operand, prefix))
    if isinstance(value, MatVecOp):
        return MatVecOp(
            matrix=_prefix_expr(value.matrix, prefix),
            vector=_prefix_expr(value.vector, prefix),
        )
    if isinstance(value, IndexOp):
        return IndexOp(
            _prefix_expr(value.base, prefix),
            _prefix_index_spec(value.index, prefix),
        )
    if isinstance(value, VectorScatterOp):
        return VectorScatterOp(
            length=_prefix_expr(value.length, prefix),
            observed_idx=_prefix_expr(value.observed_idx, prefix),
            observed_values=_prefix_expr(value.observed_values, prefix),
            missing_idx=_prefix_expr(value.missing_idx, prefix),
            missing_values=_prefix_expr(value.missing_values, prefix),
        )


def _prefix_index_spec(value: IndexSpec, prefix: str) -> IndexSpec:
    """Prefix expression references inside an explicit index specification."""
    if isinstance(value, ScalarIndex):
        return ScalarIndex(_prefix_expr(value.expr, prefix))
    if isinstance(value, FullSlice):
        return value
    return IndexTuple(tuple(_prefix_index_spec(item, prefix) for item in value.items))


def _prefix_data_schema(schema: ResolvedDataSchema, prefix: str) -> ResolvedDataSchema:
    """Prefix scalar-data references used by an exact shape schema."""
    if isinstance(schema, ResolvedDataRankSchema):
        return schema
    return ResolvedDataShapeSchema(
        tuple(
            DataDimRef(_qualified_name(prefix, dim.name)) if isinstance(dim, DataDimRef) else dim
            for dim in schema.dims
        )
    )


def _prefix_size(size: DataRef | int | None, prefix: str) -> DataRef | int | None:
    """Prefix a data-dependent free-value size."""
    if isinstance(size, DataRef):
        return DataRef(_qualified_name(prefix, size.name))
    return size


def _merge_unique[T](target: dict[str, T], source: dict[str, T], *, label: str) -> None:
    """Merge one ordered resolved map while rejecting ambiguous names."""
    for name, value in source.items():
        if name in target:
            raise ValueError(f"Composed model has duplicate {label} name {name!r}")
        target[name] = value


def _resolve_model_declaration(cls: ModelClass) -> ModelMeta:
    """Resolve a declaration class into final model metadata."""
    _reject_declaration_inheritance(cls)
    symbols = _collect_declaration_symbols(cls)
    declarations = _resolve_declarations(cls, symbols)
    expressions = _resolve_expressions(cls, symbols)

    return ModelMeta(
        params=declarations.params,
        data=declarations.data,
        observed_nodes=declarations.observed_nodes,
        expressions=expressions,
        free_values=declarations.free_values,
        stochastic_sites=declarations.stochastic_sites,
    )


def _reject_declaration_inheritance(cls: ModelClass) -> None:
    """Reject base classes so model meaning stays local to one class body."""
    if cls.__bases__ == (object,):
        return
    base_names = ", ".join(repr(base.__name__) for base in cls.__bases__ if base is not object)
    raise TypeError(
        f"Model declaration classes must not use inheritance: {cls.__name__!r} "
        f"inherits from {base_names}. All declarations must live in the decorated "
        "class body; inherited declarations would be silently ignored otherwise"
    )


def _collect_declaration_symbols(cls: ModelClass) -> SymbolTable:
    """Collect declaration symbols and reject declaration aliases."""
    symbols: SymbolTable = {}

    for name, value in cls.__dict__.items():
        if isinstance(value, Param | Data | Observed | PartiallyObserved | Submodel):
            symbol = _submodel_symbol(value) if isinstance(value, Submodel) else value.symbol
            existing_name = symbols.get(symbol)
            if existing_name is not None:
                raise ValueError(
                    "Declaration aliases are not supported: "
                    f"{existing_name!r} and {name!r} share one symbol"
                )
            symbols[symbol] = name

    return symbols


def _resolve_declarations(cls: ModelClass, symbols: SymbolTable) -> _ResolvedDeclarations:
    """Resolve top-level declaration inventory into final named metadata."""
    params: dict[str, ResolvedParam] = {}
    data: dict[str, ResolvedData] = {}
    observed_nodes: list[ResolvedObserved] = []
    free_values: dict[str, ResolvedFreeValue] = {}
    stochastic_sites: list[ResolvedStochasticSite] = []

    for name, value in cls.__dict__.items():
        if isinstance(value, Param):
            distribution = _resolve_declaration_distribution(value.distribution, symbols)
            if _contains_discrete_distribution(distribution):
                raise TypeError(
                    "Discrete distributions cannot be used as Param priors; "
                    "use them for Observed likelihoods or marginalize discrete latents"
                )
            size = _resolve_declaration_size(value.size, symbols)
            _validate_param_prior_constraint(
                name=name,
                distribution=distribution,
                constraint=value.constraint,
            )
            params[name] = ResolvedParam(
                distribution=distribution,
                constraint=value.constraint,
                size=size,
            )
            free_values[name] = ResolvedFreeValue(
                constraint=value.constraint,
                size=size,
            )
            stochastic_sites.append(
                ResolvedStochasticSite(
                    name=name,
                    distribution=distribution,
                    value=ParamRef(name),
                )
            )
        elif isinstance(value, Data):
            data[name] = ResolvedData(_resolve_data_schema(value.schema, symbols))
        elif isinstance(value, Observed):
            distribution = _resolve_declaration_distribution(value.distribution, symbols)
            _validate_supported_truncated_distribution(
                name=name,
                distribution=distribution,
                role="Observed",
            )
            observed_nodes.append(
                ResolvedObserved(
                    name=name,
                    distribution=distribution,
                )
            )
            stochastic_sites.append(
                ResolvedStochasticSite(
                    name=name,
                    distribution=distribution,
                    value=DataRef(name),
                )
            )
        elif isinstance(value, Submodel):
            child = _prefix_model_meta(model_meta(submodel_target(value)), name)
            _merge_unique(params, child.params, label="parameter")
            _merge_unique(data, child.data, label="data")
            observed_nodes.extend(child.observed_nodes)
            _merge_unique(free_values, child.free_values, label="free value")
            stochastic_sites.extend(child.stochastic_sites)
        elif isinstance(value, PartiallyObserved):
            distribution = _resolve_declaration_distribution(value.distribution, symbols)
            if _contains_discrete_distribution(distribution):
                raise TypeError(
                    "Discrete distributions cannot be partially observed NUTS values; "
                    "marginalize discrete missing values or impute them posterior-predictively"
                )
            if isinstance(distribution, Truncated) and _partially_observed_has_bounds(value):
                raise TypeError(
                    "bounds on PartiallyObserved cannot be combined with Truncated bases"
                )
            _validate_supported_truncated_distribution(
                name=name,
                distribution=distribution,
                role="PartiallyObserved",
            )
            free_values[name] = ResolvedFreeValue(
                constraint=_resolve_partially_observed_vector_bounds(value, symbols),
                size=_resolve_partially_observed_missing_size(value, symbols),
            )
            stochastic_sites.append(
                ResolvedStochasticSite(
                    name=name,
                    distribution=distribution,
                    value=_resolve_partially_observed_vector(value, name, symbols),
                )
            )

    if not stochastic_sites:
        raise ValueError("Model declarations must contain at least one stochastic declaration")

    return _ResolvedDeclarations(
        params=params,
        data=data,
        observed_nodes=tuple(observed_nodes),
        free_values=free_values,
        stochastic_sites=tuple(stochastic_sites),
    )


def _contains_discrete_distribution(distribution: Distribution) -> bool:
    """Return whether a distribution or first-level wrapper is discrete."""
    if isinstance(distribution, DiscreteDistribution):
        return True
    if isinstance(distribution, Truncated):
        return isinstance(distribution.base, DiscreteDistribution)
    return False


def _validate_param_prior_constraint(
    *,
    name: str,
    distribution: Distribution,
    constraint: Constraint | None,
) -> None:
    """Validate known concrete prior supports against explicit constraints."""
    if isinstance(distribution, Truncated):
        _validate_truncated_param_prior_constraint(
            name=name,
            distribution=distribution,
            constraint=constraint,
        )
        return

    if isinstance(distribution, Normal) and isinstance(
        constraint, Positive | Interval | UnitInterval
    ):
        raise TypeError(
            f"Parameter {name!r} uses a constrained Normal prior. Use Truncated(..., "
            "lower=..., upper=...) with a matching constraint so the truncation "
            "normalizer is explicit."
        )

    if isinstance(distribution, StudentT) and isinstance(
        constraint, Positive | Interval | UnitInterval
    ):
        raise TypeError(
            f"Parameter {name!r} uses a constrained StudentT prior, but StudentT "
            "truncation is not supported because the backend has no StudentT CDF"
        )

    if isinstance(distribution, Exponential | HalfNormal):
        if not isinstance(constraint, Positive):
            raise TypeError(
                f"Parameter {name!r} prior has support (0, inf); declare "
                f"Param({type(distribution).__name__}(...), constraint=Positive())"
            )
        return

    if isinstance(distribution, Beta):
        if not isinstance(constraint, UnitInterval):
            raise TypeError(
                f"Parameter {name!r} prior has support (0, 1); declare "
                "Param(Beta(...), constraint=UnitInterval())"
            )
        return

    if isinstance(distribution, Uniform):
        bounds = _concrete_uniform_bounds(distribution)
        if bounds is None:
            return
        low, high = bounds
        if _constraint_matches_uniform_support(constraint, low=low, high=high):
            return
        raise TypeError(
            f"Parameter {name!r} Uniform prior has support ({low}, {high}); declare "
            f"Param(Uniform({low}, {high}), constraint=Interval({low}, {high}))"
        )


def _validate_truncated_param_prior_constraint(
    *,
    name: str,
    distribution: Truncated,
    constraint: Constraint | None,
) -> None:
    """Require explicit truncation bounds to match the parameter constraint."""
    if isinstance(distribution.base, DiscreteDistribution):
        raise TypeError(
            "Discrete distributions cannot be used as Param priors; use them for Observed "
            "likelihoods or marginalize discrete latents"
        )
    _validate_supported_truncated_distribution(
        name=name,
        distribution=distribution,
        role="Parameter",
    )

    support = _truncated_effective_support(distribution)
    if support is None:
        raise TypeError(
            f"Parameter {name!r} Truncated prior uses symbolic or unknown bounds; "
            "constrained Param priors require concrete effective support that matches "
            "the declared constraint"
        )
    lower, upper = support
    if _constraint_matches_truncated_support(constraint, lower=lower, upper=upper):
        return
    raise TypeError(
        f"Parameter {name!r} Truncated prior effective support must match its constraint; "
        "use Positive() for lower=0 or Interval(lower, upper)/UnitInterval() for finite bounds"
    )


def _validate_supported_truncated_distribution(
    *,
    name: str,
    distribution: Distribution,
    role: str,
) -> None:
    """Reject Truncated distributions whose base cannot support normalization."""
    if not isinstance(distribution, Truncated):
        return
    if has_scalar_inverse_cdf(distribution.base):
        return
    raise TypeError(
        f"{role} {name!r} Truncated distribution base "
        f"{type(distribution.base).__name__} has no supported CDF/ICDF backend"
    )


def _truncated_effective_support(
    distribution: Truncated,
) -> tuple[float | None, float | None] | None:
    """Return concrete support after intersecting base support with truncation bounds."""
    base_support = _known_distribution_support(distribution.base)
    truncation_bounds = _concrete_truncated_bounds(distribution)
    if base_support is None or truncation_bounds is None:
        return None
    base_lower, base_upper = base_support
    truncation_lower, truncation_upper = truncation_bounds
    lower = _max_optional_bound(base_lower, truncation_lower)
    upper = _min_optional_bound(base_upper, truncation_upper)
    if lower is not None and upper is not None and lower >= upper:
        return None
    return (lower, upper)


def _known_distribution_support(
    distribution: Distribution,
) -> tuple[float | None, float | None] | None:
    """Return concrete scalar support for built-in distributions, or None if unknown."""
    if isinstance(distribution, Normal):
        return (None, None)
    if isinstance(distribution, HalfNormal | Exponential):
        return (0.0, None)
    if isinstance(distribution, Uniform):
        return _concrete_uniform_bounds(distribution)
    return None


def _max_optional_bound(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _min_optional_bound(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _constraint_matches_truncated_support(
    constraint: Constraint | None,
    *,
    lower: float | None,
    upper: float | None,
) -> bool:
    if upper is None and lower is not None and _same_scalar_bound(lower, 0.0):
        return isinstance(constraint, Positive)
    if lower is not None and upper is not None:
        return _constraint_matches_uniform_support(constraint, low=lower, high=upper)
    return False


def _concrete_truncated_bounds(distribution: Truncated) -> tuple[float | None, float | None] | None:
    """Return concrete scalar Truncated bounds, or None for symbolic bounds."""
    lower = None if distribution.lower is None else _concrete_scalar_bound(distribution.lower)
    upper = None if distribution.upper is None else _concrete_scalar_bound(distribution.upper)
    if (distribution.lower is not None and lower is None) or (
        distribution.upper is not None and upper is None
    ):
        return None
    return (lower, upper)


def _concrete_uniform_bounds(distribution: Uniform) -> tuple[float, float] | None:
    """Return concrete scalar Uniform bounds, or None for symbolic bounds."""
    low = _concrete_scalar_bound(distribution.low)
    high = _concrete_scalar_bound(distribution.high)
    if low is None or high is None:
        return None
    return (low, high)


def _concrete_scalar_bound(value: object) -> float | None:
    """Return a concrete scalar distribution bound after declaration resolution."""
    if isinstance(value, ConstNode):
        return float(value.value)
    if is_final_expr_node(value):
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        if tuple(shape) != ():
            return None
        return float(cast(SupportsFloat, value))
    if isinstance(value, int | float):
        return float(value)
    return None


def _constraint_matches_uniform_support(
    constraint: Constraint | None,
    *,
    low: float,
    high: float,
) -> bool:
    """Return whether a Uniform prior has an explicit compatible constraint."""
    if (
        _same_scalar_bound(low, 0.0)
        and _same_scalar_bound(high, 1.0)
        and isinstance(constraint, UnitInterval)
    ):
        return True
    if isinstance(constraint, Interval):
        return _same_scalar_bound(constraint.lower, low) and _same_scalar_bound(
            constraint.upper, high
        )
    return False


def _same_scalar_bound(left: float, right: float) -> bool:
    """Compare scalar bounds as exact Python floats.

    Declaration semantics are backend-free: no import-order-dependent float
    resolution sniffing. Bounds either match exactly or the declaration is
    rejected.
    """
    return float(left) == float(right)


def _resolve_expressions(cls: ModelClass, symbols: SymbolTable) -> dict[str, ExprNode]:
    """Resolve top-level derived declaration expressions into final IR."""
    expressions: dict[str, ExprNode] = {}

    for name, value in cls.__dict__.items():
        if isinstance(value, Submodel):
            child = _prefix_model_meta(model_meta(submodel_target(value)), name)
            _merge_unique(expressions, child.expressions, label="expression")
        elif isinstance(value, _SubmodelMember):
            expressions[name] = _resolve_submodel_member_expr(value, symbols)
        elif is_deferred_expr(value):
            expressions[name] = _resolve_declaration_expr(value, symbols)

    return expressions


def _resolve_dimension_metadata(cls: ModelClass) -> ResolvedModelDimensions:
    """Resolve authoring-side dimension labels into named metadata."""
    variables: dict[str, ResolvedVariableDims] = {}
    coords: dict[str, tuple[CoordValue, ...]] = {}

    for name, value in cls.__dict__.items():
        if isinstance(value, Param):
            if value.dims is None:
                if value.size is None:
                    variables[name] = ResolvedVariableDims(())
                continue
            variables[name] = _resolve_variable_dims(
                value.dims,
                coords,
                static_axis_sizes=_param_static_axis_sizes(value.size),
            )
        elif isinstance(value, Data):
            if value.dims is None:
                continue
            variables[name] = _resolve_variable_dims(
                value.dims,
                coords,
                static_axis_sizes=_data_static_axis_sizes(value.schema),
            )
        elif isinstance(value, Observed):
            if value.dims is None:
                continue
            variables[name] = _resolve_variable_dims(
                value.dims,
                coords,
                static_axis_sizes=tuple(None for _ in value.dims),
            )
        elif isinstance(value, Submodel):
            child_dimensions = attached_model_dimensions(submodel_target(value))
            if child_dimensions is None:
                continue
            prefixed = _prefix_model_dimensions(child_dimensions, name)
            _merge_unique(variables, prefixed.variables, label="dimension variable")
            _merge_dimension_coords(coords, prefixed.coords)

    return ResolvedModelDimensions(variables=variables, coords=coords)


def _prefix_model_dimensions(
    dimensions: ResolvedModelDimensions,
    prefix: str,
) -> ResolvedModelDimensions:
    """Prefix child variable and dimension names under a closed namespace."""
    return ResolvedModelDimensions(
        variables={
            _qualified_name(prefix, variable): ResolvedVariableDims(
                tuple(_qualified_name(prefix, name) for name in variable_dims.names)
            )
            for variable, variable_dims in dimensions.variables.items()
        },
        coords={
            _qualified_name(prefix, name): values for name, values in dimensions.coords.items()
        },
    )


def _merge_dimension_coords(
    target: dict[str, tuple[CoordValue, ...]],
    source: dict[str, tuple[CoordValue, ...]],
) -> None:
    """Merge prefixed coordinate metadata and reject conflicting definitions."""
    for name, values in source.items():
        existing = target.get(name)
        if existing is not None and existing != values:
            raise ValueError(f"Dimension {name!r} has conflicting coordinate values")
        target[name] = values


def _resolve_variable_dims(
    dims: tuple[Dim, ...],
    coords: dict[str, tuple[CoordValue, ...]],
    *,
    static_axis_sizes: tuple[int | None, ...],
) -> ResolvedVariableDims:
    _validate_static_coordinate_lengths(dims, static_axis_sizes)
    _record_dimension_coords(dims, coords)
    return ResolvedVariableDims(tuple(dim.name for dim in dims))


def _param_static_axis_sizes(size: Data | int | None) -> tuple[int | None, ...]:
    if size is None:
        return ()
    if isinstance(size, int):
        return (size,)
    return (None,)


def _data_static_axis_sizes(schema: DataRankSchema | DataShapeSchema) -> tuple[int | None, ...]:
    if isinstance(schema, DataRankSchema):
        return tuple(None for _ in range(schema.rank))
    return tuple(dim if isinstance(dim, int) else None for dim in schema.dims)


def _validate_static_coordinate_lengths(
    dims: tuple[Dim, ...],
    static_axis_sizes: tuple[int | None, ...],
) -> None:
    for dim, static_axis_size in zip(dims, static_axis_sizes, strict=True):
        if dim.coords is None or static_axis_size is None:
            continue
        if len(dim.coords) != static_axis_size:
            raise ValueError(
                f"Dimension {dim.name!r} coordinate length {len(dim.coords)} does not match "
                f"static axis size {static_axis_size}"
            )


def _record_dimension_coords(
    dims: tuple[Dim, ...],
    coords: dict[str, tuple[CoordValue, ...]],
) -> None:
    for dim in dims:
        if dim.coords is None:
            continue
        existing = coords.get(dim.name)
        if existing is not None and existing != dim.coords:
            raise ValueError(f"Dimension {dim.name!r} has conflicting coordinate values")
        coords[dim.name] = dim.coords


def model(cls: ModelClass) -> ModelClass:
    """Attach final model metadata to a declaration class."""
    meta = _resolve_model_declaration(cls)
    dimensions = _resolve_dimension_metadata(cls)
    setattr(cls, "_model_meta", meta)  # noqa: B010
    setattr(cls, "_model_dimensions", dimensions)  # noqa: B010
    return cls


def is_model_class(value: object) -> bool:
    """Return whether ``value`` is a bayeswire model class."""
    if not isinstance(value, type):
        return False
    return isinstance(value.__dict__.get("_model_meta"), ModelMeta)


def model_meta(model_cls: object) -> ModelMeta:
    """Return resolved model metadata attached by ``@model`` or IR decoding."""
    _cls, meta = _require_model_class_and_meta(model_cls)
    return meta


def _require_model_class_and_meta(value: object) -> tuple[ModelClass, ModelMeta]:
    if isinstance(value, type):
        metadata = value.__dict__.get("_model_meta")
        if isinstance(metadata, ModelMeta):
            return value, metadata
    raise TypeError(
        "Object is not a bayeswire model class; expected a class decorated with @model "
        "or produced by bindable_from_meta"
    )


def attached_model_dimensions(model_cls: object) -> ResolvedModelDimensions | None:
    """Return dimension metadata attached to a model class, or None when absent.

    A model reconstructed by ``bindable_from_meta(...)`` without the dimension
    sidecar carries no dimension metadata; backends use this accessor for the
    optional view where ``model_dimensions(...)`` would raise.
    """
    cls, _meta = _require_model_class_and_meta(model_cls)
    metadata = cls.__dict__.get("_model_dimensions")
    if isinstance(metadata, ResolvedModelDimensions):
        return metadata
    return None


def resolved_free_values(meta: ModelMeta) -> dict[str, ResolvedFreeValue]:
    """Return the free NUTS values defining flat state layout, in insertion order.

    Falls back to deriving the layout from ``params`` for metadata whose
    ``free_values`` field is empty.
    """
    if meta.free_values:
        return meta.free_values
    return {
        name: ResolvedFreeValue(constraint=param.constraint, size=param.size)
        for name, param in meta.params.items()
    }


def resolved_stochastic_sites(meta: ModelMeta) -> tuple[ResolvedStochasticSite, ...]:
    """Return the stochastic log-density sites, in declaration order.

    Falls back to deriving the sites from ``params`` and ``observed_nodes`` for
    metadata whose ``stochastic_sites`` field is empty.
    """
    if meta.stochastic_sites:
        return meta.stochastic_sites
    param_sites = tuple(
        ResolvedStochasticSite(
            name=name,
            distribution=param.distribution,
            value=ParamRef(name),
        )
        for name, param in meta.params.items()
    )
    observed_sites = tuple(
        ResolvedStochasticSite(
            name=observed.name,
            distribution=observed.distribution,
            value=DataRef(observed.name),
        )
        for observed in meta.observed_nodes
    )
    return param_sites + observed_sites


def _validate_parameter_size(size: int, label: str) -> int:
    if isinstance(size, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    if size < 0:
        raise ValueError(f"{label} must be non-negative")
    return size


def _resolve_data_schema(
    schema: DataRankSchema | DataShapeSchema, symbols: SymbolTable
) -> ResolvedDataSchema:
    """Resolve a data declaration schema into named metadata."""
    if isinstance(schema, DataRankSchema):
        return ResolvedDataRankSchema(schema.rank)
    return ResolvedDataShapeSchema(
        tuple(_resolve_data_shape_schema_dim(dim, symbols) for dim in schema.dims)
    )


def _resolve_data_shape_schema_dim(
    dim: int | DataDimSymbol | SubmodelDataDimSymbol,
    symbols: SymbolTable,
) -> ResolvedDataShapeDim:
    if isinstance(dim, int):
        return dim
    if isinstance(dim, DataDimSymbol):
        return DataDimRef(_resolve_symbol(dim.symbol, symbols))
    if isinstance(dim, SubmodelDataDimSymbol):
        prefix = _resolve_symbol(dim.submodel_symbol, symbols)
        return DataDimRef(_qualified_name(prefix, dim.member_path))
    raise TypeError(f"Unknown data shape dimension: {type(dim).__name__}")


def _resolve_submodel_data_ref(value: _SubmodelMember, symbols: SymbolTable) -> DataRef:
    """Resolve one child data member to its flattened qualified name."""
    state = _submodel_member_state(value)
    if state.kind is not _SubmodelMemberKind.DATA:
        raise TypeError(f"Submodel member {state.member_path!r} is not a data declaration")
    prefix = _resolve_symbol(state.submodel_symbol, symbols)
    return DataRef(_qualified_name(prefix, state.member_path))


def _resolve_submodel_member_expr(value: _SubmodelMember, symbols: SymbolTable) -> ExprNode:
    """Resolve one child member into the parent's final expression tree."""
    state = _submodel_member_state(value)
    if state.kind is _SubmodelMemberKind.DATA:
        return _resolve_submodel_data_ref(value, symbols)

    prefix = _resolve_symbol(state.submodel_symbol, symbols)
    qualified = _qualified_name(prefix, state.member_path)
    if state.kind is _SubmodelMemberKind.PARAM:
        return ParamRef(qualified)

    meta = model_meta(state.model_cls)
    if state.kind is _SubmodelMemberKind.EXPRESSION:
        return _prefix_expr(meta.expressions[state.member_path], prefix)
    if state.kind is _SubmodelMemberKind.PARTIALLY_OBSERVED:
        for site in resolved_stochastic_sites(meta):
            if site.name == state.member_path:
                return _prefix_expr(site.value, prefix)
        raise ValueError(
            f"Submodel member {state.member_path!r} has no stochastic value expression"
        )
    raise TypeError(f"Submodel namespace {state.member_path!r} is not a declaration expression")


def _resolve_declaration_size(size: object, symbols: SymbolTable) -> DataRef | int | None:
    """Resolve a declaration-size value into final size metadata."""
    if size is None:
        return None
    if isinstance(size, int):
        return _validate_parameter_size(size, "Parameter size")
    if isinstance(size, Data):
        if isinstance(size.schema, DataShapeSchema) and size.schema.dims == ():
            return DataRef(_resolve_symbol(size.symbol, symbols))
        if isinstance(size.schema, DataRankSchema) and size.schema.rank == 0:
            return DataRef(_resolve_symbol(size.symbol, symbols))
        raise TypeError("Data-dependent parameter sizes must use scalar data declarations")
    if isinstance(size, _SubmodelMember):
        state = _submodel_member_state(size)
        if (
            state.kind is _SubmodelMemberKind.DATA
            and isinstance(state.schema, ResolvedDataShapeSchema)
            and state.schema.dims == ()
        ):
            return _resolve_submodel_data_ref(size, symbols)
        if (
            state.kind is _SubmodelMemberKind.DATA
            and isinstance(state.schema, ResolvedDataRankSchema)
            and state.schema.rank == 0
        ):
            return _resolve_submodel_data_ref(size, symbols)
        raise TypeError("Data-dependent parameter sizes must use scalar data declarations")
    raise TypeError(f"Cannot resolve {type(size).__name__} as a declaration size")


def _resolve_partially_observed_missing_size(
    value: PartiallyObserved,
    symbols: SymbolTable,
) -> DataRef | int:
    """Resolve the free-coordinate size from an exact missing-index data schema."""
    missing_idx = value.missing_idx
    if isinstance(missing_idx, Data):
        schema = missing_idx.schema
        if not isinstance(schema, DataShapeSchema) or len(schema.dims) != 1:
            raise TypeError("PartiallyObserved missing_idx must be declared as Data.vector(length)")
        dim = schema.dims[0]
        if isinstance(dim, int):
            return _validate_parameter_size(dim, "PartiallyObserved missing size")
        if isinstance(dim, DataDimSymbol):
            return DataRef(_resolve_symbol(dim.symbol, symbols))
        if isinstance(dim, SubmodelDataDimSymbol):
            prefix = _resolve_symbol(dim.submodel_symbol, symbols)
            return DataRef(_qualified_name(prefix, dim.member_path))
    else:
        state = _submodel_member_state(missing_idx)
        schema = state.schema
        if isinstance(schema, ResolvedDataShapeSchema) and len(schema.dims) == 1:
            dim = schema.dims[0]
            if isinstance(dim, int):
                return _validate_parameter_size(dim, "PartiallyObserved missing size")
            if isinstance(dim, DataDimRef):
                prefix = _resolve_symbol(state.submodel_symbol, symbols)
                return DataRef(_qualified_name(prefix, dim.name))
    raise TypeError("PartiallyObserved missing_idx must be declared as Data.vector(length)")


def _partially_observed_has_bounds(value: PartiallyObserved) -> bool:
    return value.missing_lower is not None or value.missing_upper is not None


def _resolve_partially_observed_vector_bounds(
    value: PartiallyObserved,
    symbols: SymbolTable,
) -> VectorBounds | None:
    """Resolve optional per-missing-coordinate bounds for a partial vector."""
    if not _partially_observed_has_bounds(value):
        return None
    return VectorBounds(
        lower=_resolve_optional_data_ref(value.missing_lower, symbols),
        upper=_resolve_optional_data_ref(value.missing_upper, symbols),
    )


def _resolve_optional_data_ref(
    value: Data | _SubmodelMember | None,
    symbols: SymbolTable,
) -> DataRef | None:
    if value is None:
        return None
    if isinstance(value, _SubmodelMember):
        return _resolve_submodel_data_ref(value, symbols)
    return DataRef(_resolve_symbol(value.symbol, symbols))


def _resolve_partially_observed_vector(
    value: PartiallyObserved,
    name: str,
    symbols: SymbolTable,
) -> VectorScatterOp:
    """Resolve a partially observed declaration into full-vector assembly IR."""
    return VectorScatterOp(
        length=_resolve_declaration_expr(value.length, symbols),
        observed_idx=_resolve_declaration_expr(value.observed_idx, symbols),
        observed_values=_resolve_declaration_expr(value.observed, symbols),
        missing_idx=_resolve_declaration_expr(value.missing_idx, symbols),
        missing_values=ParamRef(name),
    )


def _resolve_declaration_distribution(
    distribution: Distribution,
    symbols: SymbolTable,
) -> Distribution:
    """Resolve symbolic distribution fields into final expression nodes."""
    if not is_dataclass(distribution) or isinstance(distribution, type):
        reject_opaque_symbolic_distribution(distribution)
        return distribution
    resolved = {
        distribution_field.name: _resolve_declaration_distribution_field(
            getattr(distribution, distribution_field.name),
            symbols,
        )
        for distribution_field in fields(distribution)
    }
    return type(distribution)(**resolved)


def _resolve_declaration_distribution_field(value: object, symbols: SymbolTable) -> object:
    if _is_declaration_expr(value):
        return _resolve_declaration_expr(value, symbols)
    if is_final_expr_node(value):
        raise TypeError("Final expression nodes are not valid in model declarations")
    if is_dataclass(value) and not isinstance(value, type):
        return _resolve_declaration_distribution(cast(Distribution, value), symbols)
    if is_non_scalar_array_like_constant(value):
        raise non_scalar_distribution_parameter_error()
    reject_opaque_symbolic_distribution(value)
    return value


def _resolve_index_spec(value: object, symbols: SymbolTable) -> IndexSpec:
    """Resolve raw class-body indexing syntax into explicit final index IR."""
    if isinstance(value, slice):
        return _resolve_slice_index_spec(value)
    if isinstance(value, tuple):
        if not value:
            raise TypeError("Empty index tuples are not supported in model declarations")
        return IndexTuple(tuple(_resolve_index_tuple_item(item, symbols) for item in value))
    if isinstance(value, bool):
        raise TypeError("Index constants must be integers, not bool")
    if isinstance(value, ScalarIndex | FullSlice | IndexTuple):
        raise TypeError("Final index nodes are not valid in model declarations")
    return ScalarIndex(_resolve_declaration_expr(value, symbols))


def _resolve_index_tuple_item(value: object, symbols: SymbolTable) -> IndexSpec:
    if isinstance(value, tuple):
        raise TypeError("Nested index tuples are not supported in model declarations")
    return _resolve_index_spec(value, symbols)


def _resolve_slice_index_spec(value: slice) -> FullSlice:
    if value.start is None and value.stop is None and value.step is None:
        return FullSlice()
    raise TypeError("Only full slices ':' are supported in model declaration indexes")


def _resolve_declaration_expr(value: object, symbols: SymbolTable) -> ExprNode:
    """Resolve class-body declaration syntax into final expression IR."""
    if isinstance(value, Param):
        return ParamRef(_resolve_symbol(value.symbol, symbols))
    if isinstance(value, Data):
        return DataRef(_resolve_symbol(value.symbol, symbols))
    if isinstance(value, PartiallyObserved):
        return _resolve_partially_observed_vector(
            value,
            _resolve_symbol(value.symbol, symbols),
            symbols,
        )
    if isinstance(value, _SubmodelMember):
        return _resolve_submodel_member_expr(value, symbols)
    if isinstance(value, int | float):
        return ConstNode(value)
    if is_array_like_constant(value):
        raise array_like_constant_error()
    if isinstance(value, DeferredBinOp):
        return BinOp(
            value.op,
            _resolve_declaration_expr(value.left, symbols),
            _resolve_declaration_expr(value.right, symbols),
        )
    if isinstance(value, DeferredUnaryOp):
        return UnaryOp(
            value.function,
            _resolve_declaration_expr(value.operand, symbols),
        )
    if isinstance(value, DeferredMatVecOp):
        return MatVecOp(
            matrix=_resolve_declaration_expr(value.matrix, symbols),
            vector=_resolve_declaration_expr(value.vector, symbols),
        )
    if isinstance(value, DeferredIndexOp):
        return IndexOp(
            _resolve_declaration_expr(value.base, symbols),
            _resolve_index_spec(value.index, symbols),
        )
    raise TypeError(f"Cannot resolve {type(value).__name__} as a declaration expression")


def _is_declaration_expr(value: object) -> bool:
    """Return whether ``value`` can resolve to final expression IR."""
    return isinstance(
        value,
        Param
        | Data
        | PartiallyObserved
        | _SubmodelMember
        | DeferredBinOp
        | DeferredIndexOp
        | DeferredMatVecOp
        | DeferredUnaryOp
        | int
        | float,
    )


def _resolve_symbol(symbol: DeclarationSymbol, symbols: SymbolTable) -> str:
    name = symbols.get(symbol)
    if name is None:
        raise ValueError(f"Unknown declaration symbol: {symbol}")
    return name
