"""Factor closed models into private prior and outcome kernels."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from bayeswire.constraints import VectorBounds
from bayeswire.model._closure import _validate_model_closure
from bayeswire.model._components import (
    _DimensionSnapshot,
    _ModelSnapshot,
    _OutcomeKernel,
    _ParameterExport,
    _ParameterInput,
    _ParameterInterface,
    _ParameterKernel,
    _snapshot_dimensions,
    _snapshot_model,
    _variable_dimensions,
)
from bayeswire.model.decorator import (
    ResolvedFreeValue,
    ResolvedObserved,
    ResolvedParam,
    ResolvedStochasticSite,
    attached_model_dimensions,
    is_model_class,
    model_meta,
)
from bayeswire.model.expr import DataRef, ParamRef, VectorScatterOp


def _parameter_interface(
    name: str,
    param: ResolvedParam,
    dimensions: _DimensionSnapshot | None,
) -> _ParameterInterface:
    return _ParameterInterface(
        name=name,
        constraint=param.constraint,
        size=param.size,
        dimensions=_variable_dimensions(dimensions, name),
    )


def _declaration_site(
    name: str,
    param: ResolvedParam,
    sites: tuple[ResolvedStochasticSite, ...],
    *,
    role: str,
) -> tuple[int, ResolvedStochasticSite]:
    matches = tuple(
        (index, site)
        for index, site in enumerate(sites)
        if site.value == ParamRef(name) and site.distribution == param.distribution
    )
    if len(matches) != 1:
        raise ValueError(
            f"{role} parameter {name!r} must have exactly one direct owner "
            f"stochastic site, found {len(matches)}; use a direct ParamRef with "
            "the Param's resolved distribution"
        )
    return matches[0]


def _validate_vector_bounds_param_owner(
    name: str,
    free_value: ResolvedFreeValue,
    owner: ResolvedStochasticSite,
    sites: tuple[ResolvedStochasticSite, ...],
    *,
    role: str,
) -> None:
    """Apply the specialized same-name owner rule for VectorBounds Params."""
    if not isinstance(free_value.constraint, VectorBounds):
        return
    named_sites = tuple(site for site in sites if site.name == name)
    if len(named_sites) != 1 or named_sites[0] is not owner:
        raise ValueError(
            f"{role} VectorBounds parameter {name!r} must use its structural Param "
            "site as the unique same-name owner"
        )


def _validate_param_free_value(
    name: str,
    param: ResolvedParam,
    free_value: ResolvedFreeValue,
    *,
    role: str,
) -> None:
    if free_value.constraint != param.constraint or free_value.size != param.size:
        raise ValueError(
            f"{role} free slot for parameter {name!r} must match its Param constraint and size"
        )


def _param_references(value: object) -> tuple[str, ...]:
    """Return ParamRef names nested in one resolved immutable value."""
    if isinstance(value, ParamRef):
        return (value.name,)
    if isinstance(value, dict):
        return tuple(name for item in value.values() for name in _param_references(item))
    if isinstance(value, tuple):
        return tuple(name for item in value for name in _param_references(item))
    if is_dataclass(value) and not isinstance(value, type):
        return tuple(
            name
            for value_field in fields(value)
            for name in _param_references(getattr(value, value_field.name))
        )
    return ()


def _validate_ancestral_param_order(exports: tuple[_ParameterExport, ...]) -> None:
    """Require every source prior dependency to refer to an earlier Param."""
    earlier: set[str] = set()
    for export in exports:
        name = export.interface.name
        invalid = tuple(
            reference
            for reference in _param_references(export.param.distribution)
            if reference not in earlier
        )
        if invalid:
            raise ValueError(
                f"prior parameter {name!r} may reference only an earlier Param in its "
                f"distribution; invalid references: {invalid}"
            )
        earlier.add(name)


def _reject_role_overlap(
    left: set[str],
    right: set[str],
    *,
    model_role: str,
    left_role: str,
    right_role: str,
) -> None:
    overlap = tuple(name for name in left if name in right)
    if overlap:
        raise ValueError(
            f"{model_role} name {overlap[0]!r} is both {left_role} and {right_role}; "
            "give every resolved value one unambiguous role"
        )


def _validate_internal_value_roles(model: _ModelSnapshot, *, model_role: str) -> None:
    """Reject all non-owner overlaps in one input's value namespace."""
    params = {name for name, _value in model.params}
    data = {name for name, _value in model.data}
    expressions = {name for name, _value in model.expressions}
    observed_names = [observed.name for observed in model.observed_nodes]
    observed = set(observed_names)
    if len(observed) != len(observed_names):
        raise ValueError(f"{model_role} contains duplicate Observed names")
    free_values = {name for name, _value in model.free_values}
    non_param_free = free_values - params

    _reject_role_overlap(
        params,
        data,
        model_role=model_role,
        left_role="a Param",
        right_role="data",
    )
    _reject_role_overlap(
        params,
        expressions,
        model_role=model_role,
        left_role="a Param",
        right_role="an expression",
    )
    _reject_role_overlap(
        params,
        observed,
        model_role=model_role,
        left_role="a Param",
        right_role="Observed",
    )
    _reject_role_overlap(
        data,
        expressions,
        model_role=model_role,
        left_role="data",
        right_role="an expression",
    )
    _reject_role_overlap(
        data,
        observed,
        model_role=model_role,
        left_role="data",
        right_role="Observed",
    )
    _reject_role_overlap(
        expressions,
        observed,
        model_role=model_role,
        left_role="an expression",
        right_role="Observed",
    )
    for role_names, role_name in (
        (params, "a Param"),
        (data, "data"),
        (expressions, "an expression"),
        (observed, "Observed"),
    ):
        _reject_role_overlap(
            non_param_free,
            role_names,
            model_role=model_role,
            left_role="a non-Param free value",
            right_role=role_name,
        )


def _factor_prior_model(source: object) -> _ParameterKernel:
    """Return a validated prior-only view of ``source``."""
    if not isinstance(source, type) or not is_model_class(source):
        raise TypeError(
            "prior must be a bayeswire model class decorated with @model or produced "
            "by bindable_from_meta(...)"
        )
    meta = model_meta(source)
    model = _snapshot_model(meta)
    dimensions = _snapshot_dimensions(attached_model_dimensions(source))
    _validate_internal_value_roles(model, model_role="source")

    params = dict(model.params)
    free_values = dict(model.free_values)
    if model.observed_nodes:
        observed = model.observed_nodes[0]
        raise ValueError(
            f"prior contains Observed declaration {observed.name!r}; use a prior-only "
            "source containing Params, data, and derived expressions"
        )

    extra_free_values = tuple(name for name in free_values if name not in params)
    for name in extra_free_values:
        owner = next((site for site in model.stochastic_sites if site.name == name), None)
        if owner is not None and isinstance(owner.value, VectorScatterOp):
            raise ValueError(
                f"prior contains PartiallyObserved free value {name!r}; move partial "
                "observations to the target model"
            )
        raise ValueError(
            f"prior free value {name!r} is not a declared Param; prior free slot names "
            "must match declared Params exactly"
        )
    if tuple(free_values) != tuple(params):
        raise ValueError("prior free slots must exactly match declared Params in parameter order")

    owner_indices: set[int] = set()
    exports: list[_ParameterExport] = []
    for name, param in model.params:
        free_value = free_values[name]
        _validate_param_free_value(name, param, free_value, role="prior")
        owner_index, site = _declaration_site(
            name,
            param,
            model.stochastic_sites,
            role="prior",
        )
        _validate_vector_bounds_param_owner(
            name,
            free_value,
            site,
            model.stochastic_sites,
            role="prior",
        )
        owner_indices.add(owner_index)
        exports.append(
            _ParameterExport(
                interface=_parameter_interface(name, param, dimensions),
                param=param,
                free_value=free_value,
                site=site,
            )
        )

    if len(owner_indices) != len(model.stochastic_sites):
        factor = next(
            site for index, site in enumerate(model.stochastic_sites) if index not in owner_indices
        )
        raise ValueError(
            f"prior density factor {factor.name!r} is not allowed; the source must contain "
            "only declaration-backed Param sites"
        )
    ordered_owner_indices = tuple(
        next(index for index, site in enumerate(model.stochastic_sites) if site is export.site)
        for export in exports
    )
    if ordered_owner_indices != tuple(sorted(ordered_owner_indices)):
        raise ValueError(
            "prior stochastic sites must follow source Param order so hierarchical "
            "generation remains ancestral"
        )
    validated_exports = tuple(exports)
    _validate_ancestral_param_order(validated_exports)
    _validate_model_closure(model, dimensions, role="source")

    return _ParameterKernel(
        model_cls=source,
        model=model,
        dimensions=dimensions,
        exports=validated_exports,
    )


def _is_canonical_free_value_owner(site: ResolvedStochasticSite, name: str) -> bool:
    """Return whether ``site`` has the existing same-name free-value owner form."""
    if site.name != name:
        return False
    if site.value == ParamRef(name):
        return True
    return isinstance(site.value, VectorScatterOp) and site.value.missing_values == ParamRef(name)


def _validate_target_non_param_owners(
    model_free_values: tuple[tuple[str, ResolvedFreeValue], ...],
    params: dict[str, ResolvedParam],
    sites: tuple[ResolvedStochasticSite, ...],
) -> None:
    """Require exactly one canonical owner for each retained free value."""
    for name, _free_value in model_free_values:
        if name in params:
            continue
        matches = tuple(site for site in sites if _is_canonical_free_value_owner(site, name))
        if len(matches) != 1:
            raise ValueError(
                f"target free value {name!r} must have exactly one canonical owner site, "
                f"found {len(matches)}; use a same-name direct ParamRef or "
                "VectorScatterOp missing-values owner"
            )


def _validate_target_observed_owners(
    observed_nodes: tuple[ResolvedObserved, ...],
    sites: tuple[ResolvedStochasticSite, ...],
) -> None:
    """Require one structurally associated site for every observed declaration."""
    for candidate in observed_nodes:
        matches = tuple(
            site
            for site in sites
            if site.value == DataRef(candidate.name) and site.distribution == candidate.distribution
        )
        if len(matches) != 1:
            raise ValueError(
                f"target Observed {candidate.name!r} must have exactly one direct owner "
                f"site, found {len(matches)}"
            )


def _factor_outcome_model(target: object) -> _OutcomeKernel:
    """Return a target view separating Param priors from retained factors."""
    if not isinstance(target, type) or not is_model_class(target):
        raise TypeError(
            "target must be a bayeswire model class decorated with @model or produced "
            "by bindable_from_meta(...)"
        )
    meta = model_meta(target)
    model = _snapshot_model(meta)
    dimensions = _snapshot_dimensions(attached_model_dimensions(target))
    _validate_internal_value_roles(model, model_role="target")

    params = dict(model.params)
    free_values = dict(model.free_values)
    param_free_values = tuple(name for name in free_values if name in params)
    if param_free_values != tuple(params):
        raise ValueError(
            "target free slots must include every Param exactly once in parameter order"
        )
    _validate_target_non_param_owners(model.free_values, params, model.stochastic_sites)
    _validate_target_observed_owners(model.observed_nodes, model.stochastic_sites)

    owner_indices: set[int] = set()
    inputs: list[_ParameterInput] = []
    for name, param in model.params:
        free_value = free_values[name]
        _validate_param_free_value(name, param, free_value, role="target")
        owner_index, site = _declaration_site(
            name,
            param,
            model.stochastic_sites,
            role="target",
        )
        _validate_vector_bounds_param_owner(
            name,
            free_value,
            site,
            model.stochastic_sites,
            role="target",
        )
        owner_indices.add(owner_index)
        inputs.append(
            _ParameterInput(
                interface=_parameter_interface(name, param, dimensions),
                param=param,
                free_value=free_value,
                site=site,
            )
        )

    _validate_model_closure(model, dimensions, role="target")
    return _OutcomeKernel(
        model_cls=target,
        model=model,
        dimensions=dimensions,
        inputs=tuple(inputs),
        retained_free_values=tuple(
            (name, free_value) for name, free_value in model.free_values if name not in params
        ),
        retained_sites=tuple(
            site for index, site in enumerate(model.stochastic_sites) if index not in owner_indices
        ),
    )
