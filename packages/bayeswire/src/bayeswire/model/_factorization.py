"""Factor closed models into private prior and outcome kernels."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from bayeswire.model._components import (
    _DimensionSnapshot,
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
    ResolvedParam,
    ResolvedStochasticSite,
    attached_model_dimensions,
    is_model_class,
    model_meta,
)
from bayeswire.model.expr import ParamRef, VectorScatterOp


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

    return _ParameterKernel(
        model_cls=source,
        model=model,
        dimensions=dimensions,
        exports=validated_exports,
    )


def _factor_outcome_model(target: object) -> _OutcomeKernel:
    """Return a target view separating Param priors from retained factors."""
    meta = model_meta(target)
    if not isinstance(target, type):
        raise TypeError("target must be a bayeswire model class")
    model = _snapshot_model(meta)
    dimensions = _snapshot_dimensions(attached_model_dimensions(target))

    params = dict(model.params)
    free_values = dict(model.free_values)
    param_free_values = tuple(name for name in free_values if name in params)
    if param_free_values != tuple(params):
        raise ValueError("target parameter free values must match its parameters in order")

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
        owner_indices.add(owner_index)
        inputs.append(
            _ParameterInput(
                interface=_parameter_interface(name, param, dimensions),
                param=param,
                free_value=free_value,
                site=site,
            )
        )

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
