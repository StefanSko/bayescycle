"""Factor closed models into private prior and outcome kernels."""

from __future__ import annotations

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
    model_meta,
)
from bayeswire.model.expr import ParamRef


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
            f"{role} parameter {name!r} must have exactly one declaration-backed "
            f"stochastic site, found {len(matches)}"
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
            f"{role} parameter {name!r} must have a matching free-value constraint and size"
        )


def _factor_prior_model(source: object) -> _ParameterKernel:
    """Return a validated prior-only view of ``source``."""
    meta = model_meta(source)
    if not isinstance(source, type):
        raise TypeError("prior must be a bayeswire model class")
    model = _snapshot_model(meta)
    dimensions = _snapshot_dimensions(attached_model_dimensions(source))

    params = dict(model.params)
    free_values = dict(model.free_values)
    if tuple(free_values) != tuple(params):
        raise ValueError("prior free values must be exactly its parameters in parameter order")
    if model.observed_nodes:
        raise ValueError("prior must not contain observed declarations")

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
        raise ValueError("prior must not contain observed, partially observed, or factor sites")

    return _ParameterKernel(
        model_cls=source,
        model=model,
        dimensions=dimensions,
        exports=tuple(exports),
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
