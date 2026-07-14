"""Immutable authoring-time prior composition."""

from __future__ import annotations

import sys

from bayeswire.model._components import _ClosedComposition
from bayeswire.model._compose import (
    _build_same_name_wiring,
    _close_composition,
    _compose_kernels,
)
from bayeswire.model._factorization import _factor_outcome_model, _factor_prior_model


def with_prior(target: object, *, prior: object) -> type[object]:
    """Return a new closed model using ``prior`` for ``target`` parameters."""
    caller_module = _caller_module_name()
    source = _factor_prior_model(prior)
    outcomes = _factor_outcome_model(target)
    wiring = _build_same_name_wiring(source, outcomes)
    composition = _compose_kernels(source, outcomes, wiring)
    closed = _close_composition(composition)
    return _model_class_from_closed(closed, module_name=caller_module)


def _caller_module_name() -> str:
    """Return the module that invoked the public authoring operation."""
    module_name = sys._getframe(2).f_globals.get("__name__")
    if not isinstance(module_name, str):
        raise RuntimeError("with_prior(...) must be called from a named Python module")
    return module_name


def _model_class_from_closed(
    closed: _ClosedComposition,
    *,
    module_name: str,
) -> type[object]:
    """Construct the ordinary metadata class at the public closure boundary."""
    from bayeswire.ir import bindable_from_meta

    model_cls = bindable_from_meta(closed.meta, dimensions=closed.dimensions)
    target, source = closed.dependencies
    model_cls.__name__ = f"{target.__name__}With{source.__name__}Prior"
    model_cls.__qualname__ = model_cls.__name__
    model_cls.__module__ = module_name
    setattr(model_cls, "_model_dependencies", closed.dependencies)  # noqa: B010
    return model_cls
