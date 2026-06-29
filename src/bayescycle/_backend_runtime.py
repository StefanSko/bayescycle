"""Runtime backend resolution for first-party backend adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bayescycle._errors import WorkflowError
from bayescycle._workflow.capabilities import (
    BAYESITE,
    FIRST_PARTY_BACKENDS,
    BackendCapability,
    BackendId,
)
from bayescycle._workflow.protocols import PriorPredictiveBackend, SampleBackend
from bayescycle.backends.bayesite import BayesiteBackend
from bayescycle.backends.bayesite.preflight import (
    BayesiteCommandRequirement,
    preflight_bayesite_engine,
)
from bayescycle.backends.jaxstanv5 import Jaxstanv5Backend


@dataclass(frozen=True)
class BackendRuntimeOptions:
    """Concrete backend-runtime options resolved from CLI input."""

    backend: str
    engine: str | None
    extra_args: tuple[str, ...]
    preflight: bool


type OpaqueSampleBackend = SampleBackend[Any, Any]
type OpaquePriorPredictiveBackend = PriorPredictiveBackend[Any, Any]


def resolve_sample_backend(options: BackendRuntimeOptions) -> OpaqueSampleBackend:
    """Resolve a backend object with sample capability."""
    backend = _resolve_backend(options.backend, BackendCapability.SAMPLE)
    if backend == BAYESITE:
        return _bayesite_backend(options, command="sample", stage="sample")
    _reject_bayesite_options_for_non_bayesite(backend, options)
    return Jaxstanv5Backend()


def resolve_prior_predictive_backend(
    options: BackendRuntimeOptions,
) -> OpaquePriorPredictiveBackend:
    """Resolve a backend object with prior-predictive capability."""
    backend = _resolve_backend(options.backend, BackendCapability.PRIOR_PREDICTIVE)
    if backend == BAYESITE:
        return _bayesite_backend(
            options,
            command="prior-predictive",
            stage="prior-predictive",
        )
    _reject_bayesite_options_for_non_bayesite(backend, options)
    return Jaxstanv5Backend()


def _resolve_backend(value: str, capability: BackendCapability) -> BackendId:
    backend = FIRST_PARTY_BACKENDS.resolve(value)
    FIRST_PARTY_BACKENDS.require(backend, capability)
    return backend


def _bayesite_backend(
    options: BackendRuntimeOptions,
    *,
    command: str,
    stage: str,
) -> BayesiteBackend:
    engine = options.engine or str(BAYESITE)
    if options.preflight:
        info = preflight_bayesite_engine(
            engine,
            (BayesiteCommandRequirement(command, stage),),
        )
        engine = str(info.executable)
    return BayesiteBackend(engine, extra_args=options.extra_args)


def _reject_bayesite_options_for_non_bayesite(
    backend: BackendId,
    options: BackendRuntimeOptions,
) -> None:
    if options.engine is not None:
        raise WorkflowError(
            "--engine configures the bayesite backend, but no bayesite backend stage was selected. "
            "Did you mean --backend bayesite?"
        )
    if options.extra_args:
        raise WorkflowError("engine passthrough after -- is only supported for --backend bayesite")
