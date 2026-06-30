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
from bayescycle._workflow.protocols import (
    DiagnoseBackend,
    PosteriorCheckBackend,
    PosteriorPredictiveBackend,
    PriorPredictiveBackend,
    RecoverBackend,
    RecoverCheckBackend,
    SampleBackend,
    SbcBackend,
    SimulateBackend,
)
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
type OpaqueSimulateBackend = SimulateBackend[Any, Any]
type OpaqueRecoverBackend = RecoverBackend[Any, Any]
type OpaqueSbcBackend = SbcBackend[Any, Any]
type OpaqueDiagnoseBackend = DiagnoseBackend[Any, Any]
type OpaquePosteriorPredictiveBackend = PosteriorPredictiveBackend[Any, Any]
type OpaquePosteriorCheckBackend = PosteriorCheckBackend[Any, Any]
type OpaqueRecoverCheckBackend = RecoverCheckBackend[Any, Any]


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


def resolve_simulate_backend(options: BackendRuntimeOptions) -> OpaqueSimulateBackend:
    """Resolve a backend object with simulate capability."""
    backend = _resolve_backend(options.backend, BackendCapability.SIMULATE)
    if backend == BAYESITE:
        return _bayesite_backend(options, command="simulate", stage="simulate")
    _reject_bayesite_options_for_non_bayesite(backend, options)
    raise _unsupported_runtime_backend(backend, BackendCapability.SIMULATE)


def resolve_recover_backend(options: BackendRuntimeOptions) -> OpaqueRecoverBackend:
    """Resolve a backend object with recover capability."""
    backend = _resolve_backend(options.backend, BackendCapability.RECOVER)
    if backend == BAYESITE:
        return _bayesite_backend(options, command="recover", stage="recover")
    _reject_bayesite_options_for_non_bayesite(backend, options)
    raise _unsupported_runtime_backend(backend, BackendCapability.RECOVER)


def resolve_sbc_backend(options: BackendRuntimeOptions) -> OpaqueSbcBackend:
    """Resolve a backend object with SBC capability."""
    backend = _resolve_backend(options.backend, BackendCapability.SBC)
    if backend == BAYESITE:
        return _bayesite_backend(options, command="sbc", stage="sbc")
    _reject_bayesite_options_for_non_bayesite(backend, options)
    raise _unsupported_runtime_backend(backend, BackendCapability.SBC)


def resolve_diagnose_backend(options: BackendRuntimeOptions) -> OpaqueDiagnoseBackend:
    """Resolve a backend object with diagnose capability."""
    backend = _resolve_backend(options.backend, BackendCapability.DIAGNOSE)
    if backend == BAYESITE:
        return _bayesite_backend(options, command="diagnose", stage="diagnose")
    _reject_bayesite_options_for_non_bayesite(backend, options)
    raise _unsupported_runtime_backend(backend, BackendCapability.DIAGNOSE)


def resolve_posterior_predictive_backend(
    options: BackendRuntimeOptions,
) -> OpaquePosteriorPredictiveBackend:
    """Resolve a backend object with posterior-predictive capability."""
    backend = _resolve_backend(options.backend, BackendCapability.POSTERIOR_PREDICTIVE)
    if backend == BAYESITE:
        return _bayesite_backend(
            options,
            command="posterior-predictive",
            stage="posterior-predictive",
        )
    _reject_bayesite_options_for_non_bayesite(backend, options)
    raise _unsupported_runtime_backend(backend, BackendCapability.POSTERIOR_PREDICTIVE)


def resolve_posterior_check_backend(
    options: BackendRuntimeOptions,
) -> OpaquePosteriorCheckBackend:
    """Resolve a backend object with posterior-check capability."""
    backend = _resolve_backend(options.backend, BackendCapability.POSTERIOR_CHECK)
    if backend == BAYESITE:
        return _bayesite_backend(options, command="posterior-check", stage="posterior-check")
    _reject_bayesite_options_for_non_bayesite(backend, options)
    raise _unsupported_runtime_backend(backend, BackendCapability.POSTERIOR_CHECK)


def resolve_recover_check_backend(options: BackendRuntimeOptions) -> OpaqueRecoverCheckBackend:
    """Resolve a backend object with recover-check capability."""
    backend = _resolve_backend(options.backend, BackendCapability.RECOVER_CHECK)
    if backend == BAYESITE:
        return _bayesite_backend(options, command="recover-check", stage="recover-check")
    _reject_bayesite_options_for_non_bayesite(backend, options)
    raise _unsupported_runtime_backend(backend, BackendCapability.RECOVER_CHECK)


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


def _unsupported_runtime_backend(
    backend: BackendId,
    capability: BackendCapability,
) -> WorkflowError:
    return WorkflowError(
        f"backend {backend} passed capability validation but has no runtime adapter for "
        f"bayescycle capability {capability.value}"
    )
