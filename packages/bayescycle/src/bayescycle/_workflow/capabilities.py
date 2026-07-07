"""Backend identity and workflow capability declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bayescycle._errors import WorkflowError


class BackendCapability(StrEnum):
    """Closed set of bayescycle workflow capabilities."""

    SAMPLE = "sample"
    PRIOR_PREDICTIVE = "prior-predictive"
    SIMULATE = "simulate"
    RECOVER = "recover"
    SBC = "sbc"
    DIAGNOSE = "diagnose"
    POSTERIOR_PREDICTIVE = "posterior-predictive"
    POSTERIOR_CHECK = "posterior-check"
    RECOVER_CHECK = "recover-check"


@dataclass(frozen=True, slots=True)
class BackendId:
    """Open backend identity value resolved through a catalog."""

    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("backend id must not be empty")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class BackendProvider:
    """One backend provider declaration in a catalog snapshot."""

    id: BackendId
    capabilities: frozenset[BackendCapability]


@dataclass(frozen=True)
class BackendCatalog:
    """Immutable backend catalog used to resolve CLI/backend-plan input."""

    providers: tuple[BackendProvider, ...]

    def choices(self) -> tuple[str, ...]:
        """Return all known backend IDs in catalog order."""
        return tuple(str(provider.id) for provider in self.providers)

    def choices_for(self, capability: BackendCapability) -> tuple[str, ...]:
        """Return backend IDs supporting a bayescycle workflow capability."""
        return tuple(
            str(provider.id) for provider in self.providers if capability in provider.capabilities
        )

    def resolve(self, value: str, *, label: str = "--backend") -> BackendId:
        """Resolve a loose backend string into a backend identity value."""
        for provider in self.providers:
            if provider.id.value == value:
                return provider.id
        choices = ", ".join(self.choices())
        raise WorkflowError(f"{label} must be one of: {choices}")

    def require(self, backend: BackendId, capability: BackendCapability) -> None:
        """Require that a known backend supports a workflow capability."""
        provider = self._provider(backend)
        if capability not in provider.capabilities:
            raise WorkflowError(
                f"backend {backend} does not support bayescycle capability {capability.value}"
            )

    def supports(self, backend: BackendId, capability: BackendCapability) -> bool:
        """Return whether a known backend supports a workflow capability."""
        return capability in self._provider(backend).capabilities

    def _provider(self, backend: BackendId) -> BackendProvider:
        for provider in self.providers:
            if provider.id == backend:
                return provider
        choices = ", ".join(self.choices())
        raise WorkflowError(f"unknown backend {backend}; expected one of: {choices}")


BAYESITE = BackendId("bayesite")
BAYESJAX = BackendId("bayesjax")

FIRST_PARTY_BACKENDS = BackendCatalog(
    providers=(
        BackendProvider(
            id=BAYESITE,
            capabilities=frozenset(
                {
                    BackendCapability.SAMPLE,
                    BackendCapability.PRIOR_PREDICTIVE,
                    BackendCapability.SIMULATE,
                    BackendCapability.RECOVER,
                    BackendCapability.SBC,
                    BackendCapability.DIAGNOSE,
                    BackendCapability.POSTERIOR_PREDICTIVE,
                    BackendCapability.POSTERIOR_CHECK,
                    BackendCapability.RECOVER_CHECK,
                }
            ),
        ),
        BackendProvider(
            id=BAYESJAX,
            capabilities=frozenset(
                {
                    BackendCapability.SAMPLE,
                    BackendCapability.PRIOR_PREDICTIVE,
                }
            ),
        ),
    )
)
