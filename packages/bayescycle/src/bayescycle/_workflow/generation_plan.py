"""Closed immutable functional generation plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GenerationPlanError(ValueError):
    """Raised when a functional generation plan is invalid."""


class FitAssociation(StrEnum):
    """Authority carried by a posterior fit artifact."""

    RUNTIME = "runtime"
    PORTABLE = "portable"


@dataclass(frozen=True)
class AuthoredProvenance:
    claimed_source_model_hash: str
    claimed_outcome_model_hash: str


@dataclass(frozen=True)
class Fixed:
    parameters_bytes: bytes


@dataclass(frozen=True)
class ModelPrior:
    model_ir_bytes: bytes
    authored_provenance: AuthoredProvenance | None = None


@dataclass(frozen=True)
class FitArtifact:
    model_ir_bytes: bytes
    data_bytes: bytes
    posterior_bytes: bytes
    association: FitAssociation


@dataclass(frozen=True)
class PosteriorOf:
    fit_artifact: FitArtifact


type ParameterSource = Fixed | ModelPrior | PosteriorOf


@dataclass(frozen=True)
class OutcomesOf:
    model_ir_bytes: bytes
    design_bytes: bytes


@dataclass(frozen=True)
class JointPredict:
    parameters: ParameterSource
    outcomes: OutcomesOf


@dataclass(frozen=True)
class Draw:
    distribution: JointPredict
    count: int
    seed: int


@dataclass(frozen=True)
class GenerationPlanDocument:
    bytes: bytes
    identity_hash: str
    invalidation_key: str


def generate_datasets(
    model_ir: bytes,
    *,
    design: bytes,
    parameter_source: ParameterSource,
    count: int = 100,
    seed: int = 0,
) -> Draw:
    """Build the user-facing dataset generation plan."""
    raise NotImplementedError("functional generation plans are not implemented")


def serialize_generation_plan(plan: Draw) -> bytes:
    """Serialize hash-only versioned generation provenance."""
    raise NotImplementedError("functional generation plan serialization is not implemented")


def parse_generation_plan_document(data: bytes) -> GenerationPlanDocument:
    """Parse strict hash-only generation provenance."""
    raise NotImplementedError("functional generation plan parsing is not implemented")


def generation_plan_identity(plan: Draw) -> str:
    """Return the deterministic serialized-plan identity."""
    raise NotImplementedError("functional generation plan identity is not implemented")


def generation_invalidation_key(plan: Draw) -> str:
    """Return the deterministic generation dependency key."""
    raise NotImplementedError("functional generation invalidation is not implemented")
