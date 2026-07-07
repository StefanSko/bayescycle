"""Typed plan descriptions for backend integration modes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

type IntegrationMode = Literal["external-command", "in-process-python"]
type JsonNumber = int | float


class BackendPlanFields(TypedDict, total=False):
    """JSON-ready backend-owned plan fields."""

    backend: str
    integration_mode: IntegrationMode
    command: list[str]
    backend_simulated_data: str
    sampler: dict[str, JsonNumber]
    settings: dict[str, JsonNumber]


@dataclass(frozen=True)
class ExternalCommandPlanDescription:
    """Backend plan description for a Unix-style external command."""

    backend: str
    command: tuple[str, ...]
    backend_simulated_data: Path | None = None
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class InProcessSamplePlanDescription:
    """Backend plan description for in-process sampling."""

    backend: str
    sampler: Mapping[str, JsonNumber]


@dataclass(frozen=True)
class InProcessSettingsPlanDescription:
    """Backend plan description for in-process non-sampling settings."""

    backend: str
    settings: Mapping[str, JsonNumber]


type BackendPlanDescription = (
    ExternalCommandPlanDescription
    | InProcessSamplePlanDescription
    | InProcessSettingsPlanDescription
)


def backend_replay_extra_args(description: BackendPlanDescription) -> tuple[str, ...]:
    """Return backend passthrough args that must be preserved for replay."""
    match description:
        case ExternalCommandPlanDescription(extra_args=extra_args):
            return extra_args
        case InProcessSamplePlanDescription() | InProcessSettingsPlanDescription():
            return ()


def backend_plan_description_fields(
    description: BackendPlanDescription,
) -> BackendPlanFields:
    """Lower a typed backend plan description to JSON-ready plan fields."""
    match description:
        case ExternalCommandPlanDescription(
            backend=backend,
            command=command,
            backend_simulated_data=backend_simulated_data,
        ):
            fields: BackendPlanFields = {
                "backend": backend,
                "integration_mode": "external-command",
                "command": list(command),
            }
            if backend_simulated_data is not None:
                fields["backend_simulated_data"] = str(backend_simulated_data)
            return fields
        case InProcessSamplePlanDescription(backend=backend, sampler=sampler):
            return {
                "backend": backend,
                "integration_mode": "in-process-python",
                "sampler": dict(sampler),
            }
        case InProcessSettingsPlanDescription(backend=backend, settings=settings):
            return {
                "backend": backend,
                "integration_mode": "in-process-python",
                "settings": dict(settings),
            }
