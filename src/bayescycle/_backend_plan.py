"""Backend-plan resolution for multi-stage workflow intent."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from bayescycle._errors import WorkflowError

BackendId = Literal["bayesite", "jaxstanv5"]
StageId = Literal["simulate", "recover"]

STAGES: tuple[StageId, ...] = ("simulate", "recover")
BACKENDS: tuple[BackendId, ...] = ("bayesite", "jaxstanv5")


@dataclass(frozen=True)
class StageBackend:
    """One explicit stage-to-backend assignment."""

    stage: StageId
    backend: BackendId


@dataclass(frozen=True)
class SingleBackendPlan:
    """One backend selected for all stages."""

    backend: BackendId


@dataclass(frozen=True)
class ExplicitMixedBackendPlan:
    """Complete explicit stage-to-backend assignment for a mixed workflow."""

    stage_backends: tuple[StageBackend, ...]


BackendPlan = SingleBackendPlan | ExplicitMixedBackendPlan


@dataclass(frozen=True)
class BackendPlanRequest:
    """Loose CLI/backend-plan input normalized before run-directory writes."""

    backend: str | None
    simulate_backend: str | None
    recover_backend: str | None
    engine: str | None


@dataclass(frozen=True)
class ResolvedBackendPlan:
    """Validated backend plan with concrete stage assignments."""

    plan: BackendPlan
    stage_backends: tuple[StageBackend, ...]
    bayesite_engine: str | None

    def as_json(self) -> dict[str, object]:
        """Return a stable JSON summary for dry planning/docs."""
        return {
            "mode": "single" if isinstance(self.plan, SingleBackendPlan) else "mixed",
            "stages": {entry.stage: entry.backend for entry in self.stage_backends},
            "backends": _backend_options_json(self.bayesite_engine),
        }


def resolve_backend_plan(request: BackendPlanRequest) -> ResolvedBackendPlan:
    """Resolve and validate a multi-stage backend plan.

    Missing stage-specific assignments are never filled in to create a mixed
    plan. Use ``--backend`` for a single-backend workflow or provide every
    stage backend explicitly for mixed intent.
    """
    overrides = {
        "simulate": request.simulate_backend,
        "recover": request.recover_backend,
    }
    provided = {stage: backend for stage, backend in overrides.items() if backend is not None}
    if not provided:
        backend = _parse_backend(request.backend or "bayesite", "--backend")
        plan: BackendPlan = SingleBackendPlan(backend=backend)
        stage_backends = tuple(StageBackend(stage=stage, backend=backend) for stage in STAGES)
    else:
        if request.backend is not None:
            raise WorkflowError(
                "Use either --backend for a single-backend workflow or provide a complete "
                "explicit mixed-backend plan with stage-specific backend options, not both."
            )
        missing = tuple(stage for stage in STAGES if overrides[stage] is None)
        if missing:
            assigned_lines = _partial_assignment_lines(overrides)
            raise WorkflowError(
                "Partial backend assignment would create an implicit workflow plan.\n\n"
                f"{assigned_lines}\n\n"
                "Either use --backend for all stages, or provide an explicit complete "
                "mixed-backend plan."
            )
        stage_backends = tuple(
            StageBackend(
                stage=stage,
                backend=_parse_backend(cast(str, overrides[stage]), f"--{stage}-backend"),
            )
            for stage in STAGES
        )
        backends = {entry.backend for entry in stage_backends}
        if len(backends) == 1:
            plan = SingleBackendPlan(backend=stage_backends[0].backend)
        else:
            plan = ExplicitMixedBackendPlan(stage_backends=stage_backends)
    _validate_engine_selection(stage_backends, request.engine)
    _validate_stage_support(stage_backends)
    return ResolvedBackendPlan(
        plan=plan,
        stage_backends=stage_backends,
        bayesite_engine=request.engine if _uses_bayesite(stage_backends) else None,
    )


def resolve_backend_plan_file(path: Path) -> ResolvedBackendPlan:
    """Resolve a backend plan from the documented TOML config schema."""
    config_path = path.expanduser().resolve()
    try:
        with config_path.open("rb") as f:
            document = cast(dict[str, object], tomllib.load(f))
    except OSError as exc:
        raise WorkflowError(f"cannot read backend plan config {config_path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise WorkflowError(f"invalid backend plan config {config_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise WorkflowError(f"backend plan config {config_path} must be a TOML table")
    workflow = _table(document, "workflow", config_path)
    mode_value = workflow.get("mode", "single")
    if not isinstance(mode_value, str):
        raise WorkflowError("[workflow].mode must be a string")
    engine = _bayesite_engine_from_config(document)
    if mode_value == "single":
        backend_value = workflow.get("backend", "bayesite")
        if not isinstance(backend_value, str):
            raise WorkflowError("[workflow].backend must be a string")
        return resolve_backend_plan(
            BackendPlanRequest(
                backend=backend_value,
                simulate_backend=None,
                recover_backend=None,
                engine=engine,
            )
        )
    if mode_value == "mixed":
        stages = _table(document, "stages", config_path)
        simulate_backend = _stage_backend_from_config(stages, "simulate")
        recover_backend = _stage_backend_from_config(stages, "recover")
        return resolve_backend_plan(
            BackendPlanRequest(
                backend=None,
                simulate_backend=simulate_backend,
                recover_backend=recover_backend,
                engine=engine,
            )
        )
    raise WorkflowError("[workflow].mode must be 'single' or 'mixed'")


def _parse_backend(value: str, label: str) -> BackendId:
    if value == "bayesite" or value == "jaxstanv5":
        return cast(BackendId, value)
    raise WorkflowError(f"{label} must be 'bayesite' or 'jaxstanv5'")


def _partial_assignment_lines(overrides: dict[str, str | None]) -> str:
    lines: list[str] = []
    for stage in STAGES:
        value = overrides[stage]
        if value is None:
            lines.append(f"{stage}: inherited from default")
        else:
            lines.append(f"{stage}: {value}")
    return "\n".join(lines)


def _validate_stage_support(stage_backends: tuple[StageBackend, ...]) -> None:
    for entry in stage_backends:
        if entry.stage == "simulate" and entry.backend != "bayesite":
            raise WorkflowError(
                "backend jaxstanv5 does not support required stage simulate; "
                "use --backend bayesite or assign simulate to bayesite in an explicit mixed plan"
            )


def _validate_engine_selection(
    stage_backends: tuple[StageBackend, ...], engine: str | None
) -> None:
    if engine is not None and not _uses_bayesite(stage_backends):
        raise WorkflowError(
            "--engine configures the bayesite backend, but no bayesite backend stage was selected. "
            "Did you mean --backend bayesite?"
        )


def _uses_bayesite(stage_backends: tuple[StageBackend, ...]) -> bool:
    return any(entry.backend == "bayesite" for entry in stage_backends)


def _backend_options_json(engine: str | None) -> dict[str, object]:
    if engine is None:
        return {}
    return {"bayesite": {"engine": engine}}


def _table(document: dict[str, object], name: str, config_path: Path) -> dict[str, object]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise WorkflowError(f"backend plan config {config_path} needs a [{name}] table")
    return cast(dict[str, object], value)


def _stage_backend_from_config(stages: dict[str, object], stage: StageId) -> str | None:
    value = stages.get(stage)
    if not isinstance(value, dict):
        return None
    backend_value = value.get("backend")
    if backend_value is None:
        return None
    if not isinstance(backend_value, str):
        raise WorkflowError(f"[stages.{stage}].backend must be a string")
    return backend_value


def _bayesite_engine_from_config(document: dict[str, object]) -> str | None:
    backends = document.get("backends")
    if backends is None:
        return None
    if not isinstance(backends, dict):
        raise WorkflowError("[backends] must be a table when present")
    backends_table = cast(dict[str, object], backends)
    bayesite = backends_table.get("bayesite")
    if bayesite is None:
        return None
    if not isinstance(bayesite, dict):
        raise WorkflowError("[backends.bayesite] must be a table when present")
    bayesite_table = cast(dict[str, object], bayesite)
    engine = bayesite_table.get("engine")
    if engine is None:
        return None
    if not isinstance(engine, str):
        raise WorkflowError("[backends.bayesite].engine must be a string")
    return engine
