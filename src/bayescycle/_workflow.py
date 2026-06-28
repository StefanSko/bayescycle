"""Workflow phases for planning, materializing, and launching backend runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, cast

from jaxstanv5.ir import canonical_bytes
from jaxstanv5.model import ModelMeta

from bayescycle._artifacts import CanonicalDataArtifact, IrArtifact
from bayescycle._dims import dims_sidecar_for_model, write_dims_sidecar
from bayescycle._errors import WorkflowError
from bayescycle._model_loader import LoadedModel, load_model
from bayescycle._settings import SamplerSettings
from bayescycle.data import DATA_DOC_FORMAT, DataDoc, DataDocError, read_data_doc, write_data_doc


@dataclass(frozen=True)
class SampleRequest:
    """Loose CLI sampling input normalized into a typed request."""

    model_path: Path
    data_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    sampler: SamplerSettings
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class PriorPredictiveRequest:
    """Loose CLI prior-predictive input normalized into a typed request."""

    model_path: Path
    data_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    seed: str | None
    draws: str | None
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class SimulateRequest:
    """Loose CLI simulation input normalized into a typed request."""

    model_path: Path
    data_path: Path
    truth_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    seed: str | None
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class RecoverRequest:
    """Loose CLI single-scenario recovery input normalized into a typed request."""

    model_path: Path
    scenario_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class SbcRequest:
    """Loose CLI SBC input normalized into a typed request."""

    model_path: Path
    scenario_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    replicates: str | None
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class DiagnoseRequest:
    """Request to run diagnostics for an existing run directory."""

    run_dir: Path


@dataclass(frozen=True)
class PosteriorPredictiveRequest:
    """Request to run posterior predictive generation for an existing run directory."""

    run_dir: Path
    seed: str


@dataclass(frozen=True)
class PosteriorCheckRequest:
    """Request to run posterior checks for an existing run directory."""

    run_dir: Path
    seed: str | None
    backend: str
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class RecoverCheckRequest:
    """Request to run recovery checks for an existing run directory."""

    run_dir: Path
    truth_path: Path
    targets_path: Path | None
    interval: str | None
    backend: str
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class PlannedModelRunContext:
    """Planned model/data run-directory inputs before durable writes."""

    model_name: str
    model_source_path: Path
    model_source_sha256: str
    loaded_model: LoadedModel
    ir_path: IrArtifact
    data_source_path: Path
    data_source_sha256: str
    data_path: CanonicalDataArtifact
    dims_path: Path | None
    output_dir: Path
    data_doc: DataDoc


@dataclass(frozen=True)
class PlannedModelScenarioContext:
    """Planned model/scenario run-directory inputs before durable writes."""

    model_name: str
    model_source_path: Path
    model_source_sha256: str
    loaded_model: LoadedModel
    ir_path: IrArtifact
    scenario_source_path: Path
    scenario_source_sha256: str
    scenario_path: Path
    dims_path: Path | None
    output_dir: Path


@dataclass(frozen=True)
class DiagnoseRunContext:
    """Validated existing-run paths for a diagnose command."""

    run_dir: Path
    fit_path: Path
    output_path: Path


@dataclass(frozen=True)
class PosteriorPredictiveRunContext:
    """Validated existing-run paths for posterior predictive generation."""

    run_dir: Path
    model_path: IrArtifact
    data_path: CanonicalDataArtifact
    fit_path: Path
    output_path: Path


@dataclass(frozen=True)
class PosteriorCheckRunContext:
    """Validated existing-run paths for posterior checks."""

    run_dir: Path
    model_path: IrArtifact
    data_path: CanonicalDataArtifact
    fit_path: Path
    output_path: Path


@dataclass(frozen=True)
class RecoverCheckRunContext:
    """Validated existing-run paths for recovery checks."""

    run_dir: Path
    fit_path: Path
    truth_path: Path
    targets_path: Path | None
    output_path: Path


type JsonNumber = int | float


class BackendPlanFields(TypedDict, total=False):
    """JSON-ready backend-owned plan fields."""

    engine_command: list[str]
    backend_simulated_data: str
    backend: str
    sampler: dict[str, JsonNumber]
    settings: dict[str, JsonNumber]


@dataclass(frozen=True)
class EngineBackendPlanDescription:
    """Backend plan description for an external engine command."""

    engine_command: tuple[str, ...]
    backend_simulated_data: Path | None = None


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
    EngineBackendPlanDescription | InProcessSamplePlanDescription | InProcessSettingsPlanDescription
)


def backend_plan_description_fields(
    description: BackendPlanDescription,
) -> BackendPlanFields:
    """Lower a typed backend plan description to JSON-ready plan fields."""
    match description:
        case EngineBackendPlanDescription(
            engine_command=engine_command,
            backend_simulated_data=backend_simulated_data,
        ):
            fields: BackendPlanFields = {"engine_command": list(engine_command)}
            if backend_simulated_data is not None:
                fields["backend_simulated_data"] = str(backend_simulated_data)
            return fields
        case InProcessSamplePlanDescription(backend=backend, sampler=sampler):
            return {"backend": backend, "sampler": dict(sampler)}
        case InProcessSettingsPlanDescription(backend=backend, settings=settings):
            return {"backend": backend, "settings": dict(settings)}


class BackendExecutor[CommandT](Protocol):
    """Backend capability for executing a materialized command."""

    def execute(self, command: CommandT) -> int:
        """Execute a typed backend command."""


class ActionBackend[ActionT, CommandT](BackendExecutor[CommandT], Protocol):
    """Backend capability shared by planned backend actions."""

    def describe(self, action: ActionT) -> BackendPlanDescription:
        """Return backend-owned plan description for an action."""

    def materialize(self, action: ActionT) -> CommandT:
        """Materialize backend-private inputs and return an executable command."""


class SampleBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for the sample workflow operation."""

    def plan_sample_action(
        self, context: PlannedModelRunContext, request: SampleRequest
    ) -> ActionT:
        """Plan a backend-private sample action from a planned run context."""


class PriorPredictiveBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for the prior-predictive workflow operation."""

    def plan_prior_predictive_action(
        self, context: PlannedModelRunContext, request: PriorPredictiveRequest
    ) -> ActionT:
        """Plan a backend-private prior-predictive action."""


class SimulateBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for the simulate workflow operation."""

    def plan_simulate_action(
        self, context: PlannedModelRunContext, request: SimulateRequest, truth_path: Path
    ) -> ActionT:
        """Plan a backend-private simulate action from a planned run context."""


class RecoverBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for the recover workflow operation."""

    def plan_recover_action(
        self, context: PlannedModelScenarioContext, request: RecoverRequest
    ) -> ActionT:
        """Plan a backend-private recover action from a planned scenario context."""


class SbcBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for the SBC workflow operation."""

    def plan_sbc_action(self, context: PlannedModelScenarioContext, request: SbcRequest) -> ActionT:
        """Plan a backend-private SBC action from a planned scenario context."""


class DiagnoseBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for an existing-run diagnose command."""

    def plan_diagnose_action(
        self, context: DiagnoseRunContext, request: DiagnoseRequest
    ) -> ActionT:
        """Plan a backend-private diagnose action."""


class PosteriorPredictiveBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for an existing-run posterior-predictive command."""

    def plan_posterior_predictive_action(
        self, context: PosteriorPredictiveRunContext, request: PosteriorPredictiveRequest
    ) -> ActionT:
        """Plan a backend-private posterior-predictive action."""


class PosteriorCheckBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for an existing-run posterior-check command."""

    def plan_posterior_check_action(
        self, context: PosteriorCheckRunContext, request: PosteriorCheckRequest
    ) -> ActionT:
        """Plan a backend-private posterior-check action."""


class RecoverCheckBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for an existing-run recover-check command."""

    def plan_recover_check_action(
        self, context: RecoverCheckRunContext, request: RecoverCheckRequest
    ) -> ActionT:
        """Plan a backend-private recover-check action."""


@dataclass(frozen=True)
class SampleRunPlan[ActionT]:
    """A sample run plan with workflow-owned paths and backend action."""

    context: PlannedModelRunContext
    draws_path: Path
    backend: str
    action: ActionT


@dataclass(frozen=True)
class PriorPredictiveRunPlan[ActionT]:
    """A prior-predictive run plan with workflow-owned paths and backend action."""

    context: PlannedModelRunContext
    prior_predictive_path: Path
    backend: str
    action: ActionT


@dataclass(frozen=True)
class SimulateRunPlan[ActionT]:
    """A simulation run plan with workflow-owned paths and backend action."""

    context: PlannedModelRunContext
    truth_source_path: Path
    truth_path: Path
    simulated_data_path: CanonicalDataArtifact
    backend: str
    action: ActionT


@dataclass(frozen=True)
class RecoverRunPlan[ActionT]:
    """A single-scenario recovery run plan with workflow-owned paths and action."""

    context: PlannedModelScenarioContext
    recovery_path: Path
    backend: str
    action: ActionT


@dataclass(frozen=True)
class SbcRunPlan[ActionT]:
    """An SBC run plan with workflow-owned paths and backend action."""

    context: PlannedModelScenarioContext
    sbc_path: Path
    backend: str
    action: ActionT


@dataclass(frozen=True)
class RunDirectoryCommandPlan[ActionT]:
    """An existing-run command plan with backend action."""

    run_dir: Path
    output_path: Path
    action: ActionT


@dataclass(frozen=True)
class RunMetadataModel:
    """Model source recorded in a run metadata document."""

    name: str
    source_path: Path
    source_sha256: str
    ir_path: Path

    def as_json(self, run_dir: Path) -> dict[str, object]:
        return {
            "name": self.name,
            "source_path": str(self.source_path),
            "sha256": self.source_sha256,
            "ir_path": _relative_run_path(run_dir, self.ir_path),
        }


@dataclass(frozen=True)
class RunMetadataInput:
    """Input source recorded in a run metadata document."""

    role: str
    source_path: Path
    source_sha256: str
    materialized_path: Path
    artifact_format: str | None = None

    def as_json(self, run_dir: Path) -> dict[str, object]:
        document: dict[str, object] = {
            "role": self.role,
            "source_path": str(self.source_path),
            "sha256": self.source_sha256,
            "path": _relative_run_path(run_dir, self.materialized_path),
        }
        if self.artifact_format is not None:
            document["format"] = self.artifact_format
        return document


@dataclass(frozen=True)
class RunMetadataOutput:
    """Declared run output recorded in a run metadata document."""

    role: str
    path: Path
    artifact_format: str | None = None

    def as_json(self, run_dir: Path) -> dict[str, object]:
        document: dict[str, object] = {
            "role": self.role,
            "path": _relative_run_path(run_dir, self.path),
        }
        if self.artifact_format is not None:
            document["format"] = self.artifact_format
        return document


@dataclass(frozen=True)
class RunMetadata:
    """Append-only metadata for a prepared run directory."""

    kind: str
    backend: str
    model: RunMetadataModel
    inputs: tuple[RunMetadataInput, ...]
    outputs: tuple[RunMetadataOutput, ...]

    def as_json(self, run_dir: Path) -> dict[str, object]:
        return {
            "format": "bayescycle.run.v1",
            "kind": self.kind,
            "backend": self.backend,
            "model": self.model.as_json(run_dir),
            "inputs": [entry.as_json(run_dir) for entry in self.inputs],
            "outputs": [entry.as_json(run_dir) for entry in self.outputs],
        }


class SamplePlanDocument(TypedDict):
    """JSON document printed for a sample plan."""

    model: str
    ir: str
    data: str
    draws: str
    output: str
    engine_command: NotRequired[list[str]]
    backend: NotRequired[str]
    sampler: NotRequired[dict[str, JsonNumber]]
    dims: NotRequired[str]


class ModelCommandPlanDocument(TypedDict):
    """JSON document printed for a model-level command plan."""

    model: str
    ir: str
    output: str
    data: NotRequired[str]
    truth: NotRequired[str]
    scenario: NotRequired[str]
    prior_predictive: NotRequired[str]
    simulated_data: NotRequired[str]
    recovery: NotRequired[str]
    sbc: NotRequired[str]
    engine_command: NotRequired[list[str]]
    backend_simulated_data: NotRequired[str]
    backend: NotRequired[str]
    settings: NotRequired[dict[str, JsonNumber]]
    dims: NotRequired[str]


class RunCommandPlanDocument(TypedDict):
    """JSON document printed for an existing-run command plan."""

    run: str
    output: str
    engine_command: list[str]


def plan_sample_run[ActionT, CommandT](
    request: SampleRequest, backend: SampleBackend[ActionT, CommandT]
) -> SampleRunPlan[ActionT]:
    """Plan a sample run without durable filesystem writes."""
    _validate_backend(request.backend)
    output_dir = request.output_dir.expanduser().resolve()
    draws_path = output_dir / "posterior.ndjson"
    _reject_reserved_engine_args(request.engine_args, draws_path)
    _reject_in_process_engine_args(request.backend, request.engine_args)
    context = plan_model_run_context(
        model_path=request.model_path,
        model_name=request.model_name,
        data_path=request.data_path,
        output_dir=output_dir,
    )
    action = backend.plan_sample_action(context, request)
    return SampleRunPlan(
        context=context,
        draws_path=draws_path,
        backend=request.backend,
        action=action,
    )


def materialize_sample_run[ActionT, CommandT](
    plan: SampleRunPlan[ActionT], backend: SampleBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned sample run for execution."""
    context = materialize_model_run_context(plan.context)
    _write_run_metadata(
        context.output_dir,
        _model_data_run_metadata(
            context,
            kind="sample",
            backend=plan.backend,
            outputs=(RunMetadataOutput(role="posterior", path=plan.draws_path),),
        ),
    )
    return backend.materialize(plan.action)


def plan_prior_predictive_run[ActionT, CommandT](
    request: PriorPredictiveRequest, backend: PriorPredictiveBackend[ActionT, CommandT]
) -> PriorPredictiveRunPlan[ActionT]:
    """Plan a prior-predictive run without durable filesystem writes."""
    _validate_backend(request.backend)
    output_dir = request.output_dir.expanduser().resolve()
    output_path = output_dir / "prior_predictive.ndjson"
    _reject_reserved_engine_args(request.engine_args, output_path)
    _reject_in_process_engine_args(request.backend, request.engine_args)
    context = plan_model_run_context(
        model_path=request.model_path,
        model_name=request.model_name,
        data_path=request.data_path,
        output_dir=output_dir,
    )
    action = backend.plan_prior_predictive_action(context, request)
    return PriorPredictiveRunPlan(
        context=context,
        prior_predictive_path=output_path,
        backend=request.backend,
        action=action,
    )


def materialize_prior_predictive_run[ActionT, CommandT](
    plan: PriorPredictiveRunPlan[ActionT], backend: PriorPredictiveBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned prior-predictive run for execution."""
    context = materialize_model_run_context(plan.context)
    _write_run_metadata(
        context.output_dir,
        _model_data_run_metadata(
            context,
            kind="prior-predictive",
            backend=plan.backend,
            outputs=(RunMetadataOutput(role="prior_predictive", path=plan.prior_predictive_path),),
        ),
    )
    return backend.materialize(plan.action)


def plan_simulate_run[ActionT, CommandT](
    request: SimulateRequest, backend: SimulateBackend[ActionT, CommandT]
) -> SimulateRunPlan[ActionT]:
    """Plan a simulation run without durable filesystem writes."""
    _validate_bayesite_only(request.backend, "simulate")
    output_dir = request.output_dir.expanduser().resolve()
    output_path = CanonicalDataArtifact(output_dir / "simulated_data.json")
    _reject_reserved_engine_args(request.engine_args, output_path.path)
    truth_source = _require_input_file(request.truth_path, "truth")
    context = plan_model_run_context(
        model_path=request.model_path,
        model_name=request.model_name,
        data_path=request.data_path,
        output_dir=output_dir,
    )
    truth_path = output_dir / "truth.json"
    action = backend.plan_simulate_action(context, request, truth_path)
    return SimulateRunPlan(
        context=context,
        truth_source_path=truth_source,
        truth_path=truth_path,
        simulated_data_path=output_path,
        backend=request.backend,
        action=action,
    )


def materialize_simulate_run[ActionT, CommandT](
    plan: SimulateRunPlan[ActionT], backend: SimulateBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned simulation run for execution."""
    context = materialize_model_run_context(
        plan.context, additional_data_paths=(plan.simulated_data_path,)
    )
    _copy_required_input(plan.truth_source_path, plan.truth_path, "truth")
    truth_sha256 = _sha256_uri(plan.truth_path)
    _write_run_metadata(
        context.output_dir,
        _model_data_run_metadata(
            context,
            kind="simulate",
            backend=plan.backend,
            extra_inputs=(
                RunMetadataInput(
                    role="truth",
                    source_path=plan.truth_source_path,
                    source_sha256=truth_sha256,
                    materialized_path=plan.truth_path,
                ),
            ),
            outputs=(
                RunMetadataOutput(
                    role="simulated_data",
                    path=plan.simulated_data_path.path,
                    artifact_format=DATA_DOC_FORMAT,
                ),
            ),
        ),
    )
    return backend.materialize(plan.action)


def plan_recover_run[ActionT, CommandT](
    request: RecoverRequest, backend: RecoverBackend[ActionT, CommandT]
) -> RecoverRunPlan[ActionT]:
    """Plan a single-scenario recovery run without durable filesystem writes."""
    _validate_bayesite_only(request.backend, "recover")
    output_dir = request.output_dir.expanduser().resolve()
    output_path = output_dir / "recovery.json"
    _reject_reserved_engine_args(request.engine_args, output_path)
    context = plan_model_scenario_context(
        model_path=request.model_path,
        model_name=request.model_name,
        scenario_path=request.scenario_path,
        output_dir=output_dir,
    )
    action = backend.plan_recover_action(context, request)
    return RecoverRunPlan(
        context=context,
        recovery_path=output_path,
        backend=request.backend,
        action=action,
    )


def materialize_recover_run[ActionT, CommandT](
    plan: RecoverRunPlan[ActionT], backend: RecoverBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned single-scenario recovery run for execution."""
    context = materialize_model_scenario_context(plan.context)
    _write_run_metadata(
        context.output_dir,
        _model_scenario_run_metadata(
            context,
            kind="recover",
            backend=plan.backend,
            outputs=(RunMetadataOutput(role="recovery", path=plan.recovery_path),),
        ),
    )
    return backend.materialize(plan.action)


def plan_sbc_run[ActionT, CommandT](
    request: SbcRequest, backend: SbcBackend[ActionT, CommandT]
) -> SbcRunPlan[ActionT]:
    """Plan an SBC run without durable filesystem writes."""
    _validate_bayesite_only(request.backend, "sbc")
    output_dir = request.output_dir.expanduser().resolve()
    output_path = output_dir / "sbc.json"
    _reject_reserved_engine_args(request.engine_args, output_path)
    context = plan_model_scenario_context(
        model_path=request.model_path,
        model_name=request.model_name,
        scenario_path=request.scenario_path,
        output_dir=output_dir,
    )
    action = backend.plan_sbc_action(context, request)
    return SbcRunPlan(
        context=context,
        sbc_path=output_path,
        backend=request.backend,
        action=action,
    )


def materialize_sbc_run[ActionT, CommandT](
    plan: SbcRunPlan[ActionT], backend: SbcBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned SBC run for execution."""
    context = materialize_model_scenario_context(plan.context)
    _write_run_metadata(
        context.output_dir,
        _model_scenario_run_metadata(
            context,
            kind="sbc",
            backend=plan.backend,
            outputs=(RunMetadataOutput(role="sbc", path=plan.sbc_path),),
        ),
    )
    return backend.materialize(plan.action)


def plan_diagnose_run[ActionT, CommandT](
    request: DiagnoseRequest, backend: DiagnoseBackend[ActionT, CommandT]
) -> RunDirectoryCommandPlan[ActionT]:
    """Plan a Bayesite diagnose command from an existing run directory."""
    run_dir = request.run_dir.expanduser().resolve()
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "diagnostics.json"
    _require_run_artifacts((fit_path,))
    _reject_existing_output_artifacts((output_path,))
    context = DiagnoseRunContext(run_dir=run_dir, fit_path=fit_path, output_path=output_path)
    action = backend.plan_diagnose_action(context, request)
    return RunDirectoryCommandPlan(run_dir=run_dir, output_path=output_path, action=action)


def plan_posterior_predictive_run[ActionT, CommandT](
    request: PosteriorPredictiveRequest,
    backend: PosteriorPredictiveBackend[ActionT, CommandT],
) -> RunDirectoryCommandPlan[ActionT]:
    """Plan a posterior-predictive command from an existing run directory."""
    run_dir = request.run_dir.expanduser().resolve()
    model_path = IrArtifact(run_dir / "model.ir.json")
    data_path = CanonicalDataArtifact(run_dir / "data.json")
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "posterior_predictive.ndjson"
    _require_run_artifacts((model_path.path, data_path.path, fit_path))
    _reject_existing_output_artifacts((output_path,))
    context = PosteriorPredictiveRunContext(
        run_dir=run_dir,
        model_path=model_path,
        data_path=data_path,
        fit_path=fit_path,
        output_path=output_path,
    )
    action = backend.plan_posterior_predictive_action(context, request)
    return RunDirectoryCommandPlan(run_dir=run_dir, output_path=output_path, action=action)


def plan_posterior_check_run[ActionT, CommandT](
    request: PosteriorCheckRequest,
    backend: PosteriorCheckBackend[ActionT, CommandT],
) -> RunDirectoryCommandPlan[ActionT]:
    """Plan a posterior-check command from an existing run directory."""
    _validate_bayesite_only(request.backend, "posterior-check")
    run_dir = request.run_dir.expanduser().resolve()
    model_path = IrArtifact(run_dir / "model.ir.json")
    data_path = CanonicalDataArtifact(run_dir / "data.json")
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "posterior_check.json"
    _reject_reserved_engine_args(request.engine_args, output_path)
    _require_run_artifacts((model_path.path, data_path.path, fit_path))
    _reject_existing_output_artifacts((output_path,))
    context = PosteriorCheckRunContext(
        run_dir=run_dir,
        model_path=model_path,
        data_path=data_path,
        fit_path=fit_path,
        output_path=output_path,
    )
    action = backend.plan_posterior_check_action(context, request)
    return RunDirectoryCommandPlan(run_dir=run_dir, output_path=output_path, action=action)


def plan_recover_check_run[ActionT, CommandT](
    request: RecoverCheckRequest,
    backend: RecoverCheckBackend[ActionT, CommandT],
) -> RunDirectoryCommandPlan[ActionT]:
    """Plan a recover-check command from an existing run directory."""
    _validate_bayesite_only(request.backend, "recover-check")
    run_dir = request.run_dir.expanduser().resolve()
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "recovery_check.json"
    _reject_reserved_engine_args(request.engine_args, output_path)
    _require_run_artifacts((fit_path,))
    _reject_existing_output_artifacts((output_path,))
    truth_path = request.truth_path.expanduser().resolve()
    if not truth_path.is_file():
        raise WorkflowError(f"truth file does not exist: {truth_path}")
    targets_path: Path | None = None
    if request.targets_path is not None:
        targets_path = request.targets_path.expanduser().resolve()
        if not targets_path.is_file():
            raise WorkflowError(f"targets file does not exist: {targets_path}")
    context = RecoverCheckRunContext(
        run_dir=run_dir,
        fit_path=fit_path,
        truth_path=truth_path,
        targets_path=targets_path,
        output_path=output_path,
    )
    action = backend.plan_recover_check_action(context, request)
    return RunDirectoryCommandPlan(run_dir=run_dir, output_path=output_path, action=action)


def materialize_run_directory_command[ActionT, CommandT](
    plan: RunDirectoryCommandPlan[ActionT], backend: ActionBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned existing-run command for execution."""
    _reject_existing_output_artifacts((plan.output_path,))
    return backend.materialize(plan.action)


def plan_model_run_context(
    *,
    model_path: Path,
    model_name: str | None,
    data_path: Path,
    output_dir: Path,
) -> PlannedModelRunContext:
    """Plan shared model/data run directory inputs without writing them."""
    source_model_path = model_path.expanduser().resolve()
    source_model_sha256 = _sha256_uri(source_model_path)
    source_data_path = _require_input_file(data_path, "data")
    source_data_sha256 = _sha256_uri(source_data_path)
    try:
        data_doc = read_data_doc(source_data_path)
    except DataDocError as exc:
        raise WorkflowError(f"invalid data file: {exc}") from exc

    _validate_output_dir(output_dir)
    loaded_model = load_model(source_model_path, model_name)
    dims_path = _planned_dims_path(output_dir, loaded_model.model_cls, loaded_model.meta)

    return PlannedModelRunContext(
        model_name=loaded_model.name,
        model_source_path=source_model_path,
        model_source_sha256=source_model_sha256,
        loaded_model=loaded_model,
        ir_path=IrArtifact(output_dir / "model.ir.json"),
        data_source_path=source_data_path,
        data_source_sha256=source_data_sha256,
        data_path=CanonicalDataArtifact(output_dir / "data.json"),
        dims_path=dims_path,
        output_dir=output_dir,
        data_doc=data_doc,
    )


def materialize_model_run_context(
    context: PlannedModelRunContext,
    *,
    additional_data_paths: tuple[CanonicalDataArtifact, ...] = (),
) -> PlannedModelRunContext:
    """Materialize shared model/data run-directory inputs."""
    _ensure_output_dir(context.output_dir)
    context.ir_path.path.write_bytes(canonical_bytes(context.loaded_model.meta))
    write_data_doc(context.data_path.path, context.data_doc)
    _write_data_manifest(context.output_dir, context.data_path, additional_data_paths)
    dims_path = _write_optional_dims_sidecar(
        context.output_dir, context.loaded_model.model_cls, context.loaded_model.meta
    )
    return replace(context, dims_path=dims_path)


def plan_model_scenario_context(
    *,
    model_path: Path,
    model_name: str | None,
    scenario_path: Path,
    output_dir: Path,
) -> PlannedModelScenarioContext:
    """Plan shared model/scenario run directory inputs without writing them."""
    source_model_path = model_path.expanduser().resolve()
    source_model_sha256 = _sha256_uri(source_model_path)
    scenario_source = _require_input_file(scenario_path, "scenario")
    scenario_source_sha256 = _sha256_uri(scenario_source)
    _validate_output_dir(output_dir)
    loaded_model = load_model(source_model_path, model_name)
    dims_path = _planned_dims_path(output_dir, loaded_model.model_cls, loaded_model.meta)

    return PlannedModelScenarioContext(
        model_name=loaded_model.name,
        model_source_path=source_model_path,
        model_source_sha256=source_model_sha256,
        loaded_model=loaded_model,
        ir_path=IrArtifact(output_dir / "model.ir.json"),
        scenario_source_path=scenario_source,
        scenario_source_sha256=scenario_source_sha256,
        scenario_path=output_dir / "scenario.json",
        dims_path=dims_path,
        output_dir=output_dir,
    )


def materialize_model_scenario_context(
    context: PlannedModelScenarioContext,
) -> PlannedModelScenarioContext:
    """Materialize shared model/scenario run-directory inputs."""
    _ensure_output_dir(context.output_dir)
    context.ir_path.path.write_bytes(canonical_bytes(context.loaded_model.meta))
    _copy_required_input(context.scenario_source_path, context.scenario_path, "scenario")
    scenario_sha256 = _sha256_uri(context.scenario_path)
    dims_path = _write_optional_dims_sidecar(
        context.output_dir, context.loaded_model.model_cls, context.loaded_model.meta
    )
    return replace(context, dims_path=dims_path, scenario_source_sha256=scenario_sha256)


def sample_plan_document[ActionT, CommandT](
    run: SampleRunPlan[ActionT], backend: SampleBackend[ActionT, CommandT]
) -> SamplePlanDocument:
    """Return a serializable sample plan summary."""
    context = run.context
    document: dict[str, object] = {
        "model": context.model_name,
        "ir": str(context.ir_path.path),
        "data": str(context.data_path.path),
        "draws": str(run.draws_path),
        "output": str(context.output_dir),
    }
    document.update(backend_plan_description_fields(backend.describe(run.action)))
    if context.dims_path is not None:
        document["dims"] = str(context.dims_path)
    return cast(SamplePlanDocument, document)


def prior_predictive_plan_document[ActionT, CommandT](
    run: PriorPredictiveRunPlan[ActionT],
    backend: PriorPredictiveBackend[ActionT, CommandT],
) -> ModelCommandPlanDocument:
    """Return a serializable prior-predictive plan summary."""
    context = run.context
    return _model_command_document(
        model_name=context.model_name,
        ir_path=context.ir_path,
        output_dir=context.output_dir,
        description=backend_plan_description_fields(backend.describe(run.action)),
        dims_path=context.dims_path,
        data_path=context.data_path,
        extra={"prior_predictive": str(run.prior_predictive_path)},
    )


def simulate_plan_document[ActionT, CommandT](
    run: SimulateRunPlan[ActionT], backend: SimulateBackend[ActionT, CommandT]
) -> ModelCommandPlanDocument:
    """Return a serializable simulate plan summary."""
    context = run.context
    return _model_command_document(
        model_name=context.model_name,
        ir_path=context.ir_path,
        output_dir=context.output_dir,
        description=backend_plan_description_fields(backend.describe(run.action)),
        dims_path=context.dims_path,
        data_path=context.data_path,
        extra={
            "truth": str(run.truth_path),
            "simulated_data": str(run.simulated_data_path.path),
        },
    )


def recover_plan_document[ActionT, CommandT](
    run: RecoverRunPlan[ActionT], backend: RecoverBackend[ActionT, CommandT]
) -> ModelCommandPlanDocument:
    """Return a serializable recover plan summary."""
    context = run.context
    return _model_command_document(
        model_name=context.model_name,
        ir_path=context.ir_path,
        output_dir=context.output_dir,
        description=backend_plan_description_fields(backend.describe(run.action)),
        dims_path=context.dims_path,
        data_path=None,
        extra={"scenario": str(context.scenario_path), "recovery": str(run.recovery_path)},
    )


def sbc_plan_document[ActionT, CommandT](
    run: SbcRunPlan[ActionT], backend: SbcBackend[ActionT, CommandT]
) -> ModelCommandPlanDocument:
    """Return a serializable SBC plan summary."""
    context = run.context
    return _model_command_document(
        model_name=context.model_name,
        ir_path=context.ir_path,
        output_dir=context.output_dir,
        description=backend_plan_description_fields(backend.describe(run.action)),
        dims_path=context.dims_path,
        data_path=None,
        extra={"scenario": str(context.scenario_path), "sbc": str(run.sbc_path)},
    )


def run_command_plan_document[ActionT, CommandT](
    plan: RunDirectoryCommandPlan[ActionT], backend: ActionBackend[ActionT, CommandT]
) -> RunCommandPlanDocument:
    """Return a serializable plan summary for an existing-run command."""
    document: dict[str, object] = {
        "run": str(plan.run_dir),
        "output": str(plan.output_path),
    }
    document.update(backend_plan_description_fields(backend.describe(plan.action)))
    return cast(RunCommandPlanDocument, document)


def _model_command_document(
    *,
    model_name: str,
    ir_path: IrArtifact,
    output_dir: Path,
    description: BackendPlanFields,
    dims_path: Path | None,
    data_path: CanonicalDataArtifact | None,
    extra: dict[str, str],
) -> ModelCommandPlanDocument:
    document: dict[str, object] = {
        "model": model_name,
        "ir": str(ir_path.path),
        "output": str(output_dir),
    }
    if data_path is not None:
        document["data"] = str(data_path.path)
    document.update(extra)
    document.update(description)
    if dims_path is not None:
        document["dims"] = str(dims_path)
    return cast(ModelCommandPlanDocument, document)


def _validate_backend(backend: str) -> None:
    if backend not in {"bayesite", "jaxstanv5"}:
        raise WorkflowError("--backend must be 'bayesite' or 'jaxstanv5'")


def _validate_bayesite_only(backend: str, command: str) -> None:
    _validate_backend(backend)
    if backend != "bayesite":
        raise WorkflowError(f"bayescycle {command} is not supported on --backend {backend}")


def _reject_in_process_engine_args(backend: str, engine_args: tuple[str, ...]) -> None:
    if backend == "jaxstanv5" and engine_args:
        raise WorkflowError("engine passthrough after -- is only supported for --backend bayesite")


def _reject_reserved_engine_args(engine_args: tuple[str, ...], output_path: Path) -> None:
    for arg in engine_args:
        if arg == "--out" or arg.startswith("--out="):
            raise WorkflowError(
                "forwarded engine args may not include --out; "
                f"bayescycle writes output to {output_path}. "
                "Use Bayesite directly for custom output or streaming."
            )


def _require_input_file(source: Path, label: str) -> Path:
    source_path = source.expanduser().resolve()
    if not source_path.is_file():
        raise WorkflowError(f"{label} file does not exist: {source_path}")
    return source_path


def _copy_required_input(source: Path, destination: Path, label: str) -> Path:
    source_path = _require_input_file(source, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path != destination:
        shutil.copyfile(source_path, destination)
    return destination


def _require_run_artifacts(paths: tuple[Path, ...]) -> None:
    missing = tuple(path for path in paths if not path.is_file())
    if not missing:
        return
    run_dir = missing[0].parent
    names = ", ".join(path.name for path in missing)
    plural = "s" if len(missing) != 1 else ""
    raise WorkflowError(
        f"missing required run artifact{plural} in {run_dir}: {names}. "
        f"Run `bayescycle sample ... -o {run_dir}` first."
    )


def _reject_existing_output_artifacts(paths: tuple[Path, ...]) -> None:
    existing = tuple(path for path in paths if path.exists() or path.is_symlink())
    if not existing:
        return
    names = ", ".join(path.name for path in existing)
    plural = "s" if len(existing) != 1 else ""
    raise WorkflowError(
        f"output artifact{plural} already exists in {existing[0].parent}: {names}; "
        "choose a new run directory or remove the derived artifact intentionally"
    )


def _model_data_run_metadata(
    context: PlannedModelRunContext,
    *,
    kind: str,
    backend: str,
    outputs: tuple[RunMetadataOutput, ...],
    extra_inputs: tuple[RunMetadataInput, ...] = (),
) -> RunMetadata:
    return RunMetadata(
        kind=kind,
        backend=backend,
        model=RunMetadataModel(
            name=context.model_name,
            source_path=context.model_source_path,
            source_sha256=context.model_source_sha256,
            ir_path=context.ir_path.path,
        ),
        inputs=(
            RunMetadataInput(
                role="data",
                source_path=context.data_source_path,
                source_sha256=context.data_source_sha256,
                materialized_path=context.data_path.path,
                artifact_format=DATA_DOC_FORMAT,
            ),
            *extra_inputs,
        ),
        outputs=outputs,
    )


def _model_scenario_run_metadata(
    context: PlannedModelScenarioContext,
    *,
    kind: str,
    backend: str,
    outputs: tuple[RunMetadataOutput, ...],
) -> RunMetadata:
    return RunMetadata(
        kind=kind,
        backend=backend,
        model=RunMetadataModel(
            name=context.model_name,
            source_path=context.model_source_path,
            source_sha256=context.model_source_sha256,
            ir_path=context.ir_path.path,
        ),
        inputs=(
            RunMetadataInput(
                role="scenario",
                source_path=context.scenario_source_path,
                source_sha256=context.scenario_source_sha256,
                materialized_path=context.scenario_path,
            ),
        ),
        outputs=outputs,
    )


def _write_run_metadata(run_dir: Path, metadata: RunMetadata) -> None:
    with (run_dir / "run.json").open("x", encoding="utf-8") as f:
        json.dump(metadata.as_json(run_dir), f, indent=2)
        f.write("\n")


def _write_data_manifest(
    output_dir: Path,
    data_path: CanonicalDataArtifact,
    additional_data_paths: tuple[CanonicalDataArtifact, ...] = (),
) -> None:
    artifacts = {
        artifact.path.name: {
            "format": "bayescycle.data.json.v1",
            "path": artifact.path.name,
        }
        for artifact in (data_path, *additional_data_paths)
    }
    manifest = {
        "manifest_format": "bayescycle.run-manifest.v1",
        "artifacts": artifacts,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def _sha256_uri(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _relative_run_path(run_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _planned_dims_path(output_dir: Path, model_cls: type[object], meta: ModelMeta) -> Path | None:
    try:
        sidecar = dims_sidecar_for_model(model_cls, meta)
    except ValueError as exc:
        raise WorkflowError(f"invalid model dimension metadata: {exc}") from exc
    if sidecar is None:
        return None
    return output_dir / "dims.json"


def _write_optional_dims_sidecar(
    output_dir: Path, model_cls: type[object], meta: ModelMeta
) -> Path | None:
    try:
        sidecar = dims_sidecar_for_model(model_cls, meta)
    except ValueError as exc:
        raise WorkflowError(f"invalid model dimension metadata: {exc}") from exc
    dims_path = output_dir / "dims.json"
    if sidecar is None:
        dims_path.unlink(missing_ok=True)
        return None
    write_dims_sidecar(dims_path, sidecar)
    return dims_path


def _validate_output_dir(path: Path) -> None:
    if (path.exists() or path.is_symlink()) and not path.is_dir():
        raise WorkflowError(f"output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise WorkflowError(f"output directory is not empty: {path}; choose a new run directory")


def _ensure_output_dir(path: Path) -> None:
    _validate_output_dir(path)
    path.mkdir(parents=True, exist_ok=True)
