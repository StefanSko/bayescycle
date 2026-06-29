"""Workflow planning and materialization operations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from jaxstanv5.ir import canonical_bytes
from jaxstanv5.model import ModelMeta

from bayescycle._errors import WorkflowError
from bayescycle._model_loader import load_model
from bayescycle._run_artifacts.canonical_data import (
    DATA_DOC_FORMAT,
    DataDocError,
    read_data_doc,
    write_data_doc,
)
from bayescycle._run_artifacts.dimensions import dims_sidecar_for_model, write_dims_sidecar
from bayescycle._run_artifacts.manifest import write_data_manifest
from bayescycle._run_artifacts.references import CanonicalDataArtifact, IrArtifact
from bayescycle._run_artifacts.run_metadata import (
    RunMetadata,
    RunMetadataInput,
    RunMetadataModel,
    RunMetadataOutput,
    sha256_uri,
    write_run_metadata,
)
from bayescycle._workflow.contexts import (
    DiagnoseRunContext,
    PlannedModelRunContext,
    PlannedModelScenarioContext,
    PosteriorCheckRunContext,
    PosteriorPredictiveRunContext,
    RecoverCheckRunContext,
)
from bayescycle._workflow.filesystem import (
    _copy_required_input,
    _ensure_output_dir,
    _reject_existing_output_artifacts,
    _require_input_file,
    _require_run_artifacts,
    _validate_output_dir,
)
from bayescycle._workflow.plans import (
    PriorPredictiveRunPlan,
    RecoverRunPlan,
    RunDirectoryCommandPlan,
    SampleRunPlan,
    SbcRunPlan,
    SimulateRunPlan,
)
from bayescycle._workflow.protocols import (
    ActionBackend,
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
from bayescycle._workflow.requests import (
    DiagnoseRequest,
    PosteriorCheckRequest,
    PosteriorPredictiveRequest,
    PriorPredictiveRequest,
    RecoverCheckRequest,
    RecoverRequest,
    SampleRequest,
    SbcRequest,
    SimulateRequest,
)


def plan_sample_run[ActionT, CommandT](
    request: SampleRequest, backend: SampleBackend[ActionT, CommandT]
) -> SampleRunPlan[ActionT]:
    """Plan a sample run without durable filesystem writes."""
    output_dir = request.output_dir.expanduser().resolve()
    draws_path = output_dir / "posterior.ndjson"
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
        backend=backend.backend_id,
        action=action,
    )


def materialize_sample_run[ActionT, CommandT](
    plan: SampleRunPlan[ActionT], backend: SampleBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned sample run for execution."""
    context = materialize_model_run_context(plan.context)
    write_run_metadata(
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
    output_dir = request.output_dir.expanduser().resolve()
    output_path = output_dir / "prior_predictive.ndjson"
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
        backend=backend.backend_id,
        action=action,
    )


def materialize_prior_predictive_run[ActionT, CommandT](
    plan: PriorPredictiveRunPlan[ActionT], backend: PriorPredictiveBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned prior-predictive run for execution."""
    context = materialize_model_run_context(plan.context)
    write_run_metadata(
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
    output_dir = request.output_dir.expanduser().resolve()
    output_path = CanonicalDataArtifact(output_dir / "simulated_data.json")
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
        backend=backend.backend_id,
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
    truth_sha256 = sha256_uri(plan.truth_path)
    write_run_metadata(
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
    output_dir = request.output_dir.expanduser().resolve()
    output_path = output_dir / "recovery.json"
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
        backend=backend.backend_id,
        action=action,
    )


def materialize_recover_run[ActionT, CommandT](
    plan: RecoverRunPlan[ActionT], backend: RecoverBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned single-scenario recovery run for execution."""
    context = materialize_model_scenario_context(plan.context)
    write_run_metadata(
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
    output_dir = request.output_dir.expanduser().resolve()
    output_path = output_dir / "sbc.json"
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
        backend=backend.backend_id,
        action=action,
    )


def materialize_sbc_run[ActionT, CommandT](
    plan: SbcRunPlan[ActionT], backend: SbcBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned SBC run for execution."""
    context = materialize_model_scenario_context(plan.context)
    write_run_metadata(
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
    run_dir = request.run_dir.expanduser().resolve()
    model_path = IrArtifact(run_dir / "model.ir.json")
    data_path = CanonicalDataArtifact(run_dir / "data.json")
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "posterior_check.json"
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
    run_dir = request.run_dir.expanduser().resolve()
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "recovery_check.json"
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
    source_model_sha256 = sha256_uri(source_model_path)
    source_data_path = _require_input_file(data_path, "data")
    source_data_sha256 = sha256_uri(source_data_path)
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
    write_data_manifest(context.output_dir, context.data_path, additional_data_paths)
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
    source_model_sha256 = sha256_uri(source_model_path)
    scenario_source = _require_input_file(scenario_path, "scenario")
    scenario_source_sha256 = sha256_uri(scenario_source)
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
    scenario_sha256 = sha256_uri(context.scenario_path)
    dims_path = _write_optional_dims_sidecar(
        context.output_dir, context.loaded_model.model_cls, context.loaded_model.meta
    )
    return replace(context, dims_path=dims_path, scenario_source_sha256=scenario_sha256)


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
