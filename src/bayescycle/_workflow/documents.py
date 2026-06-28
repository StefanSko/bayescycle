"""JSON plan document rendering."""

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict, cast

from bayescycle._integrations.descriptions import (
    BackendPlanFields,
    JsonNumber,
    backend_plan_description_fields,
)
from bayescycle._run_artifacts.references import CanonicalDataArtifact, IrArtifact
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
    PriorPredictiveBackend,
    RecoverBackend,
    SampleBackend,
    SbcBackend,
    SimulateBackend,
)


class SamplePlanDocument(TypedDict):
    """JSON document printed for a sample plan."""

    model: str
    ir: str
    data: str
    draws: str
    output: str
    command: NotRequired[list[str]]
    integration_mode: NotRequired[str]
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
    command: NotRequired[list[str]]
    integration_mode: NotRequired[str]
    backend_simulated_data: NotRequired[str]
    backend: NotRequired[str]
    settings: NotRequired[dict[str, JsonNumber]]
    dims: NotRequired[str]


class RunCommandPlanDocument(TypedDict):
    """JSON document printed for an existing-run command plan."""

    run: str
    output: str
    command: NotRequired[list[str]]
    backend: NotRequired[str]
    integration_mode: NotRequired[str]


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
