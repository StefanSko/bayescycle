"""Workflow phases for preparing and launching backend runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, cast

from jaxstanv5.ir import canonical_bytes
from jaxstanv5.model import ModelMeta

from bayescycle._commands import BayesiteCommand
from bayescycle._dims import dims_sidecar_for_model, write_dims_sidecar
from bayescycle._errors import WorkflowError
from bayescycle._model_loader import LoadedModel, load_model
from bayescycle._settings import SamplerSettings
from bayescycle.data import DataDoc, DataDocError, read_data_doc, write_data_doc


@dataclass(frozen=True)
class SampleRequest:
    """Loose CLI sampling input normalized into a typed request."""

    model_path: Path
    data_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    engine: str
    sampler: SamplerSettings
    engine_args: tuple[str, ...]
    force: bool


@dataclass(frozen=True)
class PriorPredictiveRequest:
    """Loose CLI prior-predictive input normalized into a typed request."""

    model_path: Path
    data_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    engine: str
    seed: str | None
    draws: str | None
    engine_args: tuple[str, ...]
    force: bool


@dataclass(frozen=True)
class SimulateRequest:
    """Loose CLI simulation input normalized into a typed request."""

    model_path: Path
    data_path: Path
    truth_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    engine: str
    seed: str | None
    engine_args: tuple[str, ...]
    force: bool


@dataclass(frozen=True)
class RecoverRequest:
    """Loose CLI single-scenario recovery input normalized into a typed request."""

    model_path: Path
    scenario_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    engine: str
    engine_args: tuple[str, ...]
    force: bool


@dataclass(frozen=True)
class SbcRequest:
    """Loose CLI SBC input normalized into a typed request."""

    model_path: Path
    scenario_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    engine: str
    replicates: str | None
    engine_args: tuple[str, ...]
    force: bool


@dataclass(frozen=True)
class RecoverCheckRequest:
    """Request to run recovery checks for an existing run directory."""

    run_dir: Path
    truth_path: Path
    targets_path: Path | None
    interval: str | None
    backend: str
    engine: str
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class PosteriorCheckRequest:
    """Request to run posterior checks for an existing run directory."""

    run_dir: Path
    seed: str | None
    backend: str
    engine: str
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class PreparedModelRunContext:
    """Prepared model/data files shared by model-level workflow commands."""

    model_name: str
    loaded_model: LoadedModel
    ir_path: Path
    data_path: Path
    dims_path: Path | None
    output_dir: Path
    data_doc: DataDoc


@dataclass(frozen=True)
class PreparedModelScenarioContext:
    """Prepared model/scenario files shared by scenario workflow commands."""

    model_name: str
    loaded_model: LoadedModel
    ir_path: Path
    scenario_path: Path
    dims_path: Path | None
    output_dir: Path


class BackendExecutor[CommandT](Protocol):
    """Backend capability for executing a materialized command."""

    def execute(self, command: CommandT) -> int:
        """Execute a typed backend command."""


class SampleBackend[ActionT, CommandT](BackendExecutor[CommandT], Protocol):
    """Backend capability for the sample workflow operation."""

    def plan_sample_action(
        self, context: PreparedModelRunContext, request: SampleRequest
    ) -> ActionT:
        """Plan a backend-private sample action from a planned run context."""

    def describe(self, action: ActionT) -> dict[str, object]:
        """Return backend-owned dry-run fields for an action."""

    def materialize(self, action: ActionT) -> CommandT:
        """Materialize backend-private inputs and return an executable command."""


class PriorPredictiveBackend[ActionT, CommandT](BackendExecutor[CommandT], Protocol):
    """Backend capability for the prior-predictive workflow operation."""

    def plan_prior_predictive_action(
        self, context: PreparedModelRunContext, request: PriorPredictiveRequest
    ) -> ActionT:
        """Plan a backend-private prior-predictive action."""

    def describe(self, action: ActionT) -> dict[str, object]:
        """Return backend-owned dry-run fields for an action."""

    def materialize(self, action: ActionT) -> CommandT:
        """Materialize backend-private inputs and return an executable command."""


class SimulateBackend[ActionT, CommandT](BackendExecutor[CommandT], Protocol):
    """Backend capability for the simulate workflow operation."""

    def plan_simulate_action(
        self, context: PreparedModelRunContext, request: SimulateRequest, truth_path: Path
    ) -> ActionT:
        """Plan a backend-private simulate action from a prepared run context."""

    def describe(self, action: ActionT) -> dict[str, object]:
        """Return backend-owned dry-run fields for an action."""

    def materialize(self, action: ActionT) -> CommandT:
        """Materialize backend-private inputs and return an executable command."""


class RecoverBackend[ActionT, CommandT](BackendExecutor[CommandT], Protocol):
    """Backend capability for the recover workflow operation."""

    def plan_recover_action(
        self, context: PreparedModelScenarioContext, request: RecoverRequest
    ) -> ActionT:
        """Plan a backend-private recover action from a prepared scenario context."""

    def describe(self, action: ActionT) -> dict[str, object]:
        """Return backend-owned dry-run fields for an action."""

    def materialize(self, action: ActionT) -> CommandT:
        """Materialize backend-private inputs and return an executable command."""


class SbcBackend[ActionT, CommandT](BackendExecutor[CommandT], Protocol):
    """Backend capability for the SBC workflow operation."""

    def plan_sbc_action(
        self, context: PreparedModelScenarioContext, request: SbcRequest
    ) -> ActionT:
        """Plan a backend-private SBC action from a prepared scenario context."""

    def describe(self, action: ActionT) -> dict[str, object]:
        """Return backend-owned dry-run fields for an action."""

    def materialize(self, action: ActionT) -> CommandT:
        """Materialize backend-private inputs and return an executable command."""


@dataclass(frozen=True)
class SampleRunPlan[ActionT]:
    """A sample run plan with workflow-owned paths and backend action."""

    model_name: str
    loaded_model: LoadedModel
    ir_path: Path
    data_path: Path
    dims_path: Path | None
    output_dir: Path
    data_doc: DataDoc
    draws_path: Path
    action: ActionT


@dataclass(frozen=True)
class PreparedPriorPredictiveRun[ActionT]:
    """A run directory and prior-predictive backend action produced from a request."""

    model_name: str
    ir_path: Path
    data_path: Path
    dims_path: Path | None
    output_dir: Path
    prior_predictive_path: Path
    action: ActionT


@dataclass(frozen=True)
class PreparedSimulateRun[ActionT]:
    """A run directory and simulate backend action produced from a request."""

    model_name: str
    ir_path: Path
    data_path: Path
    truth_path: Path
    dims_path: Path | None
    output_dir: Path
    simulated_data_path: Path
    action: ActionT


@dataclass(frozen=True)
class PreparedRecoverRun[ActionT]:
    """A run directory and recover backend action produced from a request."""

    model_name: str
    ir_path: Path
    scenario_path: Path
    dims_path: Path | None
    output_dir: Path
    recovery_path: Path
    action: ActionT


@dataclass(frozen=True)
class PreparedSbcRun[ActionT]:
    """A run directory and SBC backend action produced from a request."""

    model_name: str
    ir_path: Path
    scenario_path: Path
    dims_path: Path | None
    output_dir: Path
    sbc_path: Path
    action: ActionT


@dataclass(frozen=True)
class DiagnoseRequest:
    """Request to run diagnostics for an existing run directory."""

    run_dir: Path
    engine: str


@dataclass(frozen=True)
class PosteriorPredictiveRequest:
    """Request to run posterior predictive generation for an existing run directory."""

    run_dir: Path
    engine: str
    seed: str


@dataclass(frozen=True)
class PreparedRunDirectoryCommand:
    """A follow-up command produced from an existing run directory."""

    run_dir: Path
    output_path: Path
    command: BayesiteCommand


class DryRunDocument(TypedDict):
    """JSON document printed for sample ``--dry-run``."""

    model: str
    ir: str
    data: str
    draws: str
    output: str
    engine_command: NotRequired[list[str]]
    backend: NotRequired[str]
    sampler: NotRequired[dict[str, int | float]]
    dims: NotRequired[str]


class ModelCommandDryRunDocument(TypedDict):
    """JSON document printed for model-level command ``--dry-run``."""

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
    backend: NotRequired[str]
    settings: NotRequired[dict[str, int | float]]
    dims: NotRequired[str]


class RunCommandDryRunDocument(TypedDict):
    """JSON document printed for run-directory command ``--dry-run``."""

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
        force=request.force,
    )
    action = backend.plan_sample_action(context, request)
    return SampleRunPlan(
        model_name=context.model_name,
        loaded_model=context.loaded_model,
        ir_path=context.ir_path,
        data_path=context.data_path,
        dims_path=context.dims_path,
        output_dir=context.output_dir,
        data_doc=context.data_doc,
        draws_path=draws_path,
        action=action,
    )


def materialize_sample_run[ActionT, CommandT](
    plan: SampleRunPlan[ActionT], backend: SampleBackend[ActionT, CommandT]
) -> CommandT:
    """Materialize a planned sample run for execution."""
    _ensure_output_dir(plan.output_dir, force=True)
    plan.ir_path.write_bytes(canonical_bytes(plan.loaded_model.meta))
    write_data_doc(plan.data_path, plan.data_doc)
    _write_data_manifest(plan.output_dir, plan.data_path)
    _write_optional_dims_sidecar(
        plan.output_dir, plan.loaded_model.model_cls, plan.loaded_model.meta
    )
    return backend.materialize(plan.action)


def prepare_prior_predictive_run[ActionT, CommandT](
    request: PriorPredictiveRequest, backend: PriorPredictiveBackend[ActionT, CommandT]
) -> PreparedPriorPredictiveRun[ActionT]:
    """Prepare a prior-predictive run directory and command."""
    _validate_backend(request.backend)
    output_dir = request.output_dir.expanduser().resolve()
    output_path = output_dir / "prior_predictive.ndjson"
    _reject_reserved_engine_args(request.engine_args, output_path)
    _reject_in_process_engine_args(request.backend, request.engine_args)
    context = prepare_model_run_context(
        model_path=request.model_path,
        model_name=request.model_name,
        data_path=request.data_path,
        output_dir=output_dir,
        force=request.force,
    )
    action = backend.plan_prior_predictive_action(context, request)
    return PreparedPriorPredictiveRun(
        model_name=context.model_name,
        ir_path=context.ir_path,
        data_path=context.data_path,
        dims_path=context.dims_path,
        output_dir=context.output_dir,
        prior_predictive_path=output_path,
        action=action,
    )


def prepare_simulate_run[ActionT, CommandT](
    request: SimulateRequest, backend: SimulateBackend[ActionT, CommandT]
) -> PreparedSimulateRun[ActionT]:
    """Prepare a simulation run directory and command."""
    _validate_bayesite_only(request.backend, "simulate")
    output_dir = request.output_dir.expanduser().resolve()
    output_path = output_dir / "simulated_data.json"
    _reject_reserved_engine_args(request.engine_args, output_path)
    truth_source = _require_input_file(request.truth_path, "truth")
    context = prepare_model_run_context(
        model_path=request.model_path,
        model_name=request.model_name,
        data_path=request.data_path,
        output_dir=output_dir,
        force=request.force,
    )
    truth_path = _copy_required_input(truth_source, output_dir / "truth.json", "truth")
    _write_data_manifest(output_dir, context.data_path, (output_path,))
    action = backend.plan_simulate_action(context, request, truth_path)
    return PreparedSimulateRun(
        model_name=context.model_name,
        ir_path=context.ir_path,
        data_path=context.data_path,
        truth_path=truth_path,
        dims_path=context.dims_path,
        output_dir=context.output_dir,
        simulated_data_path=output_path,
        action=action,
    )


def prepare_recover_run[ActionT, CommandT](
    request: RecoverRequest, backend: RecoverBackend[ActionT, CommandT]
) -> PreparedRecoverRun[ActionT]:
    """Prepare a single-scenario recovery run directory and command."""
    _validate_bayesite_only(request.backend, "recover")
    output_dir = request.output_dir.expanduser().resolve()
    output_path = output_dir / "recovery.json"
    _reject_reserved_engine_args(request.engine_args, output_path)
    context = prepare_model_scenario_context(
        model_path=request.model_path,
        model_name=request.model_name,
        scenario_path=request.scenario_path,
        output_dir=output_dir,
        force=request.force,
    )
    action = backend.plan_recover_action(context, request)
    return PreparedRecoverRun(
        model_name=context.model_name,
        ir_path=context.ir_path,
        scenario_path=context.scenario_path,
        dims_path=context.dims_path,
        output_dir=context.output_dir,
        recovery_path=output_path,
        action=action,
    )


def prepare_sbc_run[ActionT, CommandT](
    request: SbcRequest, backend: SbcBackend[ActionT, CommandT]
) -> PreparedSbcRun[ActionT]:
    """Prepare an SBC run directory and command."""
    _validate_bayesite_only(request.backend, "sbc")
    output_dir = request.output_dir.expanduser().resolve()
    output_path = output_dir / "sbc.json"
    _reject_reserved_engine_args(request.engine_args, output_path)
    context = prepare_model_scenario_context(
        model_path=request.model_path,
        model_name=request.model_name,
        scenario_path=request.scenario_path,
        output_dir=output_dir,
        force=request.force,
    )
    action = backend.plan_sbc_action(context, request)
    return PreparedSbcRun(
        model_name=context.model_name,
        ir_path=context.ir_path,
        scenario_path=context.scenario_path,
        dims_path=context.dims_path,
        output_dir=context.output_dir,
        sbc_path=output_path,
        action=action,
    )


def plan_model_run_context(
    *,
    model_path: Path,
    model_name: str | None,
    data_path: Path,
    output_dir: Path,
    force: bool,
) -> PreparedModelRunContext:
    """Plan shared model/data run directory inputs without writing them."""
    source_data_path = _require_input_file(data_path, "data")
    try:
        data_doc = read_data_doc(source_data_path)
    except DataDocError as exc:
        raise WorkflowError(f"invalid data file: {exc}") from exc

    _validate_output_dir(output_dir, force=force)
    loaded_model = load_model(model_path, model_name)
    try:
        dims_path = (
            output_dir / "dims.json"
            if dims_sidecar_for_model(loaded_model.model_cls, loaded_model.meta) is not None
            else None
        )
    except ValueError as exc:
        raise WorkflowError(f"invalid model dimension metadata: {exc}") from exc

    return PreparedModelRunContext(
        model_name=loaded_model.name,
        loaded_model=loaded_model,
        ir_path=output_dir / "model.ir.json",
        data_path=output_dir / "data.json",
        dims_path=dims_path,
        output_dir=output_dir,
        data_doc=data_doc,
    )


def prepare_model_run_context(
    *,
    model_path: Path,
    model_name: str | None,
    data_path: Path,
    output_dir: Path,
    force: bool,
) -> PreparedModelRunContext:
    """Prepare the shared model/data run directory inputs for model-level commands."""
    context = plan_model_run_context(
        model_path=model_path,
        model_name=model_name,
        data_path=data_path,
        output_dir=output_dir,
        force=force,
    )
    _ensure_output_dir(output_dir, force=True)
    context.ir_path.write_bytes(canonical_bytes(context.loaded_model.meta))
    write_data_doc(context.data_path, context.data_doc)
    _write_data_manifest(output_dir, context.data_path)
    dims_path = _write_optional_dims_sidecar(
        output_dir, context.loaded_model.model_cls, context.loaded_model.meta
    )

    return PreparedModelRunContext(
        model_name=context.model_name,
        loaded_model=context.loaded_model,
        ir_path=context.ir_path,
        data_path=context.data_path,
        dims_path=dims_path,
        output_dir=context.output_dir,
        data_doc=context.data_doc,
    )


def prepare_model_scenario_context(
    *,
    model_path: Path,
    model_name: str | None,
    scenario_path: Path,
    output_dir: Path,
    force: bool,
) -> PreparedModelScenarioContext:
    """Prepare shared model/scenario run directory inputs for scenario commands."""
    scenario_source = _require_input_file(scenario_path, "scenario")
    _ensure_output_dir(output_dir, force=force)

    loaded_model = load_model(model_path, model_name)

    ir_path = output_dir / "model.ir.json"
    ir_path.write_bytes(canonical_bytes(loaded_model.meta))

    run_scenario_path = _copy_required_input(
        scenario_source, output_dir / "scenario.json", "scenario"
    )
    dims_path = _write_optional_dims_sidecar(output_dir, loaded_model.model_cls, loaded_model.meta)

    return PreparedModelScenarioContext(
        model_name=loaded_model.name,
        loaded_model=loaded_model,
        ir_path=ir_path,
        scenario_path=run_scenario_path,
        dims_path=dims_path,
        output_dir=output_dir,
    )


def sample_plan_document[ActionT, CommandT](
    run: SampleRunPlan[ActionT], backend: SampleBackend[ActionT, CommandT]
) -> DryRunDocument:
    """Return a serializable sample plan summary."""
    document: dict[str, object] = {
        "model": run.model_name,
        "ir": str(run.ir_path),
        "data": str(run.data_path),
        "draws": str(run.draws_path),
        "output": str(run.output_dir),
    }
    document.update(backend.describe(run.action))
    if run.dims_path is not None:
        document["dims"] = str(run.dims_path)
    return cast(DryRunDocument, document)


def prior_predictive_dry_run_document[ActionT, CommandT](
    run: PreparedPriorPredictiveRun[ActionT],
    backend: PriorPredictiveBackend[ActionT, CommandT],
) -> ModelCommandDryRunDocument:
    """Return a serializable prior-predictive dry-run summary."""
    return _model_command_document(
        model_name=run.model_name,
        ir_path=run.ir_path,
        output_dir=run.output_dir,
        description=backend.describe(run.action),
        dims_path=run.dims_path,
        data_path=run.data_path,
        extra={"prior_predictive": str(run.prior_predictive_path)},
    )


def simulate_dry_run_document[ActionT, CommandT](
    run: PreparedSimulateRun[ActionT], backend: SimulateBackend[ActionT, CommandT]
) -> ModelCommandDryRunDocument:
    """Return a serializable simulate dry-run summary."""
    return _model_command_document(
        model_name=run.model_name,
        ir_path=run.ir_path,
        output_dir=run.output_dir,
        description=backend.describe(run.action),
        dims_path=run.dims_path,
        data_path=run.data_path,
        extra={
            "truth": str(run.truth_path),
            "simulated_data": str(run.simulated_data_path),
        },
    )


def recover_dry_run_document[ActionT, CommandT](
    run: PreparedRecoverRun[ActionT], backend: RecoverBackend[ActionT, CommandT]
) -> ModelCommandDryRunDocument:
    """Return a serializable recover dry-run summary."""
    return _model_command_document(
        model_name=run.model_name,
        ir_path=run.ir_path,
        output_dir=run.output_dir,
        description=backend.describe(run.action),
        dims_path=run.dims_path,
        data_path=None,
        extra={"scenario": str(run.scenario_path), "recovery": str(run.recovery_path)},
    )


def sbc_dry_run_document[ActionT, CommandT](
    run: PreparedSbcRun[ActionT], backend: SbcBackend[ActionT, CommandT]
) -> ModelCommandDryRunDocument:
    """Return a serializable SBC dry-run summary."""
    return _model_command_document(
        model_name=run.model_name,
        ir_path=run.ir_path,
        output_dir=run.output_dir,
        description=backend.describe(run.action),
        dims_path=run.dims_path,
        data_path=None,
        extra={"scenario": str(run.scenario_path), "sbc": str(run.sbc_path)},
    )


def prepare_diagnose_run(request: DiagnoseRequest) -> PreparedRunDirectoryCommand:
    """Build a Bayesite diagnose command from an existing run directory."""
    run_dir = request.run_dir.expanduser().resolve()
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "diagnostics.json"
    _require_run_artifacts((fit_path,))
    command = BayesiteCommand(
        argv=(
            request.engine,
            "diagnose",
            "--fit",
            str(fit_path),
            "--out",
            str(output_path),
        ),
        output_paths=(output_path,),
    )
    return PreparedRunDirectoryCommand(
        run_dir=run_dir,
        output_path=output_path,
        command=command,
    )


def prepare_posterior_predictive_run(
    request: PosteriorPredictiveRequest,
) -> PreparedRunDirectoryCommand:
    """Build a Bayesite posterior-predictive command from an existing run directory."""
    run_dir = request.run_dir.expanduser().resolve()
    model_path = run_dir / "model.ir.json"
    data_path = run_dir / "data.json"
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "posterior_predictive.ndjson"
    _require_run_artifacts((model_path, data_path, fit_path))
    from bayescycle.backends.bayesite import materialize_run_data_for_bayesite

    engine_data_path = materialize_run_data_for_bayesite(run_dir)
    command = BayesiteCommand(
        argv=(
            request.engine,
            "posterior-predictive",
            "--model",
            str(model_path),
            "--data",
            str(engine_data_path),
            "--fit",
            str(fit_path),
            "--seed",
            request.seed,
            "--out",
            str(output_path),
        ),
        output_paths=(output_path,),
    )
    return PreparedRunDirectoryCommand(
        run_dir=run_dir,
        output_path=output_path,
        command=command,
    )


def prepare_posterior_check_run(request: PosteriorCheckRequest) -> PreparedRunDirectoryCommand:
    """Build a Bayesite posterior-check command from an existing run directory."""
    _validate_bayesite_only(request.backend, "posterior-check")
    run_dir = request.run_dir.expanduser().resolve()
    model_path = run_dir / "model.ir.json"
    data_path = run_dir / "data.json"
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "posterior_check.json"
    _reject_reserved_engine_args(request.engine_args, output_path)
    _require_run_artifacts((model_path, data_path, fit_path))
    from bayescycle.backends.bayesite import materialize_run_data_for_bayesite

    engine_data_path = materialize_run_data_for_bayesite(run_dir)
    seed_args = ("--seed", request.seed) if request.seed is not None else ()
    command = BayesiteCommand(
        argv=(
            request.engine,
            "posterior-check",
            "--model",
            str(model_path),
            "--data",
            str(engine_data_path),
            "--fit",
            str(fit_path),
            *seed_args,
            "--out",
            str(output_path),
            *request.engine_args,
        ),
        output_paths=(output_path,),
    )
    return PreparedRunDirectoryCommand(run_dir=run_dir, output_path=output_path, command=command)


def prepare_recover_check_run(request: RecoverCheckRequest) -> PreparedRunDirectoryCommand:
    """Build a Bayesite recover-check command from an existing run directory."""
    _validate_bayesite_only(request.backend, "recover-check")
    run_dir = request.run_dir.expanduser().resolve()
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "recovery_check.json"
    _reject_reserved_engine_args(request.engine_args, output_path)
    _require_run_artifacts((fit_path,))
    truth_path = request.truth_path.expanduser().resolve()
    if not truth_path.is_file():
        raise WorkflowError(f"truth file does not exist: {truth_path}")
    targets_args: tuple[str, ...] = ()
    if request.targets_path is not None:
        targets_path = request.targets_path.expanduser().resolve()
        if not targets_path.is_file():
            raise WorkflowError(f"targets file does not exist: {targets_path}")
        targets_args = ("--targets", str(targets_path))
    interval_args = ("--interval", request.interval) if request.interval is not None else ()
    command = BayesiteCommand(
        argv=(
            request.engine,
            "recover-check",
            "--fit",
            str(fit_path),
            "--truth",
            str(truth_path),
            *targets_args,
            *interval_args,
            "--out",
            str(output_path),
            *request.engine_args,
        ),
        output_paths=(output_path,),
    )
    return PreparedRunDirectoryCommand(run_dir=run_dir, output_path=output_path, command=command)


def run_command_dry_run_document(
    command: PreparedRunDirectoryCommand,
) -> RunCommandDryRunDocument:
    """Return a serializable dry-run summary for a run-directory command."""
    return {
        "run": str(command.run_dir),
        "output": str(command.output_path),
        "engine_command": list(command.command.argv),
    }


def _model_command_document(
    *,
    model_name: str,
    ir_path: Path,
    output_dir: Path,
    description: dict[str, object],
    dims_path: Path | None,
    data_path: Path | None,
    extra: dict[str, str],
) -> ModelCommandDryRunDocument:
    document: dict[str, object] = {
        "model": model_name,
        "ir": str(ir_path),
        "output": str(output_dir),
    }
    if data_path is not None:
        document["data"] = str(data_path)
    document.update(extra)
    document.update(description)
    if dims_path is not None:
        document["dims"] = str(dims_path)
    return cast(ModelCommandDryRunDocument, document)


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


def _write_data_manifest(
    output_dir: Path, data_path: Path, additional_data_paths: tuple[Path, ...] = ()
) -> None:
    artifacts = {
        path.name: {
            "format": "bayescycle.data.json.v1",
            "path": path.name,
        }
        for path in (data_path, *additional_data_paths)
    }
    manifest = {
        "manifest_format": "bayescycle.run-manifest.v1",
        "artifacts": artifacts,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


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


def _validate_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise WorkflowError(f"output path exists and is not a directory: {path}")
    if path.exists() and not force and any(path.iterdir()):
        raise WorkflowError(f"output directory is not empty: {path}; pass --force to reuse it")


def _ensure_output_dir(path: Path, *, force: bool) -> None:
    _validate_output_dir(path, force=force)
    path.mkdir(parents=True, exist_ok=True)
