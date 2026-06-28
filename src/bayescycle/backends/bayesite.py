"""Bayesite CLI backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import overload

from bayescycle._commands import BayesiteCommand, BayesiteSimulateCommand
from bayescycle._engine import run_bayesite_command
from bayescycle._errors import WorkflowError
from bayescycle._workflow import (
    PreparedModelRunContext,
    PreparedModelScenarioContext,
    PriorPredictiveRequest,
    RecoverRequest,
    SampleRequest,
    SbcRequest,
    SimulateRequest,
)
from bayescycle.data import DataDocError, read_data_doc, write_bayesite_data_doc, write_data_doc


@dataclass(frozen=True)
class BayesiteSampleAction:
    """Planned Bayesite sample action before backend-private files exist."""

    command: BayesiteCommand
    canonical_data_path: Path
    output_dir: Path


@dataclass(frozen=True)
class BayesitePriorPredictiveAction:
    """Planned Bayesite prior-predictive action before backend-private files exist."""

    command: BayesiteCommand
    canonical_data_path: Path
    output_dir: Path


@dataclass(frozen=True)
class BayesiteSimulateAction:
    """Planned Bayesite simulate action before backend-private files exist."""

    command: BayesiteCommand
    canonical_data_path: Path
    output_dir: Path
    native_data_path: Path
    canonical_data_output_path: Path


@dataclass(frozen=True)
class BayesiteRecoverAction:
    """Planned Bayesite recover action."""

    command: BayesiteCommand


@dataclass(frozen=True)
class BayesiteSbcAction:
    """Planned Bayesite SBC action."""

    command: BayesiteCommand


type BayesiteAction = (
    BayesiteSampleAction
    | BayesitePriorPredictiveAction
    | BayesiteSimulateAction
    | BayesiteRecoverAction
    | BayesiteSbcAction
)

type BayesiteExecutableCommand = BayesiteCommand | BayesiteSimulateCommand


@dataclass(frozen=True)
class BayesiteBackend:
    """Bayesite subprocess backend capabilities."""

    engine: str

    def plan_sample_action(
        self, context: PreparedModelRunContext, request: SampleRequest
    ) -> BayesiteSampleAction:
        """Plan the Bayesite sample action for a planned run directory."""
        draws_path = context.output_dir / "posterior.ndjson"
        engine_data_path = _backend_dir(context.output_dir) / "data.json"
        return BayesiteSampleAction(
            command=BayesiteCommand(
                argv=(
                    self.engine,
                    "sample",
                    "--model",
                    str(context.ir_path),
                    "--data",
                    str(engine_data_path),
                    *request.sampler.to_engine_args(),
                    "--out",
                    str(draws_path),
                    *request.engine_args,
                ),
                output_paths=(draws_path,),
            ),
            canonical_data_path=context.data_path,
            output_dir=context.output_dir,
        )

    def plan_prior_predictive_action(
        self, context: PreparedModelRunContext, request: PriorPredictiveRequest
    ) -> BayesitePriorPredictiveAction:
        """Plan the Bayesite prior-predictive action for a prepared run directory."""
        output_path = context.output_dir / "prior_predictive.ndjson"
        engine_data_path = _backend_dir(context.output_dir) / "data.json"
        return BayesitePriorPredictiveAction(
            command=BayesiteCommand(
                argv=(
                    self.engine,
                    "prior-predictive",
                    "--model",
                    str(context.ir_path),
                    "--data",
                    str(engine_data_path),
                    *_optional_arg("--seed", request.seed),
                    *_optional_arg("--draws", request.draws),
                    "--out",
                    str(output_path),
                    *request.engine_args,
                ),
                output_paths=(output_path,),
            ),
            canonical_data_path=context.data_path,
            output_dir=context.output_dir,
        )

    def plan_simulate_action(
        self, context: PreparedModelRunContext, request: SimulateRequest, truth_path: Path
    ) -> BayesiteSimulateAction:
        """Plan the Bayesite simulate action for a prepared run directory."""
        canonical_output_path = context.output_dir / "simulated_data.json"
        native_output_path = _backend_dir(context.output_dir) / "simulated_data.json"
        engine_data_path = _backend_dir(context.output_dir) / "data.json"
        return BayesiteSimulateAction(
            command=BayesiteCommand(
                argv=(
                    self.engine,
                    "simulate",
                    "--model",
                    str(context.ir_path),
                    "--data",
                    str(engine_data_path),
                    "--truth",
                    str(truth_path),
                    *_optional_arg("--seed", request.seed),
                    "--out",
                    str(native_output_path),
                    *request.engine_args,
                ),
                output_paths=(canonical_output_path, native_output_path),
            ),
            canonical_data_path=context.data_path,
            output_dir=context.output_dir,
            native_data_path=native_output_path,
            canonical_data_output_path=canonical_output_path,
        )

    def plan_recover_action(
        self, context: PreparedModelScenarioContext, request: RecoverRequest
    ) -> BayesiteRecoverAction:
        """Plan the Bayesite recover action for a prepared run directory."""
        output_path = context.output_dir / "recovery.json"
        return BayesiteRecoverAction(
            command=BayesiteCommand(
                argv=(
                    self.engine,
                    "recover",
                    "--model",
                    str(context.ir_path),
                    "--scenario",
                    str(context.scenario_path),
                    "--out",
                    str(output_path),
                    *request.engine_args,
                ),
                output_paths=(output_path,),
            )
        )

    def plan_sbc_action(
        self, context: PreparedModelScenarioContext, request: SbcRequest
    ) -> BayesiteSbcAction:
        """Plan the Bayesite SBC action for a prepared run directory."""
        output_path = context.output_dir / "sbc.json"
        return BayesiteSbcAction(
            command=BayesiteCommand(
                argv=(
                    self.engine,
                    "sbc",
                    "--model",
                    str(context.ir_path),
                    "--scenario",
                    str(context.scenario_path),
                    *_optional_arg("--replicates", request.replicates),
                    "--out",
                    str(output_path),
                    *request.engine_args,
                ),
                output_paths=(output_path,),
            )
        )

    def describe(self, action: BayesiteAction) -> dict[str, object]:
        """Return Bayesite-owned dry-run fields for a planned action."""
        fields: dict[str, object] = {"engine_command": list(action.command.argv)}
        if isinstance(action, BayesiteSimulateAction):
            fields["backend_simulated_data"] = str(action.native_data_path)
        return fields

    @overload
    def materialize(self, action: BayesiteSampleAction) -> BayesiteCommand: ...

    @overload
    def materialize(self, action: BayesitePriorPredictiveAction) -> BayesiteCommand: ...

    @overload
    def materialize(self, action: BayesiteSimulateAction) -> BayesiteSimulateCommand: ...

    @overload
    def materialize(self, action: BayesiteRecoverAction) -> BayesiteCommand: ...

    @overload
    def materialize(self, action: BayesiteSbcAction) -> BayesiteCommand: ...

    def materialize(self, action: BayesiteAction) -> BayesiteExecutableCommand:
        """Materialize Bayesite-private inputs and return an executable command."""
        if isinstance(
            action,
            (BayesiteSampleAction, BayesitePriorPredictiveAction, BayesiteSimulateAction),
        ):
            _materialize_engine_data(action.canonical_data_path, action.output_dir)
        if isinstance(action, BayesiteSimulateAction):
            return BayesiteSimulateCommand(
                engine_command=action.command,
                native_data_path=action.native_data_path,
                canonical_data_path=action.canonical_data_output_path,
            )
        return action.command

    def execute(self, command: BayesiteExecutableCommand) -> int:
        """Execute a materialized Bayesite command."""
        if isinstance(command, BayesiteSimulateCommand):
            code = run_bayesite_command(command.engine_command)
            if code != 0:
                return code
            try:
                generated = read_data_doc(command.native_data_path)
                write_data_doc(command.canonical_data_path, generated)
            except DataDocError as exc:
                raise WorkflowError(
                    f"Bayesite simulate did not produce a valid data artifact: {exc}"
                ) from exc
            return 0
        return run_bayesite_command(command)


def materialize_run_data_for_bayesite(run_dir: Path) -> Path:
    """Materialize ``run/data.json`` into Bayesite's backend-private data file."""
    return _materialize_engine_data(run_dir / "data.json", run_dir)


def _materialize_engine_data(canonical_data_path: Path, output_dir: Path) -> Path:
    try:
        doc = read_data_doc(canonical_data_path)
        engine_data_path = _backend_dir(output_dir) / "data.json"
        write_bayesite_data_doc(engine_data_path, doc)
    except DataDocError as exc:
        raise WorkflowError(f"invalid data artifact for Bayesite backend: {exc}") from exc
    return engine_data_path


def _backend_dir(output_dir: Path) -> Path:
    return output_dir / ".bayesite"


def _optional_arg(flag: str, value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (flag, value)
