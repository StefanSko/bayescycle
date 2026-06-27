"""Bayesite CLI backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
class BayesiteBackend:
    """Bayesite subprocess backend capabilities."""

    engine: str

    def build_sample_command(
        self, context: PreparedModelRunContext, request: SampleRequest
    ) -> BayesiteCommand:
        """Build the Bayesite sample command for a prepared run directory."""
        draws_path = context.output_dir / "posterior.ndjson"
        engine_data_path = _materialize_engine_data(context.data_path, context.output_dir)
        return BayesiteCommand(
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
        )

    def run_sample(self, command: BayesiteCommand) -> int:
        """Execute a Bayesite sample command."""
        return run_bayesite_command(command)

    def build_prior_predictive_command(
        self, context: PreparedModelRunContext, request: PriorPredictiveRequest
    ) -> BayesiteCommand:
        """Build the Bayesite prior-predictive command for a prepared run directory."""
        output_path = context.output_dir / "prior_predictive.ndjson"
        engine_data_path = _materialize_engine_data(context.data_path, context.output_dir)
        return BayesiteCommand(
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
        )

    def run_prior_predictive(self, command: BayesiteCommand) -> int:
        """Execute a Bayesite prior-predictive command."""
        return run_bayesite_command(command)

    def build_simulate_command(
        self, context: PreparedModelRunContext, request: SimulateRequest, truth_path: Path
    ) -> BayesiteSimulateCommand:
        """Build the Bayesite simulate command for a prepared run directory."""
        canonical_output_path = context.output_dir / "simulated_data.json"
        native_output_path = _backend_dir(context.output_dir) / "simulated_data.json"
        engine_data_path = _materialize_engine_data(context.data_path, context.output_dir)
        engine_command = BayesiteCommand(
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
        )
        return BayesiteSimulateCommand(
            engine_command=engine_command,
            native_data_path=native_output_path,
            canonical_data_path=canonical_output_path,
        )

    def run_simulate(self, command: BayesiteSimulateCommand) -> int:
        """Execute Bayesite simulate and rewrite its output as canonical DataDoc."""
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

    def build_recover_command(
        self, context: PreparedModelScenarioContext, request: RecoverRequest
    ) -> BayesiteCommand:
        """Build the Bayesite recover command for a prepared run directory."""
        output_path = context.output_dir / "recovery.json"
        return BayesiteCommand(
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

    def run_recover(self, command: BayesiteCommand) -> int:
        """Execute a Bayesite recover command."""
        return run_bayesite_command(command)

    def build_sbc_command(
        self, context: PreparedModelScenarioContext, request: SbcRequest
    ) -> BayesiteCommand:
        """Build the Bayesite SBC command for a prepared run directory."""
        output_path = context.output_dir / "sbc.json"
        return BayesiteCommand(
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

    def run_sbc(self, command: BayesiteCommand) -> int:
        """Execute a Bayesite SBC command."""
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
