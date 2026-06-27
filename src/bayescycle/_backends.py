"""Concrete backend capability implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bayescycle._commands import (
    BayesiteCommand,
    Jaxstanv5PriorPredictiveCommand,
    Jaxstanv5SampleCommand,
)
from bayescycle._engine import run_bayesite_command
from bayescycle._errors import WorkflowError
from bayescycle._inproc import run_jaxstanv5_prior_predictive, run_jaxstanv5_sample
from bayescycle._settings import PriorPredictiveSettings
from bayescycle._workflow import (
    PreparedModelRunContext,
    PreparedModelScenarioContext,
    PriorPredictiveRequest,
    RecoverRequest,
    SampleRequest,
    SbcRequest,
    SimulateRequest,
)


@dataclass(frozen=True)
class BayesiteBackend:
    """Bayesite subprocess backend capabilities."""

    engine: str

    def build_sample_command(
        self, context: PreparedModelRunContext, request: SampleRequest
    ) -> BayesiteCommand:
        """Build the Bayesite sample command for a prepared run directory."""
        draws_path = context.output_dir / "posterior.ndjson"
        return BayesiteCommand(
            argv=(
                self.engine,
                "sample",
                "--model",
                str(context.ir_path),
                "--data",
                str(context.data_path),
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
        return BayesiteCommand(
            argv=(
                self.engine,
                "prior-predictive",
                "--model",
                str(context.ir_path),
                "--data",
                str(context.data_path),
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
    ) -> BayesiteCommand:
        """Build the Bayesite simulate command for a prepared run directory."""
        output_path = context.output_dir / "simulated_data.json"
        return BayesiteCommand(
            argv=(
                self.engine,
                "simulate",
                "--model",
                str(context.ir_path),
                "--data",
                str(context.data_path),
                "--truth",
                str(truth_path),
                *_optional_arg("--seed", request.seed),
                "--out",
                str(output_path),
                *request.engine_args,
            ),
            output_paths=(output_path,),
        )

    def run_simulate(self, command: BayesiteCommand) -> int:
        """Execute a Bayesite simulate command."""
        return run_bayesite_command(command)

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


@dataclass(frozen=True)
class Jaxstanv5Backend:
    """In-process jaxstanv5 backend capabilities."""

    def build_sample_command(
        self, context: PreparedModelRunContext, request: SampleRequest
    ) -> Jaxstanv5SampleCommand:
        """Build the in-process jaxstanv5 sample command for a prepared run directory."""
        if request.engine_args:
            raise WorkflowError(
                "engine passthrough after -- is only supported for --backend bayesite"
            )
        return Jaxstanv5SampleCommand(
            loaded_model=context.loaded_model,
            ir_path=context.ir_path,
            data_path=context.data_path,
            draws_path=context.output_dir / "posterior.ndjson",
            settings=request.sampler.resolve_for_in_process(),
        )

    def run_sample(self, command: Jaxstanv5SampleCommand) -> int:
        """Execute an in-process jaxstanv5 sample command."""
        return run_jaxstanv5_sample(command)

    def build_prior_predictive_command(
        self, context: PreparedModelRunContext, request: PriorPredictiveRequest
    ) -> Jaxstanv5PriorPredictiveCommand:
        """Build the in-process jaxstanv5 prior-predictive command."""
        if request.engine_args:
            raise WorkflowError(
                "engine passthrough after -- is only supported for --backend bayesite"
            )
        return Jaxstanv5PriorPredictiveCommand(
            loaded_model=context.loaded_model,
            data_path=context.data_path,
            output_path=context.output_dir / "prior_predictive.ndjson",
            settings=PriorPredictiveSettings(
                seed=request.seed,
                draws=request.draws,
            ).resolve_for_in_process(),
        )

    def run_prior_predictive(self, command: Jaxstanv5PriorPredictiveCommand) -> int:
        """Execute an in-process jaxstanv5 prior-predictive command."""
        return run_jaxstanv5_prior_predictive(command)


def _optional_arg(flag: str, value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (flag, value)
