"""In-process jaxstanv5 backend adapter."""

from __future__ import annotations

from dataclasses import dataclass

from bayescycle._commands import Jaxstanv5PriorPredictiveCommand, Jaxstanv5SampleCommand
from bayescycle._errors import WorkflowError
from bayescycle._inproc import run_jaxstanv5_prior_predictive, run_jaxstanv5_sample
from bayescycle._settings import PriorPredictiveSettings
from bayescycle._workflow import PreparedModelRunContext, PriorPredictiveRequest, SampleRequest


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
