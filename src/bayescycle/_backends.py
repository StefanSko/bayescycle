"""Concrete backend capability implementations."""

from __future__ import annotations

from dataclasses import dataclass

from bayescycle._commands import BayesiteCommand, Jaxstanv5SampleCommand
from bayescycle._engine import run_bayesite_command
from bayescycle._errors import WorkflowError
from bayescycle._inproc import run_jaxstanv5_sample
from bayescycle._workflow import PreparedModelRunContext, SampleRequest


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
