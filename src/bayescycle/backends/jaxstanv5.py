"""In-process jaxstanv5 backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

from bayescycle._commands import Jaxstanv5PriorPredictiveCommand, Jaxstanv5SampleCommand
from bayescycle._errors import WorkflowError
from bayescycle._inproc import run_jaxstanv5_prior_predictive, run_jaxstanv5_sample
from bayescycle._settings import PriorPredictiveSettings
from bayescycle._workflow import (
    InProcessSamplePlanDescription,
    InProcessSettingsPlanDescription,
    PlannedModelRunContext,
    PriorPredictiveRequest,
    SampleRequest,
)


@dataclass(frozen=True)
class Jaxstanv5SampleAction:
    """Planned in-process jaxstanv5 sample action."""

    command: Jaxstanv5SampleCommand


@dataclass(frozen=True)
class Jaxstanv5PriorPredictiveAction:
    """Planned in-process jaxstanv5 prior-predictive action."""

    command: Jaxstanv5PriorPredictiveCommand


type Jaxstanv5Action = Jaxstanv5SampleAction | Jaxstanv5PriorPredictiveAction
type Jaxstanv5ExecutableCommand = Jaxstanv5SampleCommand | Jaxstanv5PriorPredictiveCommand


@dataclass(frozen=True)
class Jaxstanv5Backend:
    """In-process jaxstanv5 backend capabilities."""

    def plan_sample_action(
        self, context: PlannedModelRunContext, request: SampleRequest
    ) -> Jaxstanv5SampleAction:
        """Plan the in-process jaxstanv5 sample action."""
        if request.engine_args:
            raise WorkflowError(
                "engine passthrough after -- is only supported for --backend bayesite"
            )
        return Jaxstanv5SampleAction(
            command=Jaxstanv5SampleCommand(
                loaded_model=context.loaded_model,
                ir_path=context.ir_path,
                data_path=context.data_path,
                draws_path=context.output_dir / "posterior.ndjson",
                settings=request.sampler.resolve_for_in_process(),
            )
        )

    def plan_prior_predictive_action(
        self, context: PlannedModelRunContext, request: PriorPredictiveRequest
    ) -> Jaxstanv5PriorPredictiveAction:
        """Plan the in-process jaxstanv5 prior-predictive action."""
        if request.engine_args:
            raise WorkflowError(
                "engine passthrough after -- is only supported for --backend bayesite"
            )
        return Jaxstanv5PriorPredictiveAction(
            command=Jaxstanv5PriorPredictiveCommand(
                loaded_model=context.loaded_model,
                data_path=context.data_path,
                output_path=context.output_dir / "prior_predictive.ndjson",
                settings=PriorPredictiveSettings(
                    seed=request.seed,
                    draws=request.draws,
                ).resolve_for_in_process(),
            )
        )

    def describe(
        self, action: Jaxstanv5Action
    ) -> InProcessSamplePlanDescription | InProcessSettingsPlanDescription:
        """Return jaxstanv5-owned plan description for a planned action."""
        if isinstance(action, Jaxstanv5SampleAction):
            return InProcessSamplePlanDescription(
                backend="jaxstanv5",
                sampler=action.command.settings.as_json(),
            )
        return InProcessSettingsPlanDescription(
            backend="jaxstanv5",
            settings=action.command.settings.as_json(),
        )

    @overload
    def materialize(self, action: Jaxstanv5SampleAction) -> Jaxstanv5SampleCommand: ...

    @overload
    def materialize(
        self, action: Jaxstanv5PriorPredictiveAction
    ) -> Jaxstanv5PriorPredictiveCommand: ...

    def materialize(self, action: Jaxstanv5Action) -> Jaxstanv5ExecutableCommand:
        """Return the executable in-process command for an action."""
        return action.command

    def execute(self, command: Jaxstanv5ExecutableCommand) -> int:
        """Execute an in-process jaxstanv5 command."""
        match command:
            case Jaxstanv5SampleCommand():
                return run_jaxstanv5_sample(command)
            case Jaxstanv5PriorPredictiveCommand():
                return run_jaxstanv5_prior_predictive(command)
