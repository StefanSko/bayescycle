"""In-process bayesjax backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

from bayescycle._integrations.descriptions import (
    InProcessSamplePlanDescription,
    InProcessSettingsPlanDescription,
)
from bayescycle._settings import PriorPredictiveSettings
from bayescycle._workflow.contexts import PlannedModelRunContext
from bayescycle._workflow.requests import PriorPredictiveRequest, SampleRequest
from bayescycle.backends.bayesjax.commands import (
    BayesjaxPriorPredictiveCommand,
    BayesjaxSampleCommand,
)
from bayescycle.backends.bayesjax.runner import (
    run_bayesjax_prior_predictive,
    run_bayesjax_sample,
)


@dataclass(frozen=True)
class BayesjaxSampleAction:
    """Planned in-process bayesjax sample action."""

    command: BayesjaxSampleCommand


@dataclass(frozen=True)
class BayesjaxPriorPredictiveAction:
    """Planned in-process bayesjax prior-predictive action."""

    command: BayesjaxPriorPredictiveCommand


type BayesjaxAction = BayesjaxSampleAction | BayesjaxPriorPredictiveAction
type BayesjaxExecutableCommand = BayesjaxSampleCommand | BayesjaxPriorPredictiveCommand


@dataclass(frozen=True)
class BayesjaxBackend:
    """In-process bayesjax backend capabilities."""

    @property
    def backend_id(self) -> str:
        """Return the stable backend identifier for run metadata."""
        return "bayesjax"

    def plan_sample_action(
        self, context: PlannedModelRunContext, request: SampleRequest
    ) -> BayesjaxSampleAction:
        """Plan the in-process bayesjax sample action."""
        return BayesjaxSampleAction(
            command=BayesjaxSampleCommand(
                loaded_model=context.loaded_model,
                ir_path=context.ir_path,
                data_path=context.data_path,
                draws_path=context.output_dir / "posterior.ndjson",
                settings=request.sampler.resolve_for_in_process(),
            )
        )

    def plan_prior_predictive_action(
        self, context: PlannedModelRunContext, request: PriorPredictiveRequest
    ) -> BayesjaxPriorPredictiveAction:
        """Plan the in-process bayesjax prior-predictive action."""
        return BayesjaxPriorPredictiveAction(
            command=BayesjaxPriorPredictiveCommand(
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
        self, action: BayesjaxAction
    ) -> InProcessSamplePlanDescription | InProcessSettingsPlanDescription:
        """Return bayesjax-owned plan description for a planned action."""
        if isinstance(action, BayesjaxSampleAction):
            return InProcessSamplePlanDescription(
                backend="bayesjax",
                sampler=action.command.settings.as_json(),
            )
        return InProcessSettingsPlanDescription(
            backend="bayesjax",
            settings=action.command.settings.as_json(),
        )

    @overload
    def materialize(self, action: BayesjaxSampleAction) -> BayesjaxSampleCommand: ...

    @overload
    def materialize(
        self, action: BayesjaxPriorPredictiveAction
    ) -> BayesjaxPriorPredictiveCommand: ...

    def materialize(self, action: BayesjaxAction) -> BayesjaxExecutableCommand:
        """Return the executable in-process command for an action."""
        return action.command

    def execute(self, command: BayesjaxExecutableCommand) -> int:
        """Execute an in-process bayesjax command."""
        match command:
            case BayesjaxSampleCommand():
                return run_bayesjax_sample(command)
            case BayesjaxPriorPredictiveCommand():
                return run_bayesjax_prior_predictive(command)
