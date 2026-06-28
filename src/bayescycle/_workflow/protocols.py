"""Backend capability protocols used by workflow operations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from bayescycle._integrations.descriptions import BackendPlanDescription
from bayescycle._workflow.contexts import (
    DiagnoseRunContext,
    PlannedModelRunContext,
    PlannedModelScenarioContext,
    PosteriorCheckRunContext,
    PosteriorPredictiveRunContext,
    RecoverCheckRunContext,
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


class BackendExecutor[CommandT](Protocol):
    """Backend capability for executing a materialized command."""

    def execute(self, command: CommandT) -> int:
        """Execute a typed backend command."""


class ActionBackend[ActionT, CommandT](BackendExecutor[CommandT], Protocol):
    """Backend capability shared by planned backend actions."""

    def describe(self, action: ActionT) -> BackendPlanDescription:
        """Return backend-owned plan description for an action."""

    def materialize(self, action: ActionT) -> CommandT:
        """Materialize backend-private inputs and return an executable command."""


class SampleBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for the sample workflow operation."""

    def plan_sample_action(
        self, context: PlannedModelRunContext, request: SampleRequest
    ) -> ActionT:
        """Plan a backend-private sample action from a planned run context."""


class PriorPredictiveBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for the prior-predictive workflow operation."""

    def plan_prior_predictive_action(
        self, context: PlannedModelRunContext, request: PriorPredictiveRequest
    ) -> ActionT:
        """Plan a backend-private prior-predictive action."""


class SimulateBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for the simulate workflow operation."""

    def plan_simulate_action(
        self, context: PlannedModelRunContext, request: SimulateRequest, truth_path: Path
    ) -> ActionT:
        """Plan a backend-private simulate action from a planned run context."""


class RecoverBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for the recover workflow operation."""

    def plan_recover_action(
        self, context: PlannedModelScenarioContext, request: RecoverRequest
    ) -> ActionT:
        """Plan a backend-private recover action from a planned scenario context."""


class SbcBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for the SBC workflow operation."""

    def plan_sbc_action(self, context: PlannedModelScenarioContext, request: SbcRequest) -> ActionT:
        """Plan a backend-private SBC action from a planned scenario context."""


class DiagnoseBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for an existing-run diagnose command."""

    def plan_diagnose_action(
        self, context: DiagnoseRunContext, request: DiagnoseRequest
    ) -> ActionT:
        """Plan a backend-private diagnose action."""


class PosteriorPredictiveBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for an existing-run posterior-predictive command."""

    def plan_posterior_predictive_action(
        self, context: PosteriorPredictiveRunContext, request: PosteriorPredictiveRequest
    ) -> ActionT:
        """Plan a backend-private posterior-predictive action."""


class PosteriorCheckBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for an existing-run posterior-check command."""

    def plan_posterior_check_action(
        self, context: PosteriorCheckRunContext, request: PosteriorCheckRequest
    ) -> ActionT:
        """Plan a backend-private posterior-check action."""


class RecoverCheckBackend[ActionT, CommandT](ActionBackend[ActionT, CommandT], Protocol):
    """Backend capability for an existing-run recover-check command."""

    def plan_recover_check_action(
        self, context: RecoverCheckRunContext, request: RecoverCheckRequest
    ) -> ActionT:
        """Plan a backend-private recover-check action."""
