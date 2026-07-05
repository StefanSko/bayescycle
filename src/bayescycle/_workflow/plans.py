"""Workflow run plan dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bayescycle._run_artifacts.references import CanonicalDataArtifact
from bayescycle._run_artifacts.run_metadata import RunMetadataSetting
from bayescycle._workflow.contexts import PlannedModelRunContext, PlannedModelScenarioContext


@dataclass(frozen=True)
class SampleRunPlan[ActionT]:
    """A sample run plan with workflow-owned paths and backend action."""

    context: PlannedModelRunContext
    draws_path: Path
    backend: str
    settings: tuple[RunMetadataSetting, ...]
    action: ActionT


@dataclass(frozen=True)
class PriorPredictiveRunPlan[ActionT]:
    """A prior-predictive run plan with workflow-owned paths and backend action."""

    context: PlannedModelRunContext
    prior_predictive_path: Path
    backend: str
    settings: tuple[RunMetadataSetting, ...]
    action: ActionT


@dataclass(frozen=True)
class SimulateRunPlan[ActionT]:
    """A simulation run plan with workflow-owned paths and backend action."""

    context: PlannedModelRunContext
    truth_source_path: Path
    truth_path: Path
    simulated_data_path: CanonicalDataArtifact
    backend: str
    settings: tuple[RunMetadataSetting, ...]
    action: ActionT


@dataclass(frozen=True)
class RecoverRunPlan[ActionT]:
    """A single-scenario recovery run plan with workflow-owned paths and action."""

    context: PlannedModelScenarioContext
    recovery_path: Path
    backend: str
    action: ActionT


@dataclass(frozen=True)
class SbcRunPlan[ActionT]:
    """An SBC run plan with workflow-owned paths and backend action."""

    context: PlannedModelScenarioContext
    sbc_path: Path
    backend: str
    settings: tuple[RunMetadataSetting, ...]
    action: ActionT


@dataclass(frozen=True)
class RunDirectoryCommandPlan[ActionT]:
    """An existing-run command plan with backend action."""

    run_dir: Path
    output_path: Path
    action: ActionT
