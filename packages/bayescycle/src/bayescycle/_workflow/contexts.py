"""Workflow run contexts planned before durable writes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bayescycle._model_loader import LoadedModel
from bayescycle._run_artifacts.canonical_data import DataDoc
from bayescycle._run_artifacts.references import CanonicalDataArtifact, IrArtifact


@dataclass(frozen=True)
class PlannedModelRunContext:
    """Planned model/data run-directory inputs before durable writes."""

    model_name: str
    model_source_path: Path
    model_source_sha256: str
    loaded_model: LoadedModel
    ir_path: IrArtifact
    data_source_path: Path
    data_source_sha256: str
    data_path: CanonicalDataArtifact
    dims_path: Path | None
    output_dir: Path
    data_doc: DataDoc


@dataclass(frozen=True)
class PlannedModelScenarioContext:
    """Planned model/scenario run-directory inputs before durable writes."""

    model_name: str
    model_source_path: Path
    model_source_sha256: str
    loaded_model: LoadedModel
    ir_path: IrArtifact
    scenario_source_path: Path
    scenario_source_sha256: str
    scenario_path: Path
    dims_path: Path | None
    output_dir: Path


@dataclass(frozen=True)
class DiagnoseRunContext:
    """Validated existing-run paths for a diagnose command."""

    run_dir: Path
    fit_path: Path
    output_path: Path


@dataclass(frozen=True)
class PosteriorPredictiveRunContext:
    """Validated existing-run paths for posterior predictive generation."""

    run_dir: Path
    model_path: IrArtifact
    data_path: CanonicalDataArtifact
    fit_path: Path
    output_path: Path


@dataclass(frozen=True)
class PosteriorCheckRunContext:
    """Validated existing-run paths for posterior checks."""

    run_dir: Path
    model_path: IrArtifact
    data_path: CanonicalDataArtifact
    fit_path: Path
    output_path: Path


@dataclass(frozen=True)
class RecoverCheckRunContext:
    """Validated existing-run paths for recovery checks."""

    run_dir: Path
    fit_path: Path
    truth_path: Path
    targets_path: Path | None
    output_path: Path
