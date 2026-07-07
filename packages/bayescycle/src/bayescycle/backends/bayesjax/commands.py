"""Typed in-process bayesjax command values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bayescycle._model_loader import LoadedModel
from bayescycle._run_artifacts.references import CanonicalDataArtifact, IrArtifact
from bayescycle._settings import ResolvedPriorPredictiveSettings, ResolvedSamplerSettings


@dataclass(frozen=True)
class BayesjaxPriorPredictiveCommand:
    """An in-process bayesjax prior-predictive command."""

    loaded_model: LoadedModel
    data_path: CanonicalDataArtifact
    output_path: Path
    settings: ResolvedPriorPredictiveSettings


@dataclass(frozen=True)
class BayesjaxSampleCommand:
    """An in-process bayesjax sampling command."""

    loaded_model: LoadedModel
    ir_path: IrArtifact
    data_path: CanonicalDataArtifact
    draws_path: Path
    settings: ResolvedSamplerSettings
