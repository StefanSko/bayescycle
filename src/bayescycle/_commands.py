"""Typed backend command values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bayescycle._model_loader import LoadedModel
from bayescycle._run_artifacts.references import CanonicalDataArtifact, IrArtifact
from bayescycle._settings import ResolvedPriorPredictiveSettings, ResolvedSamplerSettings


@dataclass(frozen=True)
class BayesiteCommand:
    """A concrete Bayesite command ready for subprocess execution."""

    argv: tuple[str, ...]
    output_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Jaxstanv5PriorPredictiveCommand:
    """An in-process jaxstanv5 prior-predictive command."""

    loaded_model: LoadedModel
    data_path: CanonicalDataArtifact
    output_path: Path
    settings: ResolvedPriorPredictiveSettings


@dataclass(frozen=True)
class Jaxstanv5SampleCommand:
    """An in-process jaxstanv5 sampling command."""

    loaded_model: LoadedModel
    ir_path: IrArtifact
    data_path: CanonicalDataArtifact
    draws_path: Path
    settings: ResolvedSamplerSettings
