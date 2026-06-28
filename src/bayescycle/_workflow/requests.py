"""Typed workflow requests normalized from CLI input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bayescycle._settings import SamplerSettings


@dataclass(frozen=True)
class SampleRequest:
    """Loose CLI sampling input normalized into a typed request."""

    model_path: Path
    data_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    sampler: SamplerSettings
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class PriorPredictiveRequest:
    """Loose CLI prior-predictive input normalized into a typed request."""

    model_path: Path
    data_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    seed: str | None
    draws: str | None
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class SimulateRequest:
    """Loose CLI simulation input normalized into a typed request."""

    model_path: Path
    data_path: Path
    truth_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    seed: str | None
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class RecoverRequest:
    """Loose CLI single-scenario recovery input normalized into a typed request."""

    model_path: Path
    scenario_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class SbcRequest:
    """Loose CLI SBC input normalized into a typed request."""

    model_path: Path
    scenario_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    replicates: str | None
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class DiagnoseRequest:
    """Request to run diagnostics for an existing run directory."""

    run_dir: Path


@dataclass(frozen=True)
class PosteriorPredictiveRequest:
    """Request to run posterior predictive generation for an existing run directory."""

    run_dir: Path
    seed: str


@dataclass(frozen=True)
class PosteriorCheckRequest:
    """Request to run posterior checks for an existing run directory."""

    run_dir: Path
    seed: str | None
    backend: str
    engine_args: tuple[str, ...]


@dataclass(frozen=True)
class RecoverCheckRequest:
    """Request to run recovery checks for an existing run directory."""

    run_dir: Path
    truth_path: Path
    targets_path: Path | None
    interval: str | None
    backend: str
    engine_args: tuple[str, ...]
