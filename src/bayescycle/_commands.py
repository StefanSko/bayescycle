"""Typed backend command values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bayescycle._model_loader import LoadedModel
from bayescycle._settings import ResolvedPriorPredictiveSettings, ResolvedSamplerSettings

JsonObject = dict[str, object]


class DryRunCommand(Protocol):
    """Command values that can summarize themselves for ``--dry-run`` output."""

    def dry_run_fields(self) -> JsonObject: ...


@dataclass(frozen=True)
class BayesiteCommand:
    """A concrete Bayesite command ready for subprocess execution."""

    argv: tuple[str, ...]
    output_paths: tuple[Path, ...] = ()

    def dry_run_fields(self) -> JsonObject:
        """Return command fields for dry-run JSON output."""
        return {"engine_command": list(self.argv)}


@dataclass(frozen=True)
class BayesiteSimulateCommand:
    """Bayesite simulate command plus canonical-data postprocessing paths."""

    engine_command: BayesiteCommand
    native_data_path: Path
    canonical_data_path: Path

    def dry_run_fields(self) -> JsonObject:
        """Return command fields for dry-run JSON output."""
        fields = self.engine_command.dry_run_fields()
        fields["backend_simulated_data"] = str(self.native_data_path)
        return fields


@dataclass(frozen=True)
class Jaxstanv5PriorPredictiveCommand:
    """An in-process jaxstanv5 prior-predictive command."""

    loaded_model: LoadedModel
    data_path: Path
    output_path: Path
    settings: ResolvedPriorPredictiveSettings

    def dry_run_fields(self) -> JsonObject:
        """Return command fields for dry-run JSON output."""
        return {"backend": "jaxstanv5", "settings": self.settings.as_json()}


@dataclass(frozen=True)
class Jaxstanv5SampleCommand:
    """An in-process jaxstanv5 sampling command."""

    loaded_model: LoadedModel
    ir_path: Path
    data_path: Path
    draws_path: Path
    settings: ResolvedSamplerSettings

    def dry_run_fields(self) -> JsonObject:
        """Return command fields for dry-run JSON output."""
        return {"backend": "jaxstanv5", "sampler": self.settings.as_json()}
