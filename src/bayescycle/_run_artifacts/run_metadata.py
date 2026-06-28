"""Append-only bayescycle run metadata artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunMetadataModel:
    """Model source recorded in a run metadata document."""

    name: str
    source_path: Path
    source_sha256: str
    ir_path: Path

    def as_json(self, run_dir: Path) -> dict[str, object]:
        return {
            "name": self.name,
            "source_path": str(self.source_path),
            "sha256": self.source_sha256,
            "ir_path": relative_run_path(run_dir, self.ir_path),
        }


@dataclass(frozen=True)
class RunMetadataInput:
    """Input source recorded in a run metadata document."""

    role: str
    source_path: Path
    source_sha256: str
    materialized_path: Path
    artifact_format: str | None = None

    def as_json(self, run_dir: Path) -> dict[str, object]:
        document: dict[str, object] = {
            "role": self.role,
            "source_path": str(self.source_path),
            "sha256": self.source_sha256,
            "path": relative_run_path(run_dir, self.materialized_path),
        }
        if self.artifact_format is not None:
            document["format"] = self.artifact_format
        return document


@dataclass(frozen=True)
class RunMetadataOutput:
    """Declared run output recorded in a run metadata document."""

    role: str
    path: Path
    artifact_format: str | None = None

    def as_json(self, run_dir: Path) -> dict[str, object]:
        document: dict[str, object] = {
            "role": self.role,
            "path": relative_run_path(run_dir, self.path),
        }
        if self.artifact_format is not None:
            document["format"] = self.artifact_format
        return document


@dataclass(frozen=True)
class RunMetadata:
    """Append-only metadata for a prepared run directory."""

    kind: str
    backend: str
    model: RunMetadataModel
    inputs: tuple[RunMetadataInput, ...]
    outputs: tuple[RunMetadataOutput, ...]

    def as_json(self, run_dir: Path) -> dict[str, object]:
        return {
            "format": "bayescycle.run.v1",
            "kind": self.kind,
            "backend": self.backend,
            "model": self.model.as_json(run_dir),
            "inputs": [entry.as_json(run_dir) for entry in self.inputs],
            "outputs": [entry.as_json(run_dir) for entry in self.outputs],
        }


def write_run_metadata(run_dir: Path, metadata: RunMetadata) -> None:
    """Write append-only run metadata."""
    with (run_dir / "run.json").open("x", encoding="utf-8") as f:
        json.dump(metadata.as_json(run_dir), f, indent=2)
        f.write("\n")


def sha256_uri(path: Path) -> str:
    """Return a sha256 URI for the current file contents."""
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def relative_run_path(run_dir: Path, path: Path) -> str:
    """Return path relative to run_dir when possible."""
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)
