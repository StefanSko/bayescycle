"""Append-only bayescycle run metadata artifact."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

RUN_METADATA_FORMAT = "bayescycle.run.v1"


class RunMetadataError(ValueError):
    """Raised when run metadata cannot be parsed."""


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
class RunMetadataSetting:
    """Replay-relevant CLI setting recorded in run metadata."""

    name: str
    value: str


@dataclass(frozen=True)
class RunMetadataEngine:
    """Bayesite engine provenance recorded for a materialized run.

    ``kind`` records how bayescycle resolved the engine executable:
    ``"explicit"`` (passed via ``--engine``), ``"system"`` (found on
    ``PATH``), or ``"provisioned"`` (auto-downloaded into the bayescycle
    cache). ``version`` and ``sha256`` are omitted from the wire format when
    unknown; ``sha256`` is only ever set when the engine came from
    auto-provisioning, since bayescycle does not hash arbitrary user-supplied
    or PATH-resolved binaries.
    """

    kind: str
    path: str
    version: str | None = None
    sha256: str | None = None

    def as_json(self) -> dict[str, object]:
        document: dict[str, object] = {"kind": self.kind, "path": self.path}
        if self.version is not None:
            document["version"] = self.version
        if self.sha256 is not None:
            document["sha256"] = self.sha256
        return document


@dataclass(frozen=True)
class RunMetadata:
    """Append-only metadata for a prepared run directory."""

    kind: str
    backend: str
    model: RunMetadataModel
    inputs: tuple[RunMetadataInput, ...]
    outputs: tuple[RunMetadataOutput, ...]
    settings: tuple[RunMetadataSetting, ...] = ()
    backend_extra_args: tuple[str, ...] = ()
    engine: RunMetadataEngine | None = None

    def as_json(self, run_dir: Path) -> dict[str, object]:
        document: dict[str, object] = {
            "format": RUN_METADATA_FORMAT,
            "kind": self.kind,
            "backend": self.backend,
            "settings": {entry.name: entry.value for entry in self.settings},
            "model": self.model.as_json(run_dir),
            "inputs": [entry.as_json(run_dir) for entry in self.inputs],
            "outputs": [entry.as_json(run_dir) for entry in self.outputs],
        }
        if self.backend_extra_args:
            document["backend_options"] = {"extra_args": list(self.backend_extra_args)}
        if self.engine is not None:
            document["engine"] = self.engine.as_json()
        return document


@dataclass(frozen=True)
class RecordedRunMetadataModel:
    """Model source loaded from an existing run metadata document."""

    name: str
    source_path: Path
    source_sha256: str
    ir_path: Path


@dataclass(frozen=True)
class RecordedRunMetadataInput:
    """Input source loaded from an existing run metadata document."""

    role: str
    source_path: Path
    source_sha256: str
    path: Path
    artifact_format: str | None


@dataclass(frozen=True)
class RecordedRunMetadataOutput:
    """Declared output loaded from an existing run metadata document."""

    role: str
    path: Path
    artifact_format: str | None


@dataclass(frozen=True)
class RecordedRunMetadataEngine:
    """Bayesite engine provenance loaded from an existing run metadata document."""

    kind: str
    path: str
    version: str | None
    sha256: str | None


@dataclass(frozen=True)
class RecordedRunMetadata:
    """Parsed append-only metadata for an existing run directory."""

    kind: str
    backend: str
    model: RecordedRunMetadataModel
    inputs: tuple[RecordedRunMetadataInput, ...]
    outputs: tuple[RecordedRunMetadataOutput, ...]
    settings: tuple[RunMetadataSetting, ...]
    backend_extra_args: tuple[str, ...]
    engine: RecordedRunMetadataEngine | None = None

    def setting(self, name: str) -> str | None:
        """Return a recorded setting value by name."""
        for setting in self.settings:
            if setting.name == name:
                return setting.value
        return None


def write_run_metadata(run_dir: Path, metadata: RunMetadata) -> None:
    """Write append-only run metadata."""
    with (run_dir / "run.json").open("x", encoding="utf-8") as f:
        json.dump(metadata.as_json(run_dir), f, indent=2)
        f.write("\n")


def read_run_metadata(run_dir: Path) -> RecordedRunMetadata:
    """Read and validate append-only run metadata from a run directory."""
    path = run_dir / "run.json"
    try:
        raw_document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunMetadataError(f"run metadata does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunMetadataError(f"invalid run metadata JSON in {path}: {exc.msg}") from exc
    return _parse_recorded_run_metadata(raw_document)


def _parse_recorded_run_metadata(raw_document: object) -> RecordedRunMetadata:
    document = _json_object(raw_document, "run metadata")
    metadata_format = _json_string(document.get("format"), "format")
    if metadata_format != RUN_METADATA_FORMAT:
        raise RunMetadataError(
            f"unsupported run metadata format: {metadata_format}; expected {RUN_METADATA_FORMAT}"
        )
    if "settings" not in document:
        raise RunMetadataError(
            "run metadata is not replay-capable: missing settings; "
            "re-run with a newer bayescycle before using replay"
        )
    return RecordedRunMetadata(
        kind=_json_string(document.get("kind"), "kind"),
        backend=_json_string(document.get("backend"), "backend"),
        model=_parse_recorded_model(document.get("model")),
        inputs=tuple(
            _parse_recorded_input(entry, index)
            for index, entry in enumerate(_json_array(document.get("inputs"), "inputs"))
        ),
        outputs=tuple(
            _parse_recorded_output(entry, index)
            for index, entry in enumerate(_json_array(document.get("outputs"), "outputs"))
        ),
        settings=_parse_recorded_settings(document.get("settings")),
        backend_extra_args=_parse_recorded_backend_extra_args(document.get("backend_options", {})),
        engine=_parse_recorded_engine(document.get("engine")),
    )


def _parse_recorded_model(value: object) -> RecordedRunMetadataModel:
    document = _json_object(value, "model")
    return RecordedRunMetadataModel(
        name=_json_string(document.get("name"), "model.name"),
        source_path=Path(_json_string(document.get("source_path"), "model.source_path")),
        source_sha256=_json_string(document.get("sha256"), "model.sha256"),
        ir_path=Path(_json_string(document.get("ir_path"), "model.ir_path")),
    )


def _parse_recorded_input(value: object, index: int) -> RecordedRunMetadataInput:
    label = f"inputs[{index}]"
    document = _json_object(value, label)
    return RecordedRunMetadataInput(
        role=_json_string(document.get("role"), f"{label}.role"),
        source_path=Path(_json_string(document.get("source_path"), f"{label}.source_path")),
        source_sha256=_json_string(document.get("sha256"), f"{label}.sha256"),
        path=Path(_json_string(document.get("path"), f"{label}.path")),
        artifact_format=_optional_json_string(document.get("format"), f"{label}.format"),
    )


def _parse_recorded_output(value: object, index: int) -> RecordedRunMetadataOutput:
    label = f"outputs[{index}]"
    document = _json_object(value, label)
    return RecordedRunMetadataOutput(
        role=_json_string(document.get("role"), f"{label}.role"),
        path=Path(_json_string(document.get("path"), f"{label}.path")),
        artifact_format=_optional_json_string(document.get("format"), f"{label}.format"),
    )


def _parse_recorded_settings(value: object) -> tuple[RunMetadataSetting, ...]:
    document = _json_object(value, "settings")
    return tuple(
        RunMetadataSetting(name=name, value=_json_string(setting_value, f"settings.{name}"))
        for name, setting_value in document.items()
    )


def _parse_recorded_backend_extra_args(value: object) -> tuple[str, ...]:
    document = _json_object(value, "backend_options")
    if "extra_args" not in document:
        return ()
    return _json_string_array(document.get("extra_args"), "backend_options.extra_args")


def _parse_recorded_engine(value: object) -> RecordedRunMetadataEngine | None:
    if value is None:
        return None
    document = _json_object(value, "engine")
    return RecordedRunMetadataEngine(
        kind=_json_string(document.get("kind"), "engine.kind"),
        path=_json_string(document.get("path"), "engine.path"),
        version=_optional_json_string(document.get("version"), "engine.version"),
        sha256=_optional_json_string(document.get("sha256"), "engine.sha256"),
    )


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RunMetadataError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _json_array(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise RunMetadataError(f"{label} must be a JSON array")
    return tuple(value)


def _json_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise RunMetadataError(f"{label} must be a JSON string")
    return value


def _json_string_array(value: object, label: str) -> tuple[str, ...]:
    return tuple(
        _json_string(entry, f"{label}[{index}]")
        for index, entry in enumerate(_json_array(value, label))
    )


def _optional_json_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _json_string(value, label)


def sha256_uri(path: Path) -> str:
    """Return a sha256 URI for the current file contents."""
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def relative_run_path(run_dir: Path, path: Path) -> str:
    """Return path relative to run_dir when possible."""
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)
