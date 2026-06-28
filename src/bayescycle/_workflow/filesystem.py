"""Filesystem helpers for workflow materialization."""

from __future__ import annotations

import shutil
from pathlib import Path

from bayescycle._errors import WorkflowError


def _require_input_file(source: Path, label: str) -> Path:
    source_path = source.expanduser().resolve()
    if not source_path.is_file():
        raise WorkflowError(f"{label} file does not exist: {source_path}")
    return source_path


def _copy_required_input(source: Path, destination: Path, label: str) -> Path:
    source_path = _require_input_file(source, label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path != destination:
        shutil.copyfile(source_path, destination)
    return destination


def _require_run_artifacts(paths: tuple[Path, ...]) -> None:
    missing = tuple(path for path in paths if not path.is_file())
    if not missing:
        return
    run_dir = missing[0].parent
    names = ", ".join(path.name for path in missing)
    plural = "s" if len(missing) != 1 else ""
    raise WorkflowError(
        f"missing required run artifact{plural} in {run_dir}: {names}. "
        f"Run `bayescycle sample ... -o {run_dir}` first."
    )


def _reject_existing_output_artifacts(paths: tuple[Path, ...]) -> None:
    existing = tuple(path for path in paths if path.exists() or path.is_symlink())
    if not existing:
        return
    names = ", ".join(path.name for path in existing)
    plural = "s" if len(existing) != 1 else ""
    raise WorkflowError(
        f"output artifact{plural} already exists in {existing[0].parent}: {names}; "
        "choose a new run directory or remove the derived artifact intentionally"
    )


def _validate_output_dir(path: Path) -> None:
    if (path.exists() or path.is_symlink()) and not path.is_dir():
        raise WorkflowError(f"output path exists and is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise WorkflowError(f"output directory is not empty: {path}; choose a new run directory")


def _ensure_output_dir(path: Path) -> None:
    _validate_output_dir(path)
    path.mkdir(parents=True, exist_ok=True)
