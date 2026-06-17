"""Workflow phases for preparing and launching a Bayesite run."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from jaxstanv5.ir import canonical_bytes

from bayescycle._engine import EngineCommand
from bayescycle._model_loader import load_model


class WorkflowError(RuntimeError):
    """Raised when a workflow request cannot be prepared."""


@dataclass(frozen=True)
class SampleRequest:
    """Loose CLI sampling input normalized into a typed request."""

    model_path: Path
    data_path: Path
    output_dir: Path
    model_name: str | None
    engine: str
    engine_args: tuple[str, ...]
    force: bool


@dataclass(frozen=True)
class PreparedSampleRun:
    """A run directory and engine command produced from a sample request."""

    model_name: str
    ir_path: Path
    data_path: Path
    output_dir: Path
    engine_command: EngineCommand


class DryRunDocument(TypedDict):
    """JSON document printed for ``--dry-run``."""

    model: str
    ir: str
    data: str
    output: str
    engine_command: list[str]


def prepare_sample_run(request: SampleRequest) -> PreparedSampleRun:
    """Load model metadata, write IR/data files, and build the engine command."""
    output_dir = request.output_dir.expanduser().resolve()
    _ensure_output_dir(output_dir, force=request.force)

    source_data_path = request.data_path.expanduser().resolve()
    if not source_data_path.is_file():
        raise WorkflowError(f"data file does not exist: {source_data_path}")

    loaded_model = load_model(request.model_path, request.model_name)

    ir_path = output_dir / "model.ir.json"
    ir_path.write_bytes(canonical_bytes(loaded_model.meta))

    run_data_path = output_dir / "data.json"
    if source_data_path != run_data_path:
        shutil.copyfile(source_data_path, run_data_path)

    command = EngineCommand(
        argv=(
            request.engine,
            "sample",
            str(ir_path),
            "--data",
            str(run_data_path),
            "-o",
            str(output_dir),
            *request.engine_args,
        )
    )
    return PreparedSampleRun(
        model_name=loaded_model.name,
        ir_path=ir_path,
        data_path=run_data_path,
        output_dir=output_dir,
        engine_command=command,
    )


def dry_run_document(run: PreparedSampleRun) -> DryRunDocument:
    """Return a serializable dry-run summary."""
    return {
        "model": run.model_name,
        "ir": str(run.ir_path),
        "data": str(run.data_path),
        "output": str(run.output_dir),
        "engine_command": list(run.engine_command.argv),
    }


def _ensure_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise WorkflowError(f"output path exists and is not a directory: {path}")
    if path.exists() and not force and any(path.iterdir()):
        raise WorkflowError(f"output directory is not empty: {path}; pass --force to reuse it")
    path.mkdir(parents=True, exist_ok=True)
