"""Workflow phases for preparing and launching a Bayesite run."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, TypedDict

from jaxstanv5.ir import canonical_bytes
from jaxstanv5.model import ModelMeta

from bayescycle._dims import dims_sidecar_for_model, write_dims_sidecar
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
    dims_path: Path | None
    output_dir: Path
    engine_command: EngineCommand


class DryRunDocument(TypedDict):
    """JSON document printed for ``--dry-run``."""

    model: str
    ir: str
    data: str
    draws: str
    output: str
    engine_command: list[str]
    dims: NotRequired[str]


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

    dims_path = _write_optional_dims_sidecar(output_dir, loaded_model.model_cls, loaded_model.meta)

    draws_path = output_dir / "posterior.ndjson"
    command = EngineCommand(
        argv=(
            request.engine,
            "sample",
            "--model",
            str(ir_path),
            "--data",
            str(run_data_path),
            *request.engine_args,
        ),
        stdout_path=draws_path,
    )
    return PreparedSampleRun(
        model_name=loaded_model.name,
        ir_path=ir_path,
        data_path=run_data_path,
        dims_path=dims_path,
        output_dir=output_dir,
        engine_command=command,
    )


def dry_run_document(run: PreparedSampleRun) -> DryRunDocument:
    """Return a serializable dry-run summary."""
    document: DryRunDocument = {
        "model": run.model_name,
        "ir": str(run.ir_path),
        "data": str(run.data_path),
        "draws": str(run.engine_command.stdout_path),
        "output": str(run.output_dir),
        "engine_command": list(run.engine_command.argv),
    }
    if run.dims_path is not None:
        document["dims"] = str(run.dims_path)
    return document


def _write_optional_dims_sidecar(
    output_dir: Path, model_cls: type[object], meta: ModelMeta
) -> Path | None:
    try:
        sidecar = dims_sidecar_for_model(model_cls, meta)
    except ValueError as exc:
        raise WorkflowError(f"invalid model dimension metadata: {exc}") from exc
    dims_path = output_dir / "dims.json"
    if sidecar is None:
        dims_path.unlink(missing_ok=True)
        return None
    write_dims_sidecar(dims_path, sidecar)
    return dims_path


def _ensure_output_dir(path: Path, *, force: bool) -> None:
    if path.exists() and not path.is_dir():
        raise WorkflowError(f"output path exists and is not a directory: {path}")
    if path.exists() and not force and any(path.iterdir()):
        raise WorkflowError(f"output directory is not empty: {path}; pass --force to reuse it")
    path.mkdir(parents=True, exist_ok=True)
