"""Workflow phases for preparing and launching backend runs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, cast

from jaxstanv5.ir import canonical_bytes
from jaxstanv5.model import ModelMeta

from bayescycle._commands import BayesiteCommand, DryRunCommand
from bayescycle._dims import dims_sidecar_for_model, write_dims_sidecar
from bayescycle._errors import WorkflowError
from bayescycle._model_loader import LoadedModel, load_model
from bayescycle._settings import SamplerSettings


@dataclass(frozen=True)
class SampleRequest:
    """Loose CLI sampling input normalized into a typed request."""

    model_path: Path
    data_path: Path
    output_dir: Path
    model_name: str | None
    backend: str
    engine: str
    sampler: SamplerSettings
    engine_args: tuple[str, ...]
    force: bool


@dataclass(frozen=True)
class PreparedModelRunContext:
    """Prepared model/data files shared by model-level workflow commands."""

    model_name: str
    loaded_model: LoadedModel
    ir_path: Path
    data_path: Path
    dims_path: Path | None
    output_dir: Path


class SampleBackend[CommandT: DryRunCommand](Protocol):
    """Backend capability for the sample workflow operation."""

    def build_sample_command(
        self, context: PreparedModelRunContext, request: SampleRequest
    ) -> CommandT:
        """Build a typed sample command from a prepared run context."""

    def run_sample(self, command: CommandT) -> int:
        """Execute a typed sample command."""


@dataclass(frozen=True)
class PreparedSampleRun[CommandT: DryRunCommand]:
    """A run directory and sample command produced from a sample request."""

    model_name: str
    ir_path: Path
    data_path: Path
    dims_path: Path | None
    output_dir: Path
    draws_path: Path
    command: CommandT


@dataclass(frozen=True)
class DiagnoseRequest:
    """Request to run diagnostics for an existing run directory."""

    run_dir: Path
    engine: str


@dataclass(frozen=True)
class PosteriorPredictiveRequest:
    """Request to run posterior predictive generation for an existing run directory."""

    run_dir: Path
    engine: str
    seed: str


@dataclass(frozen=True)
class PreparedRunDirectoryCommand:
    """A follow-up command produced from an existing run directory."""

    run_dir: Path
    output_path: Path
    command: BayesiteCommand


class DryRunDocument(TypedDict):
    """JSON document printed for sample ``--dry-run``."""

    model: str
    ir: str
    data: str
    draws: str
    output: str
    engine_command: NotRequired[list[str]]
    backend: NotRequired[str]
    sampler: NotRequired[dict[str, int | float]]
    dims: NotRequired[str]


class RunCommandDryRunDocument(TypedDict):
    """JSON document printed for run-directory command ``--dry-run``."""

    run: str
    output: str
    engine_command: list[str]


def prepare_sample_run[CommandT: DryRunCommand](
    request: SampleRequest, backend: SampleBackend[CommandT]
) -> PreparedSampleRun[CommandT]:
    """Load model metadata, write IR/data files, and build a typed sample command."""
    _validate_sample_backend(request.backend)
    output_dir = request.output_dir.expanduser().resolve()
    draws_path = output_dir / "posterior.ndjson"
    _reject_reserved_engine_args(request.engine_args, draws_path)
    context = prepare_model_run_context(
        model_path=request.model_path,
        model_name=request.model_name,
        data_path=request.data_path,
        output_dir=output_dir,
        force=request.force,
    )
    command = backend.build_sample_command(context, request)
    return PreparedSampleRun(
        model_name=context.model_name,
        ir_path=context.ir_path,
        data_path=context.data_path,
        dims_path=context.dims_path,
        output_dir=context.output_dir,
        draws_path=draws_path,
        command=command,
    )


def prepare_model_run_context(
    *,
    model_path: Path,
    model_name: str | None,
    data_path: Path,
    output_dir: Path,
    force: bool,
) -> PreparedModelRunContext:
    """Prepare the shared model/data run directory inputs for model-level commands."""
    _ensure_output_dir(output_dir, force=force)

    source_data_path = data_path.expanduser().resolve()
    if not source_data_path.is_file():
        raise WorkflowError(f"data file does not exist: {source_data_path}")

    loaded_model = load_model(model_path, model_name)

    ir_path = output_dir / "model.ir.json"
    ir_path.write_bytes(canonical_bytes(loaded_model.meta))

    run_data_path = output_dir / "data.json"
    if source_data_path != run_data_path:
        shutil.copyfile(source_data_path, run_data_path)

    dims_path = _write_optional_dims_sidecar(output_dir, loaded_model.model_cls, loaded_model.meta)

    return PreparedModelRunContext(
        model_name=loaded_model.name,
        loaded_model=loaded_model,
        ir_path=ir_path,
        data_path=run_data_path,
        dims_path=dims_path,
        output_dir=output_dir,
    )


def dry_run_document[CommandT: DryRunCommand](
    run: PreparedSampleRun[CommandT],
) -> DryRunDocument:
    """Return a serializable sample dry-run summary."""
    document: dict[str, object] = {
        "model": run.model_name,
        "ir": str(run.ir_path),
        "data": str(run.data_path),
        "draws": str(run.draws_path),
        "output": str(run.output_dir),
    }
    document.update(run.command.dry_run_fields())
    if run.dims_path is not None:
        document["dims"] = str(run.dims_path)
    return cast(DryRunDocument, document)


def prepare_diagnose_run(request: DiagnoseRequest) -> PreparedRunDirectoryCommand:
    """Build a Bayesite diagnose command from an existing run directory."""
    run_dir = request.run_dir.expanduser().resolve()
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "diagnostics.json"
    _require_run_artifacts((fit_path,))
    command = BayesiteCommand(
        argv=(
            request.engine,
            "diagnose",
            "--fit",
            str(fit_path),
            "--out",
            str(output_path),
        ),
        output_paths=(output_path,),
    )
    return PreparedRunDirectoryCommand(
        run_dir=run_dir,
        output_path=output_path,
        command=command,
    )


def prepare_posterior_predictive_run(
    request: PosteriorPredictiveRequest,
) -> PreparedRunDirectoryCommand:
    """Build a Bayesite posterior-predictive command from an existing run directory."""
    run_dir = request.run_dir.expanduser().resolve()
    model_path = run_dir / "model.ir.json"
    data_path = run_dir / "data.json"
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "posterior_predictive.ndjson"
    _require_run_artifacts((model_path, data_path, fit_path))
    command = BayesiteCommand(
        argv=(
            request.engine,
            "posterior-predictive",
            "--model",
            str(model_path),
            "--data",
            str(data_path),
            "--fit",
            str(fit_path),
            "--seed",
            request.seed,
            "--out",
            str(output_path),
        ),
        output_paths=(output_path,),
    )
    return PreparedRunDirectoryCommand(
        run_dir=run_dir,
        output_path=output_path,
        command=command,
    )


def run_command_dry_run_document(
    command: PreparedRunDirectoryCommand,
) -> RunCommandDryRunDocument:
    """Return a serializable dry-run summary for a run-directory command."""
    return {
        "run": str(command.run_dir),
        "output": str(command.output_path),
        "engine_command": list(command.command.argv),
    }


def _validate_sample_backend(backend: str) -> None:
    if backend not in {"bayesite", "jaxstanv5"}:
        raise WorkflowError("--backend must be 'bayesite' or 'jaxstanv5'")


def _reject_reserved_engine_args(engine_args: tuple[str, ...], output_path: Path) -> None:
    for arg in engine_args:
        if arg == "--out" or arg.startswith("--out="):
            raise WorkflowError(
                "forwarded engine args may not include --out; "
                f"bayescycle writes output to {output_path}. "
                "Use Bayesite directly for custom output or streaming."
            )


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
