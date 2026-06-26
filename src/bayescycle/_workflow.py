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
from bayescycle._model_loader import LoadedModel, load_model


class WorkflowError(RuntimeError):
    """Raised when a workflow request cannot be prepared."""


DEFAULT_SEED = 0
DEFAULT_CHAINS = 4
DEFAULT_WARMUP = 1000
DEFAULT_DRAWS = 1000
DEFAULT_MAX_TREE_DEPTH = 10
DEFAULT_TARGET_ACCEPT = 0.8
MAX_REPORTABLE_I64 = 9_223_372_036_854_775_807


@dataclass(frozen=True)
class ResolvedSamplerSettings:
    """Typed sampler settings used by the in-process backend."""

    seed: int
    chains: int
    warmup: int
    draws: int
    max_tree_depth: int
    target_accept: float

    def as_json(self) -> dict[str, int | float]:
        """Return a JSON-ready settings summary."""
        return {
            "seed": self.seed,
            "chains": self.chains,
            "warmup": self.warmup,
            "draws": self.draws,
            "max_tree_depth": self.max_tree_depth,
            "target_accept": self.target_accept,
        }


@dataclass(frozen=True)
class SamplerSettings:
    """Common sampler CLI settings forwarded or resolved by backend."""

    seed: str | None
    chains: str | None
    warmup: str | None
    draws: str | None
    max_tree_depth: str | None
    target_accept: str | None

    def to_engine_args(self) -> tuple[str, ...]:
        """Return Bayesite CLI arguments for explicitly requested settings."""
        args: list[str] = []
        if self.seed is not None:
            args.extend(("--seed", self.seed))
        if self.chains is not None:
            args.extend(("--chains", self.chains))
        if self.warmup is not None:
            args.extend(("--warmup", self.warmup))
        if self.draws is not None:
            args.extend(("--draws", self.draws))
        if self.max_tree_depth is not None:
            args.extend(("--max-treedepth", self.max_tree_depth))
        if self.target_accept is not None:
            args.extend(("--target-accept", self.target_accept))
        return tuple(args)

    def resolve_for_in_process(self) -> ResolvedSamplerSettings:
        """Parse sampler settings for direct Python execution."""
        seed = _parse_nonnegative_int(self.seed, "--seed", default=DEFAULT_SEED)
        chains = _parse_positive_int(self.chains, "--chains", default=DEFAULT_CHAINS)
        warmup = _parse_positive_int(self.warmup, "--warmup", default=DEFAULT_WARMUP)
        draws = _parse_positive_int(self.draws, "--draws", default=DEFAULT_DRAWS)
        if draws < 4:
            raise WorkflowError(
                "--draws must be at least 4 because posterior artifacts include diagnostics"
            )
        max_tree_depth = _parse_positive_int(
            self.max_tree_depth, "--max-treedepth", default=DEFAULT_MAX_TREE_DEPTH
        )
        if max_tree_depth > 20:
            raise WorkflowError("--max-treedepth must be in 1..=20")
        target_accept = _parse_probability(
            self.target_accept, "--target-accept", default=DEFAULT_TARGET_ACCEPT
        )
        return ResolvedSamplerSettings(
            seed=seed,
            chains=chains,
            warmup=warmup,
            draws=draws,
            max_tree_depth=max_tree_depth,
            target_accept=target_accept,
        )


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
class InProcessSampleCommand:
    """An in-process jaxstanv5 sampling command."""

    loaded_model: LoadedModel
    ir_path: Path
    data_path: Path
    draws_path: Path
    settings: ResolvedSamplerSettings


type SampleExecution = EngineCommand | InProcessSampleCommand


@dataclass(frozen=True)
class PreparedSampleRun:
    """A run directory and sample execution produced from a sample request."""

    model_name: str
    ir_path: Path
    data_path: Path
    dims_path: Path | None
    output_dir: Path
    draws_path: Path
    execution: SampleExecution


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
    engine_command: EngineCommand


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


def prepare_sample_run(request: SampleRequest) -> PreparedSampleRun:
    """Load model metadata, write IR/data files, and build the sample execution."""
    _validate_sample_backend(request.backend)
    output_dir = request.output_dir.expanduser().resolve()
    draws_path = output_dir / "posterior.ndjson"
    _reject_reserved_engine_args(request.engine_args, draws_path)
    if request.backend == "jaxstanv5" and request.engine_args:
        raise WorkflowError("engine passthrough after -- is only supported for --backend bayesite")
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

    if request.backend == "bayesite":
        execution: SampleExecution = EngineCommand(
            argv=(
                request.engine,
                "sample",
                "--model",
                str(ir_path),
                "--data",
                str(run_data_path),
                *request.sampler.to_engine_args(),
                "--out",
                str(draws_path),
                *request.engine_args,
            ),
            output_paths=(draws_path,),
        )
    else:
        execution = InProcessSampleCommand(
            loaded_model=loaded_model,
            ir_path=ir_path,
            data_path=run_data_path,
            draws_path=draws_path,
            settings=request.sampler.resolve_for_in_process(),
        )
    return PreparedSampleRun(
        model_name=loaded_model.name,
        ir_path=ir_path,
        data_path=run_data_path,
        dims_path=dims_path,
        output_dir=output_dir,
        draws_path=draws_path,
        execution=execution,
    )


def dry_run_document(run: PreparedSampleRun) -> DryRunDocument:
    """Return a serializable sample dry-run summary."""
    document: DryRunDocument = {
        "model": run.model_name,
        "ir": str(run.ir_path),
        "data": str(run.data_path),
        "draws": str(run.draws_path),
        "output": str(run.output_dir),
    }
    if isinstance(run.execution, EngineCommand):
        document["engine_command"] = list(run.execution.argv)
    else:
        document["backend"] = "jaxstanv5"
        document["sampler"] = run.execution.settings.as_json()
    if run.dims_path is not None:
        document["dims"] = str(run.dims_path)
    return document


def prepare_diagnose_run(request: DiagnoseRequest) -> PreparedRunDirectoryCommand:
    """Build a Bayesite diagnose command from an existing run directory."""
    run_dir = request.run_dir.expanduser().resolve()
    fit_path = run_dir / "posterior.ndjson"
    output_path = run_dir / "diagnostics.json"
    _require_run_artifacts((fit_path,))
    command = EngineCommand(
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
        engine_command=command,
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
    command = EngineCommand(
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
        engine_command=command,
    )


def run_command_dry_run_document(
    command: PreparedRunDirectoryCommand,
) -> RunCommandDryRunDocument:
    """Return a serializable dry-run summary for a run-directory command."""
    return {
        "run": str(command.run_dir),
        "output": str(command.output_path),
        "engine_command": list(command.engine_command.argv),
    }


def _validate_sample_backend(backend: str) -> None:
    if backend not in {"bayesite", "jaxstanv5"}:
        raise WorkflowError("--backend must be 'bayesite' or 'jaxstanv5'")


def _parse_nonnegative_int(value: str | None, name: str, *, default: int) -> int:
    parsed = _parse_int(value, name, default=default)
    if parsed < 0:
        raise WorkflowError(f"{name} must be non-negative")
    if parsed > MAX_REPORTABLE_I64:
        raise WorkflowError(
            f"{name} must be in 0..={MAX_REPORTABLE_I64} because artifacts report it "
            "as a JSON integer"
        )
    return parsed


def _parse_positive_int(value: str | None, name: str, *, default: int) -> int:
    parsed = _parse_int(value, name, default=default)
    if parsed < 1:
        raise WorkflowError(f"{name} must be at least 1")
    if parsed > MAX_REPORTABLE_I64:
        raise WorkflowError(
            f"{name} must be in 1..={MAX_REPORTABLE_I64} because artifacts report it "
            "as a JSON integer"
        )
    return parsed


def _parse_int(value: str | None, name: str, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise WorkflowError(f"{name} must be an integer") from exc


def _parse_probability(value: str | None, name: str, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise WorkflowError(f"{name} must be a number in (0, 1)") from exc
    if not 0.0 < parsed < 1.0:
        raise WorkflowError(f"{name} must be in (0, 1)")
    return parsed


def _reject_reserved_engine_args(engine_args: tuple[str, ...], draws_path: Path) -> None:
    for arg in engine_args:
        if arg == "--out" or arg.startswith("--out="):
            raise WorkflowError(
                "forwarded engine args may not include --out; "
                f"bayescycle writes draws to {draws_path}. "
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
