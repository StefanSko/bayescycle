"""Portable functional-generation run materialization and replay."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from bayeswire.ir import canonical_bytes

from bayescycle._errors import WorkflowError
from bayescycle._integrations.external_command import ExternalCommand, run_external_command
from bayescycle._model_loader import load_model
from bayescycle._run_artifacts.generated_datasets import (
    GeneratedDatasetsArtifactError,
    parse_generated_datasets,
    verify_generated_datasets,
)
from bayescycle._run_artifacts.posterior_source import (
    PortablePosteriorError,
    validate_portable_posterior,
)
from bayescycle._workflow.filesystem import _ensure_output_dir, _validate_output_dir
from bayescycle._workflow.generation_plan import (
    Draw,
    FitArtifact,
    FitAssociation,
    Fixed,
    GenerationPlanError,
    ModelPrior,
    PosteriorOf,
    generate_datasets,
    resolve_generation_plan_document,
    serialize_generation_plan,
)

GENERATION_RUN_FORMAT = "bayescycle.generation-run.v0"
_MAX_INPUT_BYTES = 8 * 1024 * 1024
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_PATH_BYTES = 255
_MAX_DEPTH = 64
_HASH = "sha256:"
_BACKEND = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")


@dataclass(frozen=True)
class GenerationRunRecord:
    """Validated, locally resolved generation run."""

    run_dir: Path
    backend: str
    plan: Draw
    plan_path: Path
    model_path: Path
    design_path: Path
    output_path: Path
    source_paths: tuple[tuple[str, Path], ...]
    checks: tuple[tuple[str, Path, str], ...]


@dataclass(frozen=True)
class GenerationExecution:
    """Materialized generation command and its declared output."""

    command: ExternalCommand
    output_path: Path


def build_generation_plan(
    *,
    model_path: Path,
    model_name: str | None,
    design_path: Path,
    source_kind: str,
    parameters_path: Path | None,
    fit_path: Path | None,
    fit_data_path: Path | None,
    count: int,
    seed: int,
) -> Draw:
    """Compile one model source and build an immutable generation plan."""
    loaded = load_model(model_path.expanduser().resolve(), model_name)
    model_bytes = canonical_bytes(loaded.meta)
    design_bytes = _read_bounded(design_path, "design")
    if source_kind == "fixed":
        if parameters_path is None or fit_path is not None or fit_data_path is not None:
            raise WorkflowError(
                "fixed generation requires --parameters and rejects --fit/--fit-data"
            )
        parameter_source = Fixed(_read_bounded(parameters_path, "fixed parameters"))
    elif source_kind == "model-prior":
        if parameters_path is not None or fit_path is not None or fit_data_path is not None:
            raise WorkflowError(
                "model-prior generation rejects --parameters, --fit, and --fit-data"
            )
        parameter_source = ModelPrior(model_bytes)
    elif source_kind == "posterior":
        if parameters_path is not None or fit_path is None or fit_data_path is None:
            raise WorkflowError(
                "posterior generation requires --fit and --fit-data and rejects --parameters"
            )
        parameter_source = PosteriorOf(
            FitArtifact(
                model_ir_bytes=model_bytes,
                data_bytes=_read_bounded(fit_data_path, "source fit data"),
                posterior_bytes=_read_bounded(fit_path, "source posterior"),
                association=FitAssociation.PORTABLE,
            )
        )
    else:
        raise WorkflowError(f"unsupported generation source: {source_kind}")
    try:
        return generate_datasets(
            model_bytes,
            design=design_bytes,
            parameter_source=parameter_source,
            count=count,
            seed=seed,
        )
    except GenerationPlanError as exc:
        raise WorkflowError(str(exc)) from exc


def execute_generation_run(
    *, output_dir: Path, plan: Draw, engine: str, backend: str = "bayesite"
) -> int:
    """Materialize, execute, verify, and publish one portable generation run."""
    if backend != "bayesite":
        raise WorkflowError("functional generation v0 currently requires the bayesite backend")
    execution = materialize_generation_run(output_dir=output_dir, plan=plan, engine=engine)
    code = run_external_command(execution.command)
    if code != 0:
        return code
    _verify_generated_output(plan, execution.output_path)
    _write_generation_metadata(output_dir.expanduser().resolve(), plan, backend)
    return 0


def materialize_generation_run(*, output_dir: Path, plan: Draw, engine: str) -> GenerationExecution:
    """Write only local exact payloads and return the native generation command."""
    run_dir = output_dir.expanduser().resolve()
    _validate_output_dir(run_dir)
    if isinstance(plan.distribution.parameters, PosteriorOf):
        fit = plan.distribution.parameters.fit_artifact
        if fit.association is not FitAssociation.PORTABLE:
            raise WorkflowError(
                "generation run publication requires a portable posterior fingerprint"
            )
        try:
            validate_portable_posterior(
                model_bytes=fit.model_ir_bytes,
                data_bytes=fit.data_bytes,
                posterior_bytes=fit.posterior_bytes,
            )
        except PortablePosteriorError as exc:
            raise WorkflowError(f"posterior source is not portable: {exc}") from exc
    _ensure_output_dir(run_dir)
    model_path = run_dir / "model.ir.json"
    design_path = run_dir / "design.json"
    plan_path = run_dir / "generation-plan.json"
    output_path = run_dir / "generated_datasets.ndjson"
    model_path.write_bytes(plan.distribution.outcomes.model_ir_bytes)
    design_path.write_bytes(plan.distribution.outcomes.design_bytes)
    plan_path.write_bytes(serialize_generation_plan(plan))
    source = plan.distribution.parameters
    source_args: tuple[str, ...]
    if isinstance(source, Fixed):
        parameters_path = run_dir / "fixed-parameters.json"
        parameters_path.write_bytes(source.parameters_bytes)
        source_args = ("--source", "fixed", "--parameters", str(parameters_path))
    elif isinstance(source, ModelPrior):
        source_args = ("--source", "model-prior")
    else:
        posterior_path = run_dir / "source-posterior.ndjson"
        fit_data_path = run_dir / "source-fit-data.json"
        posterior_path.write_bytes(source.fit_artifact.posterior_bytes)
        fit_data_path.write_bytes(source.fit_artifact.data_bytes)
        source_args = (
            "--source",
            "posterior",
            "--fit",
            str(posterior_path),
            "--fit-data",
            str(fit_data_path),
        )
    return GenerationExecution(
        command=ExternalCommand(
            argv=(
                engine,
                "generate",
                "--model",
                str(model_path),
                "--design",
                str(design_path),
                *source_args,
                "--count",
                str(plan.count),
                "--seed",
                str(plan.seed),
                "--out",
                str(output_path),
            ),
            output_paths=(output_path,),
        ),
        output_path=output_path,
    )


def load_generation_run(run_dir: Path) -> GenerationRunRecord:
    """Validate a complete generation run using only contained regular files."""
    root = run_dir.expanduser().resolve()
    if not root.is_dir():
        raise WorkflowError(f"run directory does not exist: {root}")
    metadata_path = _contained_regular(root, "run.json", "metadata")
    try:
        metadata_bytes = metadata_path.read_bytes()
        if not metadata_bytes or len(metadata_bytes) > _MAX_METADATA_BYTES:
            raise WorkflowError(
                f"generation run metadata must contain 1..{_MAX_METADATA_BYTES} bytes"
            )
        _validate_json_depth(metadata_bytes, "generation run metadata")
        raw = cast(object, json.loads(metadata_bytes.decode("utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"invalid generation run metadata: {exc}") from exc
    document = _object(raw, "generation run metadata")
    _exact(document, ("format", "kind", "backend", "plan", "model", "inputs", "outputs"))
    if document["format"] != GENERATION_RUN_FORMAT or document["kind"] != "generate":
        raise WorkflowError("run metadata is not a functional generation run")
    backend = _string(document["backend"], "backend")
    if _BACKEND.fullmatch(backend) is None:
        raise WorkflowError(
            "backend must be 1..64 ASCII letters, digits, dots, underscores, or hyphens"
        )
    plan_entry = _entry(document["plan"], "plan")
    model_entry = _entry(document["model"], "model")
    inputs = _entries(document["inputs"], "inputs")
    outputs = _entries(document["outputs"], "outputs")
    if plan_entry[:2] != ("generation-plan.json", "v0-provisional"):
        raise WorkflowError("generation run plan path or format is invalid")
    if model_entry[:2] != ("model.ir.json", "bayeswire_ir.v1"):
        raise WorkflowError("generation run model path or format is invalid")
    if not inputs or inputs[0][0:2] != ("design", "design.json"):
        raise WorkflowError("generation run inputs must begin with design.json")
    if len(outputs) != 1 or outputs[0][0:3] != (
        "generated-datasets",
        "generated_datasets.ndjson",
        "v0-provisional",
    ):
        raise WorkflowError("generation run output declaration is invalid")
    paths: dict[str, Path] = {}
    checks: list[tuple[str, Path, str]] = []
    for role, name, artifact_format, expected_hash in (
        ("plan", plan_entry[0], plan_entry[1], plan_entry[2]),
        ("model", model_entry[0], model_entry[1], model_entry[2]),
        *((entry[0], entry[1], entry[2], entry[3]) for entry in inputs),
        *((entry[0], entry[1], entry[2], entry[3]) for entry in outputs),
    ):
        del artifact_format
        path = _contained_regular(root, name, role)
        actual = _sha256(path.read_bytes())
        if actual != expected_hash:
            raise WorkflowError(
                f"hash mismatch for generation {role}: expected {expected_hash}, found {actual}"
            )
        paths[role] = path
        checks.append((role, path, actual))
    roles = tuple(entry[0] for entry in inputs)
    if roles == ("design", "fixed-parameters"):
        expected_names = ("design.json", "fixed-parameters.json")
        expected_formats = ("bayescycle.data.json.v1", "bayescycle.data.json.v1")
    elif roles == ("design",):
        expected_names = ("design.json",)
        expected_formats = ("bayescycle.data.json.v1",)
    elif roles == ("design", "source-posterior", "source-fit-data"):
        expected_names = ("design.json", "source-posterior.ndjson", "source-fit-data.json")
        expected_formats = (
            "bayescycle.data.json.v1",
            "v0-provisional",
            "bayescycle.data.json.v1",
        )
    else:
        raise WorkflowError(f"generation run input roles are invalid: {roles}")
    if tuple(entry[1] for entry in inputs) != expected_names:
        raise WorkflowError("generation run input role-to-path mapping is invalid")
    if tuple(entry[2] for entry in inputs) != expected_formats:
        raise WorkflowError("generation run input format mapping is invalid")
    input_paths = {entry[0]: _contained_regular(root, entry[1], entry[0]) for entry in inputs}
    try:
        plan = resolve_generation_plan_document(
            paths["plan"].read_bytes(),
            model_ir_bytes=paths["model"].read_bytes(),
            design_bytes=input_paths["design"].read_bytes(),
            fixed_parameters_bytes=(
                input_paths["fixed-parameters"].read_bytes()
                if "fixed-parameters" in input_paths
                else None
            ),
            posterior_bytes=(
                input_paths["source-posterior"].read_bytes()
                if "source-posterior" in input_paths
                else None
            ),
            fit_data_bytes=(
                input_paths["source-fit-data"].read_bytes()
                if "source-fit-data" in input_paths
                else None
            ),
        )
        _verify_generated_output(plan, paths["generated-datasets"])
    except (GenerationPlanError, GeneratedDatasetsArtifactError) as exc:
        raise WorkflowError(str(exc)) from exc
    return GenerationRunRecord(
        run_dir=root,
        backend=backend,
        plan=plan,
        plan_path=paths["plan"],
        model_path=paths["model"],
        design_path=input_paths["design"],
        output_path=paths["generated-datasets"],
        source_paths=tuple((role, input_paths[role]) for role in roles[1:]),
        checks=tuple(checks),
    )


def replay_generation_run(
    *, record: GenerationRunRecord, output_dir: Path, engine: str, check_only: bool
) -> tuple[int, dict[str, object]]:
    """Plan or execute a source-free generation replay."""
    target = output_dir.expanduser().resolve()
    verified = [
        {"role": role, "path": str(path), "sha256": digest}
        for role, path, digest in record.checks
        if role != "generated-datasets"
    ]
    execution = _planned_generation_execution(target, record.plan, engine)
    if check_only:
        return 0, {
            "format": "bayescycle.replay-plan.v1",
            "source_run": str(record.run_dir),
            "output": str(target),
            "kind": "generate",
            "backend": record.backend,
            "verified_sources": verified,
            "plan": {"command": list(execution.command.argv)},
        }
    code = execute_generation_run(
        output_dir=target, plan=record.plan, engine=engine, backend=record.backend
    )
    if code != 0:
        return code, {}
    replay = load_generation_run(target)
    references = (
        ("model", record.model_path, replay.model_path),
        ("design", record.design_path, replay.design_path),
        ("plan", record.plan_path, replay.plan_path),
        ("generated-datasets", record.output_path, replay.output_path),
        *(
            (role, original, dict(replay.source_paths)[role])
            for role, original in record.source_paths
        ),
    )
    artifacts = []
    for role, original, regenerated in references:
        original_hash = _sha256(original.read_bytes())
        replay_hash = _sha256(regenerated.read_bytes())
        identical = original.read_bytes() == regenerated.read_bytes()
        artifacts.append(
            {
                "role": role,
                "path": original.name,
                "status": "identical" if identical else "different",
                "byte_identical": identical,
                "original_sha256": original_hash,
                "replay_sha256": replay_hash,
            }
        )
    identical = all(cast(bool, item["byte_identical"]) for item in artifacts)
    return (0 if identical else 1), {
        "format": "bayescycle.replay-result.v1",
        "source_run": str(record.run_dir),
        "output": str(target),
        "kind": "generate",
        "backend": record.backend,
        "status": "byte-identical" if identical else "different",
        "byte_identical": identical,
        "verified_sources": verified,
        "artifacts": artifacts,
    }


def is_generation_run(run_dir: Path) -> bool:
    """Return whether bounded regular run.json advertises the generation profile."""
    root = run_dir.expanduser().resolve()
    metadata = root / "run.json"
    if metadata.is_symlink():
        raise WorkflowError(f"run metadata must be a contained regular file: {metadata}")
    try:
        data = metadata.read_bytes()
        if not data or len(data) > _MAX_METADATA_BYTES:
            return False
        _validate_json_depth(data, "run metadata")
        value = cast(object, json.loads(data.decode("utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, WorkflowError):
        return False
    return isinstance(value, dict) and value.get("format") == GENERATION_RUN_FORMAT


def _planned_generation_execution(output_dir: Path, plan: Draw, engine: str) -> GenerationExecution:
    """Render replay argv without touching the output directory."""
    root = output_dir
    source = plan.distribution.parameters
    if isinstance(source, Fixed):
        source_args = (
            "--source",
            "fixed",
            "--parameters",
            str(root / "fixed-parameters.json"),
        )
    elif isinstance(source, ModelPrior):
        source_args = ("--source", "model-prior")
    else:
        source_args = (
            "--source",
            "posterior",
            "--fit",
            str(root / "source-posterior.ndjson"),
            "--fit-data",
            str(root / "source-fit-data.json"),
        )
    output = root / "generated_datasets.ndjson"
    return GenerationExecution(
        command=ExternalCommand(
            argv=(
                engine,
                "generate",
                "--model",
                str(root / "model.ir.json"),
                "--design",
                str(root / "design.json"),
                *source_args,
                "--count",
                str(plan.count),
                "--seed",
                str(plan.seed),
                "--out",
                str(output),
            ),
            output_paths=(output,),
        ),
        output_path=output,
    )


def _write_generation_metadata(run_dir: Path, plan: Draw, backend: str) -> None:
    plan_path = run_dir / "generation-plan.json"
    model_path = run_dir / "model.ir.json"
    design_path = run_dir / "design.json"
    output_path = run_dir / "generated_datasets.ndjson"
    inputs = [_metadata_entry("design", design_path, "bayescycle.data.json.v1")]
    source = plan.distribution.parameters
    if isinstance(source, Fixed):
        inputs.append(
            _metadata_entry(
                "fixed-parameters",
                run_dir / "fixed-parameters.json",
                "bayescycle.data.json.v1",
            )
        )
    elif isinstance(source, PosteriorOf):
        inputs.extend(
            (
                _metadata_entry(
                    "source-posterior",
                    run_dir / "source-posterior.ndjson",
                    "v0-provisional",
                ),
                _metadata_entry(
                    "source-fit-data",
                    run_dir / "source-fit-data.json",
                    "bayescycle.data.json.v1",
                ),
            )
        )
    document = {
        "format": GENERATION_RUN_FORMAT,
        "kind": "generate",
        "backend": backend,
        "plan": {
            "path": plan_path.name,
            "sha256": _sha256(plan_path.read_bytes()),
            "format": "v0-provisional",
        },
        "model": {
            "path": model_path.name,
            "sha256": _sha256(model_path.read_bytes()),
            "format": "bayeswire_ir.v1",
        },
        "inputs": inputs,
        "outputs": [_metadata_entry("generated-datasets", output_path, "v0-provisional")],
    }
    (run_dir / "run.json").write_text(
        json.dumps(document, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metadata_entry(role: str, path: Path, artifact_format: str) -> dict[str, str]:
    return {
        "role": role,
        "path": path.name,
        "sha256": _sha256(path.read_bytes()),
        "format": artifact_format,
    }


def _verify_generated_output(plan: Draw, output_path: Path) -> None:
    if not output_path.is_file():
        raise WorkflowError(f"generation output does not exist: {output_path}")
    artifact = parse_generated_datasets(output_path.read_bytes())
    fixed = plan.distribution.parameters
    verify_generated_datasets(
        artifact,
        model_bytes=plan.distribution.outcomes.model_ir_bytes,
        design_bytes=plan.distribution.outcomes.design_bytes,
        fixed_parameters_bytes=fixed.parameters_bytes if isinstance(fixed, Fixed) else None,
        model_prior_bytes=fixed.model_ir_bytes if isinstance(fixed, ModelPrior) else None,
        authored_provenance=(
            (
                fixed.authored_provenance.claimed_source_model_hash,
                fixed.authored_provenance.claimed_outcome_model_hash,
            )
            if isinstance(fixed, ModelPrior) and fixed.authored_provenance is not None
            else None
        ),
        posterior_bytes=(
            fixed.fit_artifact.posterior_bytes if isinstance(fixed, PosteriorOf) else None
        ),
        fit_data_bytes=(fixed.fit_artifact.data_bytes if isinstance(fixed, PosteriorOf) else None),
        expected_source_kind=(
            "fixed"
            if isinstance(fixed, Fixed)
            else "model-prior"
            if isinstance(fixed, ModelPrior)
            else "posterior"
        ),
        expected_count=plan.count,
        expected_seed=plan.seed,
    )


def _read_bounded(path: Path, label: str) -> bytes:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise WorkflowError(f"{label} file does not exist: {resolved}")
    data = resolved.read_bytes()
    if not data or len(data) > _MAX_INPUT_BYTES:
        raise WorkflowError(f"{label} must contain 1..{_MAX_INPUT_BYTES} bytes")
    return data


def _contained_regular(root: Path, name: str, role: str) -> Path:
    if len(name.encode("utf-8")) > _MAX_PATH_BYTES:
        raise WorkflowError(f"generation {role} path exceeds {_MAX_PATH_BYTES} UTF-8 bytes")
    path = Path(name)
    if path.name != name or path.is_absolute() or name in {".", ".."}:
        raise WorkflowError(f"generation {role} path must be one normalized path segment")
    candidate = root / path
    if candidate.is_symlink() or not candidate.is_file():
        raise WorkflowError(f"generation {role} must be a contained regular file: {candidate}")
    if candidate.resolve().parent != root:
        raise WorkflowError(f"generation {role} escapes the run directory")
    return candidate


def _validate_json_depth(data: bytes, label: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            if depth > _MAX_DEPTH:
                raise WorkflowError(f"{label} exceeds nesting depth {_MAX_DEPTH}")
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise WorkflowError(f"{label} has malformed nesting")
    if in_string or depth != 0:
        raise WorkflowError(f"{label} has malformed nesting")


def _entry(value: object, label: str) -> tuple[str, str, str]:
    document = _object(value, label)
    _exact(document, ("path", "sha256", "format"))
    return (
        _string(document["path"], f"{label}.path"),
        _string(document["format"], f"{label}.format"),
        _digest(document["sha256"], f"{label}.sha256"),
    )


def _entries(value: object, label: str) -> tuple[tuple[str, str, str, str], ...]:
    if not isinstance(value, list):
        raise WorkflowError(f"{label} must be an array")
    entries = []
    for index, raw in enumerate(cast(list[object], value)):
        document = _object(raw, f"{label}[{index}]")
        _exact(document, ("role", "path", "sha256", "format"))
        entries.append(
            (
                _string(document["role"], f"{label}[{index}].role"),
                _string(document["path"], f"{label}[{index}].path"),
                _string(document["format"], f"{label}[{index}].format"),
                _digest(document["sha256"], f"{label}[{index}].sha256"),
            )
        )
    return tuple(entries)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise WorkflowError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _exact(document: dict[str, object], keys: tuple[str, ...]) -> None:
    if tuple(document) != keys:
        raise WorkflowError(
            f"generation metadata has unknown, missing, or out-of-order keys: {tuple(document)}"
        )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkflowError(f"{label} must be a non-empty string")
    return value


def _digest(value: object, label: str) -> str:
    digest = _string(value, label)
    if not digest.startswith(_HASH) or len(digest) != 71:
        raise WorkflowError(f"{label} must be a lowercase sha256 hash")
    try:
        int(digest[7:], 16)
    except ValueError as exc:
        raise WorkflowError(f"{label} must be a lowercase sha256 hash") from exc
    if digest.lower() != digest:
        raise WorkflowError(f"{label} must be a lowercase sha256 hash")
    return digest


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"
