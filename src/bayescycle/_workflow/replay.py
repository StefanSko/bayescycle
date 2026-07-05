"""Replay planning, source verification, and artifact comparison."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from bayescycle._errors import WorkflowError
from bayescycle._run_artifacts.run_metadata import (
    RecordedRunMetadata,
    RecordedRunMetadataInput,
    RunMetadataError,
    read_run_metadata,
    sha256_uri,
)
from bayescycle._settings import SamplerSettings
from bayescycle._workflow.requests import (
    PriorPredictiveRequest,
    RecoverRequest,
    SampleRequest,
    SbcRequest,
    SimulateRequest,
)


@dataclass(frozen=True)
class ReplaySourceCheck:
    """A source file whose recorded hash has been verified before replay."""

    role: str
    path: Path
    sha256: str

    def as_json(self) -> dict[str, object]:
        return {"role": self.role, "path": str(self.path), "sha256": self.sha256}


@dataclass(frozen=True)
class ReplayArtifactReference:
    """A run-directory artifact path that should exist in original and replay runs."""

    role: str
    path: Path


@dataclass(frozen=True)
class ReplayArtifactComparison:
    """Byte comparison for one replayed artifact."""

    role: str
    path: Path
    status: str
    byte_identical: bool
    original_sha256: str | None
    replay_sha256: str | None

    def as_json(self) -> dict[str, object]:
        document: dict[str, object] = {
            "role": self.role,
            "path": str(self.path),
            "status": self.status,
            "byte_identical": self.byte_identical,
        }
        if self.original_sha256 is not None:
            document["original_sha256"] = self.original_sha256
        if self.replay_sha256 is not None:
            document["replay_sha256"] = self.replay_sha256
        return document


@dataclass(frozen=True)
class ReplayComparison:
    """Complete artifact-comparison result for a replay."""

    artifacts: tuple[ReplayArtifactComparison, ...]

    @property
    def byte_identical(self) -> bool:
        """Return whether every compared artifact is byte-identical."""
        return all(artifact.byte_identical for artifact in self.artifacts)

    @property
    def status(self) -> str:
        """Return a stable summary status."""
        if self.byte_identical:
            return "byte-identical"
        return "different"


def load_replay_metadata(run_dir: Path) -> tuple[Path, RecordedRunMetadata]:
    """Load replay metadata from an existing run directory."""
    resolved_run_dir = run_dir.expanduser().resolve()
    if not resolved_run_dir.is_dir():
        raise WorkflowError(f"run directory does not exist: {resolved_run_dir}")
    try:
        return resolved_run_dir, read_run_metadata(resolved_run_dir)
    except RunMetadataError as exc:
        raise WorkflowError(str(exc)) from exc


def verify_replay_sources(record: RecordedRunMetadata) -> tuple[ReplaySourceCheck, ...]:
    """Verify model and input source hashes recorded in run metadata."""
    checks = [
        _verify_source("model", record.model.source_path, record.model.source_sha256),
    ]
    checks.extend(
        _verify_source(f"input {entry.role}", entry.source_path, entry.source_sha256)
        for entry in record.inputs
    )
    return tuple(checks)


def sample_request_from_replay(record: RecordedRunMetadata, output_dir: Path) -> SampleRequest:
    """Reconstruct a sample request from run metadata."""
    data_input = _single_input(record, "data")
    return SampleRequest(
        model_path=record.model.source_path,
        data_path=data_input.source_path,
        output_dir=output_dir,
        model_name=record.model.name,
        sampler=SamplerSettings(
            seed=record.setting("seed"),
            chains=record.setting("chains"),
            warmup=record.setting("warmup"),
            draws=record.setting("draws"),
            max_tree_depth=record.setting("max_tree_depth"),
            target_accept=record.setting("target_accept"),
        ),
    )


def prior_predictive_request_from_replay(
    record: RecordedRunMetadata, output_dir: Path
) -> PriorPredictiveRequest:
    """Reconstruct a prior-predictive request from run metadata."""
    data_input = _single_input(record, "data")
    return PriorPredictiveRequest(
        model_path=record.model.source_path,
        data_path=data_input.source_path,
        output_dir=output_dir,
        model_name=record.model.name,
        seed=record.setting("seed"),
        draws=record.setting("draws"),
    )


def simulate_request_from_replay(record: RecordedRunMetadata, output_dir: Path) -> SimulateRequest:
    """Reconstruct a simulate request from run metadata."""
    data_input = _single_input(record, "data")
    truth_input = _single_input(record, "truth")
    return SimulateRequest(
        model_path=record.model.source_path,
        data_path=data_input.source_path,
        truth_path=truth_input.source_path,
        output_dir=output_dir,
        model_name=record.model.name,
        seed=record.setting("seed"),
    )


def recover_request_from_replay(record: RecordedRunMetadata, output_dir: Path) -> RecoverRequest:
    """Reconstruct a recover request from run metadata."""
    scenario_input = _single_input(record, "scenario")
    return RecoverRequest(
        model_path=record.model.source_path,
        scenario_path=scenario_input.source_path,
        output_dir=output_dir,
        model_name=record.model.name,
    )


def sbc_request_from_replay(record: RecordedRunMetadata, output_dir: Path) -> SbcRequest:
    """Reconstruct an SBC request from run metadata."""
    scenario_input = _single_input(record, "scenario")
    return SbcRequest(
        model_path=record.model.source_path,
        scenario_path=scenario_input.source_path,
        output_dir=output_dir,
        model_name=record.model.name,
        replicates=record.setting("replicates"),
    )


def compare_replay_artifacts(
    record: RecordedRunMetadata,
    *,
    original_run_dir: Path,
    replay_run_dir: Path,
) -> ReplayComparison:
    """Compare original run artifacts against replay artifacts."""
    artifacts = tuple(
        _compare_artifact(
            reference, original_run_dir=original_run_dir, replay_run_dir=replay_run_dir
        )
        for reference in _artifact_references(record)
    )
    return ReplayComparison(artifacts=artifacts)


def replay_plan_document(
    *,
    source_run: Path,
    output_dir: Path,
    record: RecordedRunMetadata,
    source_checks: tuple[ReplaySourceCheck, ...],
    plan: Mapping[str, object],
) -> dict[str, object]:
    """Return a JSON-ready replay check-only document."""
    return {
        "format": "bayescycle.replay-plan.v1",
        "source_run": str(source_run),
        "output": str(output_dir),
        "kind": record.kind,
        "backend": record.backend,
        "verified_sources": [check.as_json() for check in source_checks],
        "plan": dict(plan),
    }


def replay_result_document(
    *,
    source_run: Path,
    output_dir: Path,
    record: RecordedRunMetadata,
    source_checks: tuple[ReplaySourceCheck, ...],
    comparison: ReplayComparison,
) -> dict[str, object]:
    """Return a JSON-ready replay result document."""
    return {
        "format": "bayescycle.replay-result.v1",
        "source_run": str(source_run),
        "output": str(output_dir),
        "kind": record.kind,
        "backend": record.backend,
        "status": comparison.status,
        "byte_identical": comparison.byte_identical,
        "verified_sources": [check.as_json() for check in source_checks],
        "artifacts": [artifact.as_json() for artifact in comparison.artifacts],
    }


def _verify_source(role: str, source_path: Path, expected_sha256: str) -> ReplaySourceCheck:
    resolved_path = source_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise WorkflowError(f"replay source does not exist for {role}: {resolved_path}")
    actual_sha256 = sha256_uri(resolved_path)
    if actual_sha256 != expected_sha256:
        raise WorkflowError(
            f"hash mismatch for {role}: {resolved_path}; "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return ReplaySourceCheck(role=role, path=resolved_path, sha256=actual_sha256)


def _single_input(record: RecordedRunMetadata, role: str) -> RecordedRunMetadataInput:
    matches = tuple(entry for entry in record.inputs if entry.role == role)
    if len(matches) != 1:
        raise WorkflowError(
            f"run metadata for {record.kind} replay must contain exactly one {role} input"
        )
    return matches[0]


def _artifact_references(record: RecordedRunMetadata) -> tuple[ReplayArtifactReference, ...]:
    return (
        ReplayArtifactReference(role="model_ir", path=record.model.ir_path),
        *(ReplayArtifactReference(role=entry.role, path=entry.path) for entry in record.inputs),
        *(ReplayArtifactReference(role=entry.role, path=entry.path) for entry in record.outputs),
    )


def _compare_artifact(
    reference: ReplayArtifactReference,
    *,
    original_run_dir: Path,
    replay_run_dir: Path,
) -> ReplayArtifactComparison:
    original_path = _run_artifact_path(original_run_dir, reference.path)
    replay_path = _run_artifact_path(replay_run_dir, reference.path)
    if not original_path.is_file():
        return ReplayArtifactComparison(
            role=reference.role,
            path=reference.path,
            status="missing-original",
            byte_identical=False,
            original_sha256=None,
            replay_sha256=_sha256_uri_if_file(replay_path),
        )
    if not replay_path.is_file():
        return ReplayArtifactComparison(
            role=reference.role,
            path=reference.path,
            status="missing-replay",
            byte_identical=False,
            original_sha256=_sha256_uri_if_file(original_path),
            replay_sha256=None,
        )

    original_bytes = original_path.read_bytes()
    replay_bytes = replay_path.read_bytes()
    byte_identical = original_bytes == replay_bytes
    return ReplayArtifactComparison(
        role=reference.role,
        path=reference.path,
        status="identical" if byte_identical else "different",
        byte_identical=byte_identical,
        original_sha256=_sha256_uri_bytes(original_bytes),
        replay_sha256=_sha256_uri_bytes(replay_bytes),
    )


def _run_artifact_path(run_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return run_dir / path


def _sha256_uri_if_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_uri(path)


def _sha256_uri_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"
