"""Paired generated-dataset artifact contract."""

from __future__ import annotations

from dataclasses import dataclass

from bayescycle._run_artifacts.canonical_data import DataDoc

MAX_GENERATED_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_GENERATED_LINE_BYTES = 8 * 1024 * 1024


class GeneratedDatasetsArtifactError(ValueError):
    """Raised when a paired generated-dataset artifact is invalid."""


@dataclass(frozen=True)
class GeneratedDatasetDraw:
    """One validated parameter/dataset pair."""

    draw_index: int
    parameters: DataDoc
    dataset: DataDoc
    source_kind: str


@dataclass(frozen=True)
class GeneratedDatasetSelection:
    """Exact standalone bytes selected from one paired record."""

    draw_index: int
    parameters_bytes: bytes
    dataset_bytes: bytes


@dataclass(frozen=True)
class GeneratedDatasetsArtifact:
    """Validated immutable paired generated-dataset stream."""

    generation_model_hash: str
    design_hash: str
    source_kind: str
    count: int
    seed: int
    draws: tuple[GeneratedDatasetDraw, ...]
    bytes: bytes

    def select(self, draw_index: int) -> GeneratedDatasetSelection:
        """Select exact parameter and dataset bytes for one draw."""
        raise NotImplementedError("generated-dataset selection is not implemented")


def parse_generated_datasets(data: bytes) -> GeneratedDatasetsArtifact:
    """Parse a bounded paired generated-dataset NDJSON artifact."""
    raise NotImplementedError("generated-dataset parsing is not implemented")


def verify_generated_datasets(
    artifact: GeneratedDatasetsArtifact,
    *,
    model_bytes: bytes,
    design_bytes: bytes,
    fixed_parameters_bytes: bytes | None = None,
) -> None:
    """Verify hash-resolved model, design, and fixed-source lineage."""
    raise NotImplementedError("generated-dataset verification is not implemented")
