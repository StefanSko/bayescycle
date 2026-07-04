"""Bayesite backend-private data materialization steps."""

from __future__ import annotations

from dataclasses import dataclass

from bayescycle._errors import WorkflowError
from bayescycle._run_artifacts.canonical_data import (
    DataDocError,
    read_data_doc,
    write_data_doc,
)
from bayescycle._run_artifacts.references import BackendPrivateArtifact, CanonicalDataArtifact


@dataclass(frozen=True)
class CanonicalizeGeneratedData:
    """Canonicalize Bayesite-native generated data after engine execution."""

    native_path: BackendPrivateArtifact
    canonical_path: CanonicalDataArtifact


def canonicalize_generated_data(step: CanonicalizeGeneratedData) -> None:
    """Write Bayescycle canonical generated data from a Bayesite-native output."""
    try:
        generated = read_data_doc(step.native_path.path)
        write_data_doc(step.canonical_path.path, generated)
    except DataDocError as exc:
        raise WorkflowError(
            f"Bayesite simulate did not produce a valid data artifact: {exc}"
        ) from exc
