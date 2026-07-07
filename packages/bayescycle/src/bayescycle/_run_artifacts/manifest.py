"""Bayescycle run-directory manifest serialization."""

from __future__ import annotations

import json
from pathlib import Path

from bayescycle._run_artifacts.canonical_data import DATA_DOC_FORMAT
from bayescycle._run_artifacts.references import CanonicalDataArtifact

RUN_MANIFEST_FORMAT = "bayescycle.run-manifest.v1"


def write_data_manifest(
    output_dir: Path,
    data_path: CanonicalDataArtifact,
    additional_data_paths: tuple[CanonicalDataArtifact, ...] = (),
) -> None:
    """Write a manifest for canonical data artifacts in a run directory."""
    artifacts = {
        artifact.path.name: {
            "format": DATA_DOC_FORMAT,
            "path": artifact.path.name,
        }
        for artifact in (data_path, *additional_data_paths)
    }
    manifest = {
        "manifest_format": RUN_MANIFEST_FORMAT,
        "artifacts": artifacts,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
