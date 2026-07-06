"""Bayescycle run-directory discovery.

This module knows artifact filenames only. It does not parse JSON/NDJSON and it
never loads a fit into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class RunDirError(ValueError):
    """A run directory is missing required artifacts or is not a directory."""


@dataclass(frozen=True)
class RunDir:
    """Resolved artifact paths for one bayescycle run directory."""

    root: Path
    model_ir: Path
    data_json: Path
    posterior: Path
    prior_predictive: Path | None
    posterior_predictive: Path | None
    dims_json: Path | None


def _optional(root: Path, name: str) -> Path | None:
    path = root / name
    if not path.exists():
        return None
    if not path.is_file():
        msg = f"optional artifact {path} exists but is not a file"
        raise RunDirError(msg)
    return path


def discover_run_dir(path: Path) -> RunDir:
    """Resolve the v0 run-directory contract into explicit artifact paths."""
    root = path.expanduser()
    if not root.exists():
        msg = f"run directory not found: {root}"
        raise RunDirError(msg)
    if not root.is_dir():
        msg = f"run path is not a directory: {root}"
        raise RunDirError(msg)

    required_names = ("model.ir.json", "data.json", "posterior.ndjson")
    missing = [name for name in required_names if not (root / name).is_file()]
    if missing:
        msg = f"run directory {root} is missing required artifact(s): {', '.join(missing)}"
        raise RunDirError(msg)

    return RunDir(
        root=root,
        model_ir=root / "model.ir.json",
        data_json=root / "data.json",
        posterior=root / "posterior.ndjson",
        prior_predictive=_optional(root, "prior_predictive.ndjson"),
        posterior_predictive=_optional(root, "posterior_predictive.ndjson"),
        dims_json=_optional(root, "dims.json"),
    )
