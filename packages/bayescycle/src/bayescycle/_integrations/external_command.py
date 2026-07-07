"""Unix-style external command integration primitives."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExternalCommand:
    """A concrete external command ready for subprocess execution."""

    argv: tuple[str, ...]
    output_paths: tuple[Path, ...] = ()


def run_external_command(command: ExternalCommand) -> int:
    """Run an external command without mutating existing artifacts."""
    completed = subprocess.run(command.argv, check=False)  # noqa: S603
    return completed.returncode
