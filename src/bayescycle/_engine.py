"""Bayesite engine command representation and execution."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EngineCommand:
    """A concrete engine command ready for subprocess execution."""

    argv: tuple[str, ...]
    output_paths: tuple[Path, ...] = ()


def run_engine(command: EngineCommand) -> int:
    """Run the Bayesite engine command after clearing owned output artifacts."""
    for path in command.output_paths:
        path.unlink(missing_ok=True)
    completed = subprocess.run(command.argv, check=False)  # noqa: S603
    return completed.returncode
