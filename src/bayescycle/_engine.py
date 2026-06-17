"""Bayesite engine command representation and execution."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EngineCommand:
    """A concrete engine command ready for subprocess execution."""

    argv: tuple[str, ...]
    stdout_path: Path


def run_engine(command: EngineCommand) -> int:
    """Run the Bayesite engine command, writing stdout to the run artifact."""
    with command.stdout_path.open("wb") as stdout:
        completed = subprocess.run(command.argv, check=False, stdout=stdout)  # noqa: S603
    return completed.returncode
