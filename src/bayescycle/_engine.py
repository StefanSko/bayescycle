"""Bayesite engine command representation and execution."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineCommand:
    """A concrete engine command ready for subprocess execution."""

    argv: tuple[str, ...]


def run_engine(command: EngineCommand) -> int:
    """Run the Bayesite engine command, inheriting stdio."""
    completed = subprocess.run(command.argv, check=False)  # noqa: S603
    return completed.returncode
