"""Bayesite subprocess command execution."""

from __future__ import annotations

import subprocess

from bayescycle._commands import BayesiteCommand


def run_bayesite_command(command: BayesiteCommand) -> int:
    """Run a Bayesite command without mutating existing artifacts."""
    completed = subprocess.run(command.argv, check=False)  # noqa: S603
    return completed.returncode
