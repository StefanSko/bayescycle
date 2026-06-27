"""Bayesite engine binary preflight checks."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from bayescycle._errors import WorkflowError


@dataclass(frozen=True)
class BayesiteCommandRequirement:
    """A Bayesite CLI command required by one workflow stage."""

    command: str
    stage: str


@dataclass(frozen=True)
class BayesiteEngineInfo:
    """Resolved Bayesite engine executable and advertised help text."""

    executable: Path
    help_text: str


def preflight_bayesite_engine(
    engine: str, requirements: tuple[BayesiteCommandRequirement, ...]
) -> BayesiteEngineInfo:
    """Validate a Bayesite engine path and required command support."""
    executable = _resolve_engine(engine)
    help_text = _engine_help_text(executable)
    for requirement in requirements:
        if not _supports_command(help_text, requirement.command):
            raise WorkflowError(_missing_command_message(executable, requirement))
    return BayesiteEngineInfo(executable=executable, help_text=help_text)


def _resolve_engine(engine: str) -> Path:
    expanded = Path(engine).expanduser()
    has_path_separator = os.sep in engine or (os.altsep is not None and os.altsep in engine)
    if has_path_separator or expanded.is_absolute():
        candidate = expanded.resolve(strict=False)
    else:
        found = shutil.which(engine)
        if found is None:
            raise WorkflowError(
                f"Bayesite engine was not found on PATH: {engine}\n\n"
                "Install Bayesite or pass an explicit executable with --engine."
            )
        candidate = Path(found).resolve(strict=False)
    if not candidate.exists():
        raise WorkflowError(
            f"Bayesite engine does not exist: {candidate}\n\n"
            "Build Bayesite and pass the fresh binary with --engine."
        )
    if not candidate.is_file():
        raise WorkflowError(f"Bayesite engine is not a file: {candidate}")
    if not os.access(candidate, os.X_OK):
        raise WorkflowError(
            f"Bayesite engine is not executable: {candidate}\n\n"
            "Make it executable or pass a different binary with --engine."
        )
    return candidate


def _engine_help_text(executable: Path) -> str:
    try:
        completed = subprocess.run(  # noqa: S603
            [str(executable)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError as exc:
        raise WorkflowError(f"cannot execute Bayesite engine {executable}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise WorkflowError(f"Bayesite engine preflight timed out: {executable}") from exc
    return f"{completed.stdout}\n{completed.stderr}"


def _supports_command(help_text: str, command: str) -> bool:
    return (
        re.search(rf"(?<![A-Za-z0-9-]){re.escape(command)}(?![A-Za-z0-9-])", help_text) is not None
    )


def _missing_command_message(executable: Path, requirement: BayesiteCommandRequirement) -> str:
    return (
        f"Selected Bayesite engine does not support required command: {requirement.command}\n\n"
        f"engine: {executable}\n"
        f"required by stage: {requirement.stage}\n\n"
        "The binary may be stale. Rebuild Bayesite and pass the fresh binary, for example:\n"
        "  cargo build --release\n"
        "  bayescycle ... --backend bayesite --engine /path/to/bayesite/target/release/bayesite"
    )
