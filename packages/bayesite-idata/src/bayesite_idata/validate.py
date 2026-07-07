"""Optional validation through the canonical Bayesite CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

ValidateMode = Literal["require", "warn", "skip"]
VALIDATE_MODES: tuple[ValidateMode, ...] = ("require", "warn", "skip")
_VALIDATE_MODES_BY_VALUE: dict[str, ValidateMode] = {
    "require": "require",
    "warn": "warn",
    "skip": "skip",
}


class ValidationError(RuntimeError):
    """Canonical Bayesite validation failed or could not run."""


def validate_mode(value: str) -> ValidateMode:
    """Normalize a Click-validated validation mode."""
    return _VALIDATE_MODES_BY_VALUE[value]


def run_diagnose(fit: Path, mode: ValidateMode, *, bayesite: Path | None = None) -> None:
    """Run ``bayesite diagnose`` before lenient assembly.

    ``require`` fails if the binary is absent or diagnose exits non-zero.
    ``warn`` reports the same condition to stderr and continues. ``skip`` does
    nothing. Explicit binary precedence is ``--bayesite``, then ``BAYESITE``,
    then PATH lookup for ``bayesite``.
    """
    if mode == "skip":
        return
    exe = _resolve_bayesite(bayesite, mode)
    if exe is None:
        return
    try:
        result = subprocess.run(
            [str(exe), "diagnose", "--fit", str(fit)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        detail = exc.strerror or str(exc)
        _handle_failure(f"failed to run `{exe} diagnose --fit {fit}`: {detail}", mode)
        return
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        _handle_failure(f"`{exe} diagnose --fit {fit}` failed: {detail}", mode)


def _resolve_bayesite(bayesite: Path | None, mode: ValidateMode) -> Path | None:
    if bayesite is not None:
        return _explicit_bayesite(bayesite, "--bayesite", mode)
    env_value = os.environ.get("BAYESITE")
    if env_value:
        return _explicit_bayesite(Path(env_value), "BAYESITE", mode)
    exe = shutil.which("bayesite")
    if exe is None:
        _handle_failure("bayesite binary not found; cannot run `bayesite diagnose`", mode)
        return None
    return Path(exe)


def _explicit_bayesite(path: Path, source: str, mode: ValidateMode) -> Path | None:
    expanded = path.expanduser()
    if not expanded.exists():
        _handle_failure(
            f"{source} bayesite binary not found at {expanded}; "
            "pass a valid --bayesite path, set BAYESITE, or put bayesite on PATH",
            mode,
        )
        return None
    if not expanded.is_file():
        _handle_failure(
            f"{source} bayesite path is not a file: {expanded}; "
            "pass a valid --bayesite path, set BAYESITE, or put bayesite on PATH",
            mode,
        )
        return None
    if not os.access(expanded, os.X_OK):
        _handle_failure(
            f"{source} bayesite binary is not executable: {expanded}; "
            "make it executable or choose another binary",
            mode,
        )
        return None
    return expanded.resolve()


def _handle_failure(message: str, mode: ValidateMode) -> None:
    if mode == "warn":
        sys.stderr.write(f"warning: {message}\n")
        return
    raise ValidationError(message)
