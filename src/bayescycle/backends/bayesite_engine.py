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
    """Resolved Bayesite engine executable and advertised commands."""

    executable: Path
    commands: tuple[str, ...]


def preflight_bayesite_engine(
    engine: str, requirements: tuple[BayesiteCommandRequirement, ...]
) -> BayesiteEngineInfo:
    """Validate a Bayesite engine path and required command support."""
    executable = _resolve_engine(engine)
    commands = _engine_commands(executable)
    for requirement in requirements:
        if requirement.command not in commands:
            raise WorkflowError(_missing_command_message(executable, requirement))
    return BayesiteEngineInfo(executable=executable, commands=commands)


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


def _engine_commands(executable: Path) -> tuple[str, ...]:
    probe_text = "\n".join(
        (
            _run_engine_for_text(executable, "--help"),
            _run_engine_for_text(executable, "__bayescycle_capability_probe__"),
        )
    )
    commands = tuple(dict.fromkeys(_usage_commands(probe_text)))
    if not commands:
        raise WorkflowError(
            f"Bayesite engine did not advertise any supported commands: {executable}\n\n"
            "The binary may be stale or not the Bayesite CLI expected by bayescycle."
        )
    return commands


def _run_engine_for_text(executable: Path, *args: str) -> str:
    try:
        completed = subprocess.run(  # noqa: S603
            [str(executable), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise WorkflowError(f"cannot execute Bayesite engine {executable}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise WorkflowError(f"Bayesite engine preflight timed out: {executable}") from exc
    return f"{completed.stdout}\n{completed.stderr}"


def _usage_commands(text: str) -> tuple[str, ...]:
    usage_commands = tuple(
        match.group(1)
        for match in re.finditer(
            r"^.*?usage:\s*bayesite\s+([A-Za-z0-9-]+)(?=\s|$)",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    return (*usage_commands, *_commands_section_commands(text))


def _commands_section_commands(text: str) -> tuple[str, ...]:
    commands: list[str] = []
    in_commands = False
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower in {"commands:", "subcommands:"}:
            in_commands = True
            continue
        if not in_commands:
            continue
        if lower in {"options:", "arguments:", "usage:"}:
            break
        if not stripped:
            continue
        match = re.match(r"([A-Za-z0-9-]+)(?=\s|$)", stripped)
        if match is not None and not match.group(1).startswith("-"):
            commands.append(match.group(1))
    return tuple(commands)


def _missing_command_message(executable: Path, requirement: BayesiteCommandRequirement) -> str:
    return (
        f"Selected Bayesite engine does not support required command: {requirement.command}\n\n"
        f"engine: {executable}\n"
        f"required by stage: {requirement.stage}\n\n"
        "The binary may be stale. Rebuild Bayesite and pass the fresh binary, for example:\n"
        "  cargo build --release\n"
        "  bayescycle ... --backend bayesite --engine /path/to/bayesite/target/release/bayesite"
    )
