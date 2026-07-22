"""Command-line interface for validated Bayescycle study state."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from bayescycle_study import __version__
from bayescycle_study._documents import StudyDocumentError
from bayescycle_study._storage import apply_patch, initialize_study, validate_study


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone study CLI and return a process exit code."""

    parser = _build_parser()
    namespace = parser.parse_args(argv)
    try:
        command = cast(str, namespace.command)
        if command == "init":
            study = initialize_study(
                cast(Path, namespace.study_dir),
                study_id=cast(str, namespace.study_id),
                title=cast(str, namespace.title),
                actor=cast(str, namespace.actor),
                toolchain_profile=cast(str, namespace.toolchain_profile),
            )
            print(f"initialized study {study.state.study_id} at {study.root}")
            return 0
        if command == "validate":
            study = validate_study(cast(Path, namespace.study_dir))
            print(
                f"valid study {study.state.study_id}: "
                f"{len(study.events)} event(s), phase={study.state.phase}"
            )
            return 0
        if command == "apply":
            study = apply_patch(
                cast(Path, namespace.study_dir),
                cast(Path, namespace.patch),
                actor=cast(str, namespace.actor),
            )
            print(
                f"applied patch to {study.state.study_id}; latest event={study.events[-1].event_id}"
            )
            return 0
    except StudyDocumentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {namespace.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescycle-study",
        description="Initialize, validate, and transition Bayescycle study state.",
    )
    parser.add_argument("--version", action="version", version=f"bayescycle-study {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", description="Initialize a study directory.")
    init.add_argument("study_dir", type=Path)
    init.add_argument("--study-id", required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--actor", default="human")
    init.add_argument("--toolchain-profile", default="bayesjax-bayesite-v1")

    validate = commands.add_parser("validate", description="Validate state and append-only events.")
    validate.add_argument("study_dir", type=Path)

    apply = commands.add_parser("apply", description="Validate and apply an RFC 6902 state patch.")
    apply.add_argument("study_dir", type=Path)
    apply.add_argument("patch", type=Path)
    apply.add_argument("--actor", required=True)
    return parser
