"""Command-line interface for bayescycle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from bayescycle import __version__
from bayescycle._engine import run_engine
from bayescycle._model_loader import ModelLoadError
from bayescycle._workflow import (
    SampleRequest,
    SamplerSettings,
    WorkflowError,
    dry_run_document,
    prepare_sample_run,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = _build_parser()
    known_args, engine_args = _split_engine_args(argv)
    namespace = parser.parse_args(known_args)
    namespace.engine_args = engine_args
    command = cast(str, namespace.command)
    if command == "sample":
        return _sample(namespace)
    parser.print_help(sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescycle",
        description="Workflow CLI from jaxstanv5 Python models to Bayesite engine runs.",
    )
    parser.add_argument("--version", action="version", version=f"bayescycle {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser(
        "sample",
        description="Compile a Python model file to IR and invoke the Bayesite engine.",
    )
    sample.add_argument("model_path", type=Path, help="Python file containing a jaxstanv5 @model")
    sample.add_argument(
        "--model", dest="model_name", help="model class name when discovery is ambiguous"
    )
    sample.add_argument("--data", required=True, type=Path, help="JSON data file for the engine")
    sample.add_argument("-o", "--output", required=True, type=Path, help="run directory")
    sample.add_argument("--engine", default="bayesite", help="Bayesite executable to invoke")
    sample.add_argument("--seed", help="sampler seed forwarded to Bayesite")
    sample.add_argument("--chains", help="chain count forwarded to Bayesite")
    sample.add_argument("--warmup", help="warmup draw count forwarded to Bayesite")
    sample.add_argument("--draws", help="posterior draw count forwarded to Bayesite")
    sample.add_argument("--force", action="store_true", help="reuse a non-empty output directory")
    sample.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare IR/data files and print the planned engine command",
    )
    return parser


def _sample(namespace: argparse.Namespace) -> int:
    try:
        request = SampleRequest(
            model_path=cast(Path, namespace.model_path),
            data_path=cast(Path, namespace.data),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            engine=cast(str, namespace.engine),
            sampler=SamplerSettings(
                seed=cast(str | None, namespace.seed),
                chains=cast(str | None, namespace.chains),
                warmup=cast(str | None, namespace.warmup),
                draws=cast(str | None, namespace.draws),
            ),
            engine_args=tuple(cast(list[str], namespace.engine_args)),
            force=cast(bool, namespace.force),
        )
        prepared = prepare_sample_run(request)
        if cast(bool, namespace.dry_run):
            print(json.dumps(dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return run_engine(prepared.engine_command)
    except (ModelLoadError, WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _split_engine_args(argv: Sequence[str] | None) -> tuple[list[str], list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" not in values:
        return values, []
    separator = values.index("--")
    return values[:separator], values[separator + 1 :]
