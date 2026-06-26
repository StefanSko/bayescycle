"""Command-line interface for bayescycle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from bayescycle import __version__
from bayescycle._backends import BayesiteBackend, Jaxstanv5Backend
from bayescycle._commands import DryRunCommand
from bayescycle._engine import run_engine
from bayescycle._errors import WorkflowError
from bayescycle._inproc import InProcessBackendError
from bayescycle._model_loader import ModelLoadError
from bayescycle._settings import SamplerSettings
from bayescycle._workflow import (
    DiagnoseRequest,
    PosteriorCheckRequest,
    PosteriorPredictiveRequest,
    PriorPredictiveBackend,
    PriorPredictiveRequest,
    RecoverCheckRequest,
    RecoverRequest,
    SampleBackend,
    SampleRequest,
    SbcRequest,
    SimulateRequest,
    dry_run_document,
    prepare_diagnose_run,
    prepare_posterior_check_run,
    prepare_posterior_predictive_run,
    prepare_prior_predictive_run,
    prepare_recover_check_run,
    prepare_recover_run,
    prepare_sample_run,
    prepare_sbc_run,
    prepare_simulate_run,
    prior_predictive_dry_run_document,
    recover_dry_run_document,
    run_command_dry_run_document,
    sbc_dry_run_document,
    simulate_dry_run_document,
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
    if command == "prior-predictive":
        return _prior_predictive(namespace)
    if command == "simulate":
        return _simulate(namespace)
    if command == "recover":
        return _recover(namespace)
    if command == "sbc":
        return _sbc(namespace)
    if command == "diagnose":
        return _diagnose(namespace)
    if command == "posterior-predictive":
        return _posterior_predictive(namespace)
    if command == "posterior-check":
        return _posterior_check(namespace)
    if command == "recover-check":
        return _recover_check(namespace)
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
        description="Compile a Python model file to IR and invoke the selected backend.",
    )
    sample.add_argument("model_path", type=Path, help="Python file containing a jaxstanv5 @model")
    sample.add_argument(
        "--model", dest="model_name", help="model class name when discovery is ambiguous"
    )
    sample.add_argument("--data", required=True, type=Path, help="JSON data file for the engine")
    sample.add_argument("-o", "--output", required=True, type=Path, help="run directory")
    sample.add_argument(
        "--backend",
        choices=("bayesite", "jaxstanv5"),
        default="bayesite",
        help="sampling backend to use",
    )
    sample.add_argument("--engine", default="bayesite", help="Bayesite executable to invoke")
    sample.add_argument("--seed", help="sampler seed forwarded to Bayesite")
    sample.add_argument("--chains", help="chain count forwarded to Bayesite")
    sample.add_argument("--warmup", help="warmup draw count forwarded to Bayesite")
    sample.add_argument("--draws", help="posterior draw count forwarded to Bayesite")
    sample.add_argument("--max-treedepth", dest="max_tree_depth", help="maximum NUTS tree depth")
    sample.add_argument("--target-accept", dest="target_accept", help="target NUTS acceptance rate")
    sample.add_argument("--force", action="store_true", help="reuse a non-empty output directory")
    sample.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare IR/data files and print the planned engine command",
    )

    prior_predictive = subparsers.add_parser(
        "prior-predictive",
        description="Generate prior-predictive draws into a fresh run directory.",
    )
    _add_model_selection_args(prior_predictive)
    prior_predictive.add_argument(
        "--data", required=True, type=Path, help="JSON declared data file for the engine"
    )
    prior_predictive.add_argument("-o", "--output", required=True, type=Path, help="run directory")
    prior_predictive.add_argument(
        "--backend",
        choices=("bayesite", "jaxstanv5"),
        default="bayesite",
        help="prior-predictive backend to use",
    )
    prior_predictive.add_argument(
        "--engine", default="bayesite", help="Bayesite executable to invoke"
    )
    prior_predictive.add_argument("--seed", help="seed forwarded to the backend")
    prior_predictive.add_argument("--draws", help="prior-predictive draw count")
    prior_predictive.add_argument(
        "--force", action="store_true", help="reuse a non-empty output directory"
    )
    prior_predictive.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare IR/data files and print the planned backend command",
    )

    simulate = subparsers.add_parser(
        "simulate",
        description="Simulate a data document from supplied truth into a fresh run directory.",
    )
    _add_model_selection_args(simulate)
    simulate.add_argument("--data", required=True, type=Path, help="JSON declared data file")
    simulate.add_argument("--truth", required=True, type=Path, help="JSON truth file")
    simulate.add_argument("-o", "--output", required=True, type=Path, help="run directory")
    simulate.add_argument(
        "--backend",
        choices=("bayesite", "jaxstanv5"),
        default="bayesite",
        help="simulation backend to use",
    )
    simulate.add_argument("--engine", default="bayesite", help="Bayesite executable to invoke")
    simulate.add_argument("--seed", help="seed forwarded to Bayesite")
    simulate.add_argument("--force", action="store_true", help="reuse a non-empty output directory")
    simulate.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare IR/data/truth files and print the planned engine command",
    )

    recover = subparsers.add_parser(
        "recover",
        description="Run a single recovery scenario into a fresh run directory.",
    )
    _add_model_selection_args(recover)
    recover.add_argument("--scenario", required=True, type=Path, help="recovery scenario JSON")
    recover.add_argument("-o", "--output", required=True, type=Path, help="run directory")
    recover.add_argument(
        "--backend",
        choices=("bayesite", "jaxstanv5"),
        default="bayesite",
        help="recovery backend to use",
    )
    recover.add_argument("--engine", default="bayesite", help="Bayesite executable to invoke")
    recover.add_argument("--force", action="store_true", help="reuse a non-empty output directory")
    recover.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare IR/scenario files and print the planned engine command",
    )

    sbc = subparsers.add_parser(
        "sbc",
        description="Run simulation-based calibration into a fresh run directory.",
    )
    _add_model_selection_args(sbc)
    sbc.add_argument("--scenario", required=True, type=Path, help="SBC scenario JSON")
    sbc.add_argument("-o", "--output", required=True, type=Path, help="run directory")
    sbc.add_argument(
        "--backend",
        choices=("bayesite", "jaxstanv5"),
        default="bayesite",
        help="SBC backend to use",
    )
    sbc.add_argument("--engine", default="bayesite", help="Bayesite executable to invoke")
    sbc.add_argument("--replicates", help="replicate count overriding the scenario")
    sbc.add_argument("--force", action="store_true", help="reuse a non-empty output directory")
    sbc.add_argument(
        "--dry-run",
        action="store_true",
        help="prepare IR/scenario files and print the planned engine command",
    )

    diagnose = subparsers.add_parser(
        "diagnose",
        description="Run Bayesite diagnostics for an existing bayescycle run directory.",
    )
    diagnose.add_argument("run_dir", type=Path, help="bayescycle run directory")
    diagnose.add_argument("--engine", default="bayesite", help="Bayesite executable to invoke")
    diagnose.add_argument(
        "--dry-run",
        action="store_true",
        help="validate run files and print the planned engine command",
    )

    posterior_predictive = subparsers.add_parser(
        "posterior-predictive",
        description="Generate posterior predictive draws for an existing bayescycle run directory.",
    )
    posterior_predictive.add_argument("run_dir", type=Path, help="bayescycle run directory")
    posterior_predictive.add_argument("--seed", required=True, help="seed forwarded to Bayesite")
    posterior_predictive.add_argument(
        "--engine", default="bayesite", help="Bayesite executable to invoke"
    )
    posterior_predictive.add_argument(
        "--dry-run",
        action="store_true",
        help="validate run files and print the planned engine command",
    )

    posterior_check = subparsers.add_parser(
        "posterior-check",
        description="Run posterior predictive checks for an existing bayescycle run directory.",
    )
    posterior_check.add_argument("run_dir", type=Path, help="bayescycle run directory")
    posterior_check.add_argument("--seed", help="seed forwarded to Bayesite")
    posterior_check.add_argument(
        "--backend",
        choices=("bayesite", "jaxstanv5"),
        default="bayesite",
        help="posterior-check backend to use",
    )
    posterior_check.add_argument(
        "--engine", default="bayesite", help="Bayesite executable to invoke"
    )
    posterior_check.add_argument(
        "--dry-run",
        action="store_true",
        help="validate run files and print the planned engine command",
    )

    recover_check = subparsers.add_parser(
        "recover-check",
        description="Compare an existing posterior run to supplied truth values.",
    )
    recover_check.add_argument("run_dir", type=Path, help="bayescycle run directory")
    recover_check.add_argument("--truth", required=True, type=Path, help="JSON truth file")
    recover_check.add_argument("--targets", type=Path, help="optional JSON recovery target map")
    recover_check.add_argument("--interval", help="equal-tailed interval probability")
    recover_check.add_argument(
        "--backend",
        choices=("bayesite", "jaxstanv5"),
        default="bayesite",
        help="recover-check backend to use",
    )
    recover_check.add_argument("--engine", default="bayesite", help="Bayesite executable to invoke")
    recover_check.add_argument(
        "--dry-run",
        action="store_true",
        help="validate run files and print the planned engine command",
    )
    return parser


def _add_model_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_path", type=Path, help="Python file containing a jaxstanv5 @model")
    parser.add_argument(
        "--model", dest="model_name", help="model class name when discovery is ambiguous"
    )


def _sample(namespace: argparse.Namespace) -> int:
    try:
        request = SampleRequest(
            model_path=cast(Path, namespace.model_path),
            data_path=cast(Path, namespace.data),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            backend=cast(str, namespace.backend),
            engine=cast(str, namespace.engine),
            sampler=SamplerSettings(
                seed=cast(str | None, namespace.seed),
                chains=cast(str | None, namespace.chains),
                warmup=cast(str | None, namespace.warmup),
                draws=cast(str | None, namespace.draws),
                max_tree_depth=cast(str | None, namespace.max_tree_depth),
                target_accept=cast(str | None, namespace.target_accept),
            ),
            engine_args=tuple(cast(list[str], namespace.engine_args)),
            force=cast(bool, namespace.force),
        )
        if request.backend == "bayesite":
            return _sample_with_backend(
                BayesiteBackend(request.engine), request, dry_run=cast(bool, namespace.dry_run)
            )
        return _sample_with_backend(
            Jaxstanv5Backend(), request, dry_run=cast(bool, namespace.dry_run)
        )
    except (InProcessBackendError, ModelLoadError, WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _sample_with_backend[CommandT: DryRunCommand](
    backend: SampleBackend[CommandT], request: SampleRequest, *, dry_run: bool
) -> int:
    prepared = prepare_sample_run(request, backend)
    if dry_run:
        print(json.dumps(dry_run_document(prepared), indent=2, sort_keys=True))
        return 0
    return backend.run_sample(prepared.command)


def _prior_predictive(namespace: argparse.Namespace) -> int:
    try:
        request = PriorPredictiveRequest(
            model_path=cast(Path, namespace.model_path),
            data_path=cast(Path, namespace.data),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            backend=cast(str, namespace.backend),
            engine=cast(str, namespace.engine),
            seed=cast(str | None, namespace.seed),
            draws=cast(str | None, namespace.draws),
            engine_args=tuple(cast(list[str], namespace.engine_args)),
            force=cast(bool, namespace.force),
        )
        if request.backend == "bayesite":
            return _prior_predictive_with_backend(
                BayesiteBackend(request.engine), request, dry_run=cast(bool, namespace.dry_run)
            )
        return _prior_predictive_with_backend(
            Jaxstanv5Backend(), request, dry_run=cast(bool, namespace.dry_run)
        )
    except (InProcessBackendError, ModelLoadError, WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _prior_predictive_with_backend[CommandT: DryRunCommand](
    backend: PriorPredictiveBackend[CommandT], request: PriorPredictiveRequest, *, dry_run: bool
) -> int:
    prepared = prepare_prior_predictive_run(request, backend)
    if dry_run:
        print(json.dumps(prior_predictive_dry_run_document(prepared), indent=2, sort_keys=True))
        return 0
    return backend.run_prior_predictive(prepared.command)


def _simulate(namespace: argparse.Namespace) -> int:
    try:
        request = SimulateRequest(
            model_path=cast(Path, namespace.model_path),
            data_path=cast(Path, namespace.data),
            truth_path=cast(Path, namespace.truth),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            backend=cast(str, namespace.backend),
            engine=cast(str, namespace.engine),
            seed=cast(str | None, namespace.seed),
            engine_args=tuple(cast(list[str], namespace.engine_args)),
            force=cast(bool, namespace.force),
        )
        backend = BayesiteBackend(request.engine)
        prepared = prepare_simulate_run(request, backend)
        if cast(bool, namespace.dry_run):
            print(json.dumps(simulate_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return backend.run_simulate(prepared.command)
    except (WorkflowError, ModelLoadError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _recover(namespace: argparse.Namespace) -> int:
    try:
        request = RecoverRequest(
            model_path=cast(Path, namespace.model_path),
            scenario_path=cast(Path, namespace.scenario),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            backend=cast(str, namespace.backend),
            engine=cast(str, namespace.engine),
            engine_args=tuple(cast(list[str], namespace.engine_args)),
            force=cast(bool, namespace.force),
        )
        backend = BayesiteBackend(request.engine)
        prepared = prepare_recover_run(request, backend)
        if cast(bool, namespace.dry_run):
            print(json.dumps(recover_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return backend.run_recover(prepared.command)
    except (WorkflowError, ModelLoadError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _sbc(namespace: argparse.Namespace) -> int:
    try:
        request = SbcRequest(
            model_path=cast(Path, namespace.model_path),
            scenario_path=cast(Path, namespace.scenario),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            backend=cast(str, namespace.backend),
            engine=cast(str, namespace.engine),
            replicates=cast(str | None, namespace.replicates),
            engine_args=tuple(cast(list[str], namespace.engine_args)),
            force=cast(bool, namespace.force),
        )
        backend = BayesiteBackend(request.engine)
        prepared = prepare_sbc_run(request, backend)
        if cast(bool, namespace.dry_run):
            print(json.dumps(sbc_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return backend.run_sbc(prepared.command)
    except (WorkflowError, ModelLoadError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _diagnose(namespace: argparse.Namespace) -> int:
    try:
        _reject_forwarded_engine_args(tuple(cast(list[str], namespace.engine_args)))
        prepared = prepare_diagnose_run(
            DiagnoseRequest(
                run_dir=cast(Path, namespace.run_dir),
                engine=cast(str, namespace.engine),
            )
        )
        if cast(bool, namespace.dry_run):
            print(json.dumps(run_command_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return run_engine(prepared.command)
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _posterior_predictive(namespace: argparse.Namespace) -> int:
    try:
        _reject_forwarded_engine_args(tuple(cast(list[str], namespace.engine_args)))
        prepared = prepare_posterior_predictive_run(
            PosteriorPredictiveRequest(
                run_dir=cast(Path, namespace.run_dir),
                engine=cast(str, namespace.engine),
                seed=cast(str, namespace.seed),
            )
        )
        if cast(bool, namespace.dry_run):
            print(json.dumps(run_command_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return run_engine(prepared.command)
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _posterior_check(namespace: argparse.Namespace) -> int:
    try:
        prepared = prepare_posterior_check_run(
            PosteriorCheckRequest(
                run_dir=cast(Path, namespace.run_dir),
                seed=cast(str | None, namespace.seed),
                backend=cast(str, namespace.backend),
                engine=cast(str, namespace.engine),
                engine_args=tuple(cast(list[str], namespace.engine_args)),
            )
        )
        if cast(bool, namespace.dry_run):
            print(json.dumps(run_command_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return run_engine(prepared.command)
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _recover_check(namespace: argparse.Namespace) -> int:
    try:
        prepared = prepare_recover_check_run(
            RecoverCheckRequest(
                run_dir=cast(Path, namespace.run_dir),
                truth_path=cast(Path, namespace.truth),
                targets_path=cast(Path | None, namespace.targets),
                interval=cast(str | None, namespace.interval),
                backend=cast(str, namespace.backend),
                engine=cast(str, namespace.engine),
                engine_args=tuple(cast(list[str], namespace.engine_args)),
            )
        )
        if cast(bool, namespace.dry_run):
            print(json.dumps(run_command_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return run_engine(prepared.command)
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _reject_forwarded_engine_args(engine_args: tuple[str, ...]) -> None:
    if engine_args:
        raise WorkflowError("engine passthrough after -- is only supported for sample")


def _split_engine_args(argv: Sequence[str] | None) -> tuple[list[str], list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" not in values:
        return values, []
    separator = values.index("--")
    return values[:separator], values[separator + 1 :]
