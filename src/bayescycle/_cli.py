"""Command-line interface for bayescycle."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from bayescycle import __version__
from bayescycle._backend_plan import (
    BackendPlanRequest,
    resolve_backend_plan,
    resolve_backend_plan_file,
)
from bayescycle._backends import BayesiteBackend, Jaxstanv5Backend
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
    materialize_sample_run,
    plan_sample_run,
    prepare_diagnose_run,
    prepare_posterior_check_run,
    prepare_posterior_predictive_run,
    prepare_prior_predictive_run,
    prepare_recover_check_run,
    prepare_recover_run,
    prepare_sbc_run,
    prepare_simulate_run,
    prior_predictive_dry_run_document,
    recover_dry_run_document,
    run_command_dry_run_document,
    sample_plan_document,
    sbc_dry_run_document,
    simulate_dry_run_document,
)
from bayescycle.backends.bayesite import BayesitePreparedCommand
from bayescycle.backends.bayesite_engine import (
    BayesiteCommandRequirement,
    preflight_bayesite_engine,
)


@dataclass(frozen=True)
class ShowPlan:
    """CLI intent to show the sample plan without executing it."""


@dataclass(frozen=True)
class ExecutePlan:
    """CLI intent to execute the prepared command."""


type CliIntent = ShowPlan | ExecutePlan


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
    if command == "workflow-plan":
        return _workflow_plan(namespace)
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
    sample.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
    sample.add_argument("--seed", help="sampler seed forwarded to Bayesite")
    sample.add_argument("--chains", help="chain count forwarded to Bayesite")
    sample.add_argument("--warmup", help="warmup draw count forwarded to Bayesite")
    sample.add_argument("--draws", help="posterior draw count forwarded to Bayesite")
    sample.add_argument("--max-treedepth", dest="max_tree_depth", help="maximum NUTS tree depth")
    sample.add_argument("--target-accept", dest="target_accept", help="target NUTS acceptance rate")
    sample.add_argument("--force", action="store_true", help="reuse a non-empty output directory")
    sample.add_argument(
        "--show-plan",
        action="store_true",
        help="show the planned sample command without executing it",
    )
    sample.add_argument(
        "--dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
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
        "--engine", help="Bayesite executable to invoke (default: bayesite)"
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
    simulate.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
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
    recover.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
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
    sbc.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
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
    diagnose.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
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
        "--engine", help="Bayesite executable to invoke (default: bayesite)"
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
        "--engine", help="Bayesite executable to invoke (default: bayesite)"
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
    recover_check.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
    recover_check.add_argument(
        "--dry-run",
        action="store_true",
        help="validate run files and print the planned engine command",
    )

    workflow_plan = subparsers.add_parser(
        "workflow-plan",
        description="Resolve a multi-stage backend plan before creating run directories.",
    )
    workflow_plan.add_argument(
        "--backend",
        choices=("bayesite", "jaxstanv5"),
        help="single backend to use for all workflow stages",
    )
    workflow_plan.add_argument(
        "--simulate-backend",
        choices=("bayesite", "jaxstanv5"),
        help="backend for the simulate stage in an explicit complete mixed plan",
    )
    workflow_plan.add_argument(
        "--recover-backend",
        choices=("bayesite", "jaxstanv5"),
        help="backend for the recovery/sample stage in an explicit complete mixed plan",
    )
    workflow_plan.add_argument("--engine", help="Bayesite executable for selected bayesite stages")
    workflow_plan.add_argument("--config", type=Path, help="TOML backend-plan config")
    return parser


def _add_model_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_path", type=Path, help="Python file containing a jaxstanv5 @model")
    parser.add_argument(
        "--model", dest="model_name", help="model class name when discovery is ambiguous"
    )


def _sample(namespace: argparse.Namespace) -> int:
    try:
        explicit_engine = cast(str | None, namespace.engine)
        backend_name = cast(str, namespace.backend)
        _reject_engine_for_non_bayesite(backend_name, explicit_engine)
        request = SampleRequest(
            model_path=cast(Path, namespace.model_path),
            data_path=cast(Path, namespace.data),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            backend=backend_name,
            engine=explicit_engine or "bayesite",
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
        intent = _intent_from_flags(
            show_plan=cast(bool, namespace.show_plan),
            dry_run=cast(bool, namespace.dry_run),
        )
        if request.backend == "bayesite":
            engine = _preflight_bayesite_unless_dry_run(
                request.engine, _intent_skips_execution(intent), "sample", "sample"
            )
            return _sample_with_backend(BayesiteBackend(engine), request, intent=intent)
        return _sample_with_backend(Jaxstanv5Backend(), request, intent=intent)
    except (InProcessBackendError, ModelLoadError, WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _sample_with_backend[ActionT, CommandT](
    backend: SampleBackend[ActionT, CommandT], request: SampleRequest, *, intent: CliIntent
) -> int:
    plan = plan_sample_run(request, backend)
    match intent:
        case ShowPlan():
            print(json.dumps(sample_plan_document(plan, backend), indent=2, sort_keys=True))
            return 0
        case ExecutePlan():
            command = materialize_sample_run(plan, backend)
            return backend.execute(command)


def _prior_predictive(namespace: argparse.Namespace) -> int:
    try:
        explicit_engine = cast(str | None, namespace.engine)
        backend_name = cast(str, namespace.backend)
        _reject_engine_for_non_bayesite(backend_name, explicit_engine)
        request = PriorPredictiveRequest(
            model_path=cast(Path, namespace.model_path),
            data_path=cast(Path, namespace.data),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            backend=backend_name,
            engine=explicit_engine or "bayesite",
            seed=cast(str | None, namespace.seed),
            draws=cast(str | None, namespace.draws),
            engine_args=tuple(cast(list[str], namespace.engine_args)),
            force=cast(bool, namespace.force),
        )
        dry_run = cast(bool, namespace.dry_run)
        if request.backend == "bayesite":
            engine = _preflight_bayesite_unless_dry_run(
                request.engine, dry_run, "prior-predictive", "prior-predictive"
            )
            return _prior_predictive_with_backend(BayesiteBackend(engine), request, dry_run=dry_run)
        return _prior_predictive_with_backend(Jaxstanv5Backend(), request, dry_run=dry_run)
    except (InProcessBackendError, ModelLoadError, WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _prior_predictive_with_backend[ActionT, CommandT](
    backend: PriorPredictiveBackend[ActionT, CommandT],
    request: PriorPredictiveRequest,
    *,
    dry_run: bool,
) -> int:
    prepared = prepare_prior_predictive_run(request, backend)
    if dry_run:
        print(
            json.dumps(
                prior_predictive_dry_run_document(prepared, backend), indent=2, sort_keys=True
            )
        )
        return 0
    command = backend.materialize(prepared.action)
    return backend.execute(command)


def _simulate(namespace: argparse.Namespace) -> int:
    try:
        request = SimulateRequest(
            model_path=cast(Path, namespace.model_path),
            data_path=cast(Path, namespace.data),
            truth_path=cast(Path, namespace.truth),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            backend=cast(str, namespace.backend),
            engine=cast(str | None, namespace.engine) or "bayesite",
            seed=cast(str | None, namespace.seed),
            engine_args=tuple(cast(list[str], namespace.engine_args)),
            force=cast(bool, namespace.force),
        )
        dry_run = cast(bool, namespace.dry_run)
        engine = request.engine
        if request.backend == "bayesite":
            engine = _preflight_bayesite_unless_dry_run(
                request.engine, dry_run, "simulate", "simulate"
            )
        backend = BayesiteBackend(engine)
        prepared = prepare_simulate_run(request, backend)
        if dry_run:
            print(
                json.dumps(simulate_dry_run_document(prepared, backend), indent=2, sort_keys=True)
            )
            return 0
        command = backend.materialize(prepared.action)
        return backend.execute(command)
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
            engine=cast(str | None, namespace.engine) or "bayesite",
            engine_args=tuple(cast(list[str], namespace.engine_args)),
            force=cast(bool, namespace.force),
        )
        dry_run = cast(bool, namespace.dry_run)
        engine = request.engine
        if request.backend == "bayesite":
            engine = _preflight_bayesite_unless_dry_run(
                request.engine, dry_run, "recover", "recover"
            )
        backend = BayesiteBackend(engine)
        prepared = prepare_recover_run(request, backend)
        if dry_run:
            print(json.dumps(recover_dry_run_document(prepared, backend), indent=2, sort_keys=True))
            return 0
        command = backend.materialize(prepared.action)
        return backend.execute(command)
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
            engine=cast(str | None, namespace.engine) or "bayesite",
            replicates=cast(str | None, namespace.replicates),
            engine_args=tuple(cast(list[str], namespace.engine_args)),
            force=cast(bool, namespace.force),
        )
        dry_run = cast(bool, namespace.dry_run)
        engine = request.engine
        if request.backend == "bayesite":
            engine = _preflight_bayesite_unless_dry_run(request.engine, dry_run, "sbc", "sbc")
        backend = BayesiteBackend(engine)
        prepared = prepare_sbc_run(request, backend)
        if dry_run:
            print(json.dumps(sbc_dry_run_document(prepared, backend), indent=2, sort_keys=True))
            return 0
        command = backend.materialize(prepared.action)
        return backend.execute(command)
    except (WorkflowError, ModelLoadError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _diagnose(namespace: argparse.Namespace) -> int:
    try:
        _reject_forwarded_engine_args(tuple(cast(list[str], namespace.engine_args)))
        engine = cast(str | None, namespace.engine) or "bayesite"
        dry_run = cast(bool, namespace.dry_run)
        engine = _preflight_bayesite_unless_dry_run(engine, dry_run, "diagnose", "diagnose")
        prepared = prepare_diagnose_run(
            DiagnoseRequest(
                run_dir=cast(Path, namespace.run_dir),
                engine=engine,
            )
        )
        if dry_run:
            print(json.dumps(run_command_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return BayesiteBackend(engine).execute(BayesitePreparedCommand(prepared.command))
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _posterior_predictive(namespace: argparse.Namespace) -> int:
    try:
        _reject_forwarded_engine_args(tuple(cast(list[str], namespace.engine_args)))
        engine = cast(str | None, namespace.engine) or "bayesite"
        dry_run = cast(bool, namespace.dry_run)
        engine = _preflight_bayesite_unless_dry_run(
            engine, dry_run, "posterior-predictive", "posterior-predictive"
        )
        prepared = prepare_posterior_predictive_run(
            PosteriorPredictiveRequest(
                run_dir=cast(Path, namespace.run_dir),
                engine=engine,
                seed=cast(str, namespace.seed),
            )
        )
        if dry_run:
            print(json.dumps(run_command_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return BayesiteBackend(engine).execute(BayesitePreparedCommand(prepared.command))
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _posterior_check(namespace: argparse.Namespace) -> int:
    try:
        explicit_engine = cast(str | None, namespace.engine)
        backend_name = cast(str, namespace.backend)
        _reject_engine_for_non_bayesite(backend_name, explicit_engine)
        engine = explicit_engine or "bayesite"
        dry_run = cast(bool, namespace.dry_run)
        if backend_name == "bayesite":
            engine = _preflight_bayesite_unless_dry_run(
                engine, dry_run, "posterior-check", "posterior-check"
            )
        prepared = prepare_posterior_check_run(
            PosteriorCheckRequest(
                run_dir=cast(Path, namespace.run_dir),
                seed=cast(str | None, namespace.seed),
                backend=backend_name,
                engine=engine,
                engine_args=tuple(cast(list[str], namespace.engine_args)),
            )
        )
        if dry_run:
            print(json.dumps(run_command_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return BayesiteBackend(engine).execute(BayesitePreparedCommand(prepared.command))
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _recover_check(namespace: argparse.Namespace) -> int:
    try:
        explicit_engine = cast(str | None, namespace.engine)
        backend_name = cast(str, namespace.backend)
        _reject_engine_for_non_bayesite(backend_name, explicit_engine)
        engine = explicit_engine or "bayesite"
        dry_run = cast(bool, namespace.dry_run)
        if backend_name == "bayesite":
            engine = _preflight_bayesite_unless_dry_run(
                engine, dry_run, "recover-check", "recover-check"
            )
        prepared = prepare_recover_check_run(
            RecoverCheckRequest(
                run_dir=cast(Path, namespace.run_dir),
                truth_path=cast(Path, namespace.truth),
                targets_path=cast(Path | None, namespace.targets),
                interval=cast(str | None, namespace.interval),
                backend=backend_name,
                engine=engine,
                engine_args=tuple(cast(list[str], namespace.engine_args)),
            )
        )
        if dry_run:
            print(json.dumps(run_command_dry_run_document(prepared), indent=2, sort_keys=True))
            return 0
        return BayesiteBackend(engine).execute(BayesitePreparedCommand(prepared.command))
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _workflow_plan(namespace: argparse.Namespace) -> int:
    try:
        _reject_forwarded_engine_args(tuple(cast(list[str], namespace.engine_args)))
        config = cast(Path | None, namespace.config)
        has_cli_plan = any(
            value is not None
            for value in (
                cast(str | None, namespace.backend),
                cast(str | None, namespace.simulate_backend),
                cast(str | None, namespace.recover_backend),
            )
        )
        if config is not None:
            if has_cli_plan or cast(str | None, namespace.engine) is not None:
                raise WorkflowError(
                    "--config cannot be combined with CLI backend plan or --engine options"
                )
            resolved = resolve_backend_plan_file(config)
        else:
            resolved = resolve_backend_plan(
                BackendPlanRequest(
                    backend=cast(str | None, namespace.backend),
                    simulate_backend=cast(str | None, namespace.simulate_backend),
                    recover_backend=cast(str | None, namespace.recover_backend),
                    engine=cast(str | None, namespace.engine),
                )
            )
        print(json.dumps(resolved.as_json(), indent=2, sort_keys=True))
        return 0
    except WorkflowError as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _intent_from_flags(*, show_plan: bool = False, dry_run: bool = False) -> CliIntent:
    if show_plan or dry_run:
        return ShowPlan()
    return ExecutePlan()


def _intent_skips_execution(intent: CliIntent) -> bool:
    return isinstance(intent, ShowPlan)


def _preflight_bayesite_unless_dry_run(engine: str, dry_run: bool, command: str, stage: str) -> str:
    if _intent_skips_execution(_intent_from_flags(dry_run=dry_run)):
        return engine
    info = preflight_bayesite_engine(engine, (BayesiteCommandRequirement(command, stage),))
    return str(info.executable)


def _reject_forwarded_engine_args(engine_args: tuple[str, ...]) -> None:
    if engine_args:
        raise WorkflowError("engine passthrough after -- is only supported for sample")


def _reject_engine_for_non_bayesite(backend: str, explicit_engine: str | None) -> None:
    if backend != "bayesite" and explicit_engine is not None:
        raise WorkflowError(
            "--engine configures the bayesite backend, but no bayesite backend stage was selected. "
            "Did you mean --backend bayesite?"
        )


def _split_engine_args(argv: Sequence[str] | None) -> tuple[list[str], list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" not in values:
        return values, []
    separator = values.index("--")
    return values[:separator], values[separator + 1 :]
