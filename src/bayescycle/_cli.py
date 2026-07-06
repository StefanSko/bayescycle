"""Command-line interface for bayescycle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from bayescycle import __version__
from bayescycle._backend_runtime import (
    BackendRuntimeOptions,
    resolve_bayesite_engine_path,
    resolve_diagnose_backend,
    resolve_posterior_check_backend,
    resolve_posterior_predictive_backend,
    resolve_prior_predictive_backend,
    resolve_recover_backend,
    resolve_recover_check_backend,
    resolve_sample_backend,
    resolve_sbc_backend,
    resolve_simulate_backend,
)
from bayescycle._errors import WorkflowError
from bayescycle._model_loader import ModelLoadError
from bayescycle._run_artifacts.run_metadata import RecordedRunMetadata
from bayescycle._settings import SamplerSettings
from bayescycle._workflow.backend_plan import (
    BackendPlanRequest,
    resolve_backend_plan,
    resolve_backend_plan_file,
)
from bayescycle._workflow.capabilities import BAYESITE, FIRST_PARTY_BACKENDS
from bayescycle._workflow.documents import (
    prior_predictive_plan_document,
    recover_plan_document,
    run_command_plan_document,
    sample_plan_document,
    sbc_plan_document,
    simulate_plan_document,
)
from bayescycle._workflow.operations import (
    materialize_prior_predictive_run,
    materialize_recover_run,
    materialize_run_directory_command,
    materialize_sample_run,
    materialize_sbc_run,
    materialize_simulate_run,
    plan_diagnose_run,
    plan_posterior_check_run,
    plan_posterior_predictive_run,
    plan_prior_predictive_run,
    plan_recover_check_run,
    plan_recover_run,
    plan_sample_run,
    plan_sbc_run,
    plan_simulate_run,
)
from bayescycle._workflow.plans import RunDirectoryCommandPlan
from bayescycle._workflow.protocols import (
    ActionBackend,
    PriorPredictiveBackend,
    RecoverBackend,
    SampleBackend,
    SbcBackend,
    SimulateBackend,
)
from bayescycle._workflow.replay import (
    ReplaySourceCheck,
    compare_replay_artifacts,
    load_replay_metadata,
    prior_predictive_request_from_replay,
    recover_request_from_replay,
    replay_plan_document,
    replay_result_document,
    sample_request_from_replay,
    sbc_request_from_replay,
    simulate_request_from_replay,
    verify_replay_sources,
)
from bayescycle._workflow.requests import (
    DiagnoseRequest,
    PosteriorCheckRequest,
    PosteriorPredictiveRequest,
    PriorPredictiveRequest,
    RecoverCheckRequest,
    RecoverRequest,
    SampleRequest,
    SbcRequest,
    SimulateRequest,
)
from bayescycle.backends.bayesite.preflight import preflight_bayesite_engine
from bayescycle.backends.bayesite.provisioning import (
    PINNED_ENGINE_RELEASE,
    EngineRelease,
    cached_engine_path,
    ensure_engine,
)
from bayescycle.backends.bayesite_viz.uvx_runner import (
    BAYESITE_VIZ_SOURCE,
    VIZ_VERBS,
    IdataOptions,
    PlotOptions,
    default_fit_path,
    run_idata,
    run_plot,
)
from bayescycle.backends.jaxstanv5.runner import InProcessBackendError


@dataclass(frozen=True)
class ShowPlan:
    """CLI intent to show a plan without executing it."""


@dataclass(frozen=True)
class ExecutePlan:
    """CLI intent to materialize and execute a plan."""


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
    if command == "replay":
        return _replay(namespace)
    if command == "workflow-plan":
        return _workflow_plan(namespace)
    if command == "engine":
        return _engine(namespace)
    if command == "idata":
        return _idata(namespace)
    if command == "plot":
        return _plot(namespace)
    parser.print_help(sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayescycle",
        description="Workflow CLI from bayeswire Python models to Bayesite engine runs.",
    )
    parser.add_argument("--version", action="version", version=f"bayescycle {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser(
        "sample",
        description="Compile a Python model file to IR and invoke the selected backend.",
    )
    sample.add_argument("model_path", type=Path, help="Python file containing a bayeswire @model")
    sample.add_argument(
        "--model", dest="model_name", help="model class name when discovery is ambiguous"
    )
    sample.add_argument("--data", required=True, type=Path, help="JSON data file for the engine")
    sample.add_argument("-o", "--output", required=True, type=Path, help="run directory")
    sample.add_argument(
        "--backend",
        choices=FIRST_PARTY_BACKENDS.choices(),
        default=str(BAYESITE),
        help="sampling backend to use",
    )
    sample.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
    _add_no_auto_provision_argument(sample)
    sample.add_argument("--seed", help="sampler seed forwarded to Bayesite")
    sample.add_argument("--chains", help="chain count forwarded to Bayesite")
    sample.add_argument("--warmup", help="warmup draw count forwarded to Bayesite")
    sample.add_argument("--draws", help="posterior draw count forwarded to Bayesite")
    sample.add_argument("--max-treedepth", dest="max_tree_depth", help="maximum NUTS tree depth")
    sample.add_argument("--target-accept", dest="target_accept", help="target NUTS acceptance rate")
    _add_show_plan_argument(
        sample,
        help_text="show the planned sample command without executing it",
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
        choices=FIRST_PARTY_BACKENDS.choices(),
        default=str(BAYESITE),
        help="prior-predictive backend to use",
    )
    prior_predictive.add_argument(
        "--engine", help="Bayesite executable to invoke (default: bayesite)"
    )
    _add_no_auto_provision_argument(prior_predictive)
    prior_predictive.add_argument("--seed", help="seed forwarded to the backend")
    prior_predictive.add_argument("--draws", help="prior-predictive draw count")
    _add_show_plan_argument(
        prior_predictive,
        help_text="show the planned backend command without executing it",
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
        choices=FIRST_PARTY_BACKENDS.choices(),
        default=str(BAYESITE),
        help="simulation backend to use",
    )
    simulate.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
    _add_no_auto_provision_argument(simulate)
    simulate.add_argument("--seed", help="seed forwarded to Bayesite")
    _add_show_plan_argument(
        simulate,
        help_text="show the planned backend command without executing it",
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
        choices=FIRST_PARTY_BACKENDS.choices(),
        default=str(BAYESITE),
        help="recovery backend to use",
    )
    recover.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
    _add_no_auto_provision_argument(recover)
    _add_show_plan_argument(
        recover,
        help_text="show the planned backend command without executing it",
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
        choices=FIRST_PARTY_BACKENDS.choices(),
        default=str(BAYESITE),
        help="SBC backend to use",
    )
    sbc.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
    _add_no_auto_provision_argument(sbc)
    sbc.add_argument("--replicates", help="replicate count overriding the scenario")
    _add_show_plan_argument(
        sbc,
        help_text="show the planned backend command without executing it",
    )

    diagnose = subparsers.add_parser(
        "diagnose",
        description="Run Bayesite diagnostics for an existing bayescycle run directory.",
    )
    diagnose.add_argument("run_dir", type=Path, help="bayescycle run directory")
    diagnose.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
    _add_no_auto_provision_argument(diagnose)
    _add_show_plan_argument(
        diagnose,
        help_text="validate run files and show the planned engine command",
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
    _add_no_auto_provision_argument(posterior_predictive)
    _add_show_plan_argument(
        posterior_predictive,
        help_text="validate run files and show the planned engine command",
    )

    posterior_check = subparsers.add_parser(
        "posterior-check",
        description="Run posterior predictive checks for an existing bayescycle run directory.",
    )
    posterior_check.add_argument("run_dir", type=Path, help="bayescycle run directory")
    posterior_check.add_argument("--seed", help="seed forwarded to Bayesite")
    posterior_check.add_argument(
        "--backend",
        choices=FIRST_PARTY_BACKENDS.choices(),
        default=str(BAYESITE),
        help="posterior-check backend to use",
    )
    posterior_check.add_argument(
        "--engine", help="Bayesite executable to invoke (default: bayesite)"
    )
    _add_no_auto_provision_argument(posterior_check)
    _add_show_plan_argument(
        posterior_check,
        help_text="validate run files and show the planned backend command",
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
        choices=FIRST_PARTY_BACKENDS.choices(),
        default=str(BAYESITE),
        help="recover-check backend to use",
    )
    recover_check.add_argument("--engine", help="Bayesite executable to invoke (default: bayesite)")
    _add_no_auto_provision_argument(recover_check)
    _add_show_plan_argument(
        recover_check,
        help_text="validate run files and show the planned backend command",
    )

    replay = subparsers.add_parser(
        "replay",
        description="Re-execute a run from run.json and compare replayed artifacts.",
    )
    replay.add_argument("run_dir", type=Path, help="bayescycle run directory containing run.json")
    replay.add_argument("-o", "--output", required=True, type=Path, help="fresh replay directory")
    replay.add_argument(
        "--engine",
        help="Bayesite executable to invoke when replaying a bayesite run (default: bayesite)",
    )
    _add_no_auto_provision_argument(replay)
    replay.add_argument(
        "--check-only",
        action="store_true",
        help="verify recorded source hashes and print the reconstructed plan without executing",
    )

    workflow_plan = subparsers.add_parser(
        "workflow-plan",
        description="Resolve a multi-stage backend plan before creating run directories.",
    )
    workflow_plan.add_argument(
        "--backend",
        choices=FIRST_PARTY_BACKENDS.choices(),
        help="single backend to use for all workflow stages",
    )
    workflow_plan.add_argument(
        "--simulate-backend",
        choices=FIRST_PARTY_BACKENDS.choices(),
        help="backend for the simulate stage in an explicit complete mixed plan",
    )
    workflow_plan.add_argument(
        "--recover-backend",
        choices=FIRST_PARTY_BACKENDS.choices(),
        help="backend for the recovery/sample stage in an explicit complete mixed plan",
    )
    workflow_plan.add_argument("--engine", help="Bayesite executable for selected bayesite stages")
    workflow_plan.add_argument("--config", type=Path, help="TOML backend-plan config")

    engine = subparsers.add_parser(
        "engine",
        description="Manage the pinned Bayesite engine binary used by bayescycle.",
    )
    engine_subparsers = engine.add_subparsers(dest="engine_command", required=True)

    engine_ensure = engine_subparsers.add_parser(
        "ensure",
        description="Provision the pinned Bayesite engine into the local cache if missing.",
    )
    engine_ensure.add_argument(
        "--force",
        action="store_true",
        help="re-download and re-verify even if already cached",
    )
    engine_ensure.add_argument(
        "--cache-root",
        type=Path,
        help="cache directory to install under (default: the bayescycle cache directory)",
    )
    engine_ensure.add_argument(
        "--base-url",
        help="override the pinned release base URL (for airgapped/mirrored release hosting)",
    )

    engine_path = engine_subparsers.add_parser(
        "path",
        description="Print the resolved Bayesite engine path without downloading anything.",
    )
    engine_path.add_argument(
        "--cache-root",
        type=Path,
        help="cache directory to look under (default: the bayescycle cache directory)",
    )

    engine_info = engine_subparsers.add_parser(
        "info",
        description="Print structured Bayesite engine capabilities as JSON.",
    )
    engine_info.add_argument(
        "--engine",
        help="Bayesite executable to inspect (default: PATH, then the bayescycle cache)",
    )
    engine_info.add_argument(
        "--cache-root",
        type=Path,
        help="cache directory to look under (default: the bayescycle cache directory)",
    )

    idata = subparsers.add_parser(
        "idata",
        description=(
            "Export a bayescycle run directory to an ArviZ NetCDF fit file via bayesite-viz."
        ),
    )
    idata.add_argument("run_dir", type=Path, help="bayescycle run directory")
    idata.add_argument(
        "-o", "--output", type=Path, help="output .nc path (default: RUN_DIR/fit.nc)"
    )
    idata.add_argument(
        "--validate",
        choices=("require", "warn", "skip"),
        help="bayesite-idata validation mode (default: bayesite-idata's own default, warn)",
    )
    idata.add_argument(
        "--engine", help="Bayesite executable forwarded as --bayesite (default: bayesite)"
    )
    _add_no_auto_provision_argument(idata)
    idata.add_argument("--viz-source", help=argparse.SUPPRESS)

    plot = subparsers.add_parser(
        "plot",
        description="Render a bayesite-viz plot from a bayescycle run directory.",
    )
    plot.add_argument("verb", choices=VIZ_VERBS, help="bayesite-viz plot verb")
    plot.add_argument("run_dir", type=Path, help="bayescycle run directory")
    plot.add_argument("-o", "--output", type=Path, help="output image path")
    plot.add_argument(
        "--fit", dest="fit_path", type=Path, help="fit .nc path (default: RUN_DIR/fit.nc)"
    )
    plot.add_argument("--kind", help="plot kind (posterior, ppc verbs only)")
    plot.add_argument(
        "-f", "--format", dest="output_format", help="bayesite-viz stdout announce format"
    )
    plot.add_argument("--var", dest="variables", action="append", help="variable name; repeatable")
    plot.add_argument(
        "--coords", dest="coords", action="append", help="coords key=value; repeatable"
    )
    plot.add_argument("-b", "--backend", dest="viz_backend", help="matplotlib|bokeh|plotly")
    plot.add_argument("--svg", action="store_true", help="emit SVG instead of PNG")
    plot.add_argument(
        "--no-auto-idata",
        dest="no_auto_idata",
        action="store_true",
        help="do not auto-run `bayescycle idata` when RUN_DIR/fit.nc (or --fit) is missing",
    )
    plot.add_argument(
        "--engine",
        help="Bayesite executable forwarded to the auto-run idata step (default: bayesite)",
    )
    _add_no_auto_provision_argument(plot)
    plot.add_argument("--viz-source", help=argparse.SUPPRESS)

    return parser


def _add_model_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model_path", type=Path, help="Python file containing a bayeswire @model")
    parser.add_argument(
        "--model", dest="model_name", help="model class name when discovery is ambiguous"
    )


def _add_no_auto_provision_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-auto-provision",
        dest="no_auto_provision",
        action="store_true",
        help=(
            "do not auto-download the pinned Bayesite engine when --engine is unset and "
            "bayesite is not found on PATH; the same effect as setting "
            "BAYESCYCLE_NO_AUTO_PROVISION=1 (this flag takes precedence over the env var)"
        ),
    )


def _auto_provision_enabled(
    namespace: argparse.Namespace, env: Mapping[str, str] | None = None
) -> bool:
    """CLI-boundary auto-provision decision.

    ``--no-auto-provision`` wins over ``BAYESCYCLE_NO_AUTO_PROVISION=1``;
    ``env`` defaults to the real process environment and is only overridden
    directly in tests.
    """
    if bool(getattr(namespace, "no_auto_provision", False)):
        return False
    resolved_env = env if env is not None else os.environ
    return resolved_env.get("BAYESCYCLE_NO_AUTO_PROVISION") != "1"


def _add_show_plan_argument(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--show-plan",
        dest="show_plan",
        action="store_true",
        help=help_text,
    )
    parser.add_argument(
        "--dry-run",
        dest="show_plan",
        action="store_true",
        help=argparse.SUPPRESS,
    )


def _sample(namespace: argparse.Namespace) -> int:
    try:
        intent = _intent_from_namespace(namespace)
        request = SampleRequest(
            model_path=cast(Path, namespace.model_path),
            data_path=cast(Path, namespace.data),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            sampler=SamplerSettings(
                seed=cast(str | None, namespace.seed),
                chains=cast(str | None, namespace.chains),
                warmup=cast(str | None, namespace.warmup),
                draws=cast(str | None, namespace.draws),
                max_tree_depth=cast(str | None, namespace.max_tree_depth),
                target_accept=cast(str | None, namespace.target_accept),
            ),
        )
        backend = resolve_sample_backend(
            BackendRuntimeOptions(
                backend=cast(str, namespace.backend),
                engine=cast(str | None, namespace.engine),
                extra_args=tuple(cast(list[str], namespace.engine_args)),
                preflight=not _intent_skips_execution(intent),
                auto_provision=_auto_provision_enabled(namespace),
            )
        )
        return _sample_with_backend(backend, request, intent=intent)
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
        intent = _intent_from_namespace(namespace)
        request = PriorPredictiveRequest(
            model_path=cast(Path, namespace.model_path),
            data_path=cast(Path, namespace.data),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            seed=cast(str | None, namespace.seed),
            draws=cast(str | None, namespace.draws),
        )
        backend = resolve_prior_predictive_backend(
            BackendRuntimeOptions(
                backend=cast(str, namespace.backend),
                engine=cast(str | None, namespace.engine),
                extra_args=tuple(cast(list[str], namespace.engine_args)),
                preflight=not _intent_skips_execution(intent),
                auto_provision=_auto_provision_enabled(namespace),
            )
        )
        return _prior_predictive_with_backend(backend, request, intent=intent)
    except (InProcessBackendError, ModelLoadError, WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _prior_predictive_with_backend[ActionT, CommandT](
    backend: PriorPredictiveBackend[ActionT, CommandT],
    request: PriorPredictiveRequest,
    *,
    intent: CliIntent,
) -> int:
    plan = plan_prior_predictive_run(request, backend)
    match intent:
        case ShowPlan():
            print(
                json.dumps(prior_predictive_plan_document(plan, backend), indent=2, sort_keys=True)
            )
            return 0
        case ExecutePlan():
            command = materialize_prior_predictive_run(plan, backend)
            return backend.execute(command)


def _simulate(namespace: argparse.Namespace) -> int:
    try:
        intent = _intent_from_namespace(namespace)
        request = SimulateRequest(
            model_path=cast(Path, namespace.model_path),
            data_path=cast(Path, namespace.data),
            truth_path=cast(Path, namespace.truth),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            seed=cast(str | None, namespace.seed),
        )
        backend = resolve_simulate_backend(
            BackendRuntimeOptions(
                backend=cast(str, namespace.backend),
                engine=cast(str | None, namespace.engine),
                extra_args=tuple(cast(list[str], namespace.engine_args)),
                preflight=not _intent_skips_execution(intent),
                auto_provision=_auto_provision_enabled(namespace),
            )
        )
        plan = plan_simulate_run(request, backend)
        match intent:
            case ShowPlan():
                print(json.dumps(simulate_plan_document(plan, backend), indent=2, sort_keys=True))
                return 0
            case ExecutePlan():
                command = materialize_simulate_run(plan, backend)
                return backend.execute(command)
    except (WorkflowError, ModelLoadError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _recover(namespace: argparse.Namespace) -> int:
    try:
        intent = _intent_from_namespace(namespace)
        request = RecoverRequest(
            model_path=cast(Path, namespace.model_path),
            scenario_path=cast(Path, namespace.scenario),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
        )
        backend = resolve_recover_backend(
            BackendRuntimeOptions(
                backend=cast(str, namespace.backend),
                engine=cast(str | None, namespace.engine),
                extra_args=tuple(cast(list[str], namespace.engine_args)),
                preflight=not _intent_skips_execution(intent),
                auto_provision=_auto_provision_enabled(namespace),
            )
        )
        plan = plan_recover_run(request, backend)
        match intent:
            case ShowPlan():
                print(json.dumps(recover_plan_document(plan, backend), indent=2, sort_keys=True))
                return 0
            case ExecutePlan():
                command = materialize_recover_run(plan, backend)
                return backend.execute(command)
    except (WorkflowError, ModelLoadError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _sbc(namespace: argparse.Namespace) -> int:
    try:
        intent = _intent_from_namespace(namespace)
        request = SbcRequest(
            model_path=cast(Path, namespace.model_path),
            scenario_path=cast(Path, namespace.scenario),
            output_dir=cast(Path, namespace.output),
            model_name=cast(str | None, namespace.model_name),
            replicates=cast(str | None, namespace.replicates),
        )
        backend = resolve_sbc_backend(
            BackendRuntimeOptions(
                backend=cast(str, namespace.backend),
                engine=cast(str | None, namespace.engine),
                extra_args=tuple(cast(list[str], namespace.engine_args)),
                preflight=not _intent_skips_execution(intent),
                auto_provision=_auto_provision_enabled(namespace),
            )
        )
        plan = plan_sbc_run(request, backend)
        match intent:
            case ShowPlan():
                print(json.dumps(sbc_plan_document(plan, backend), indent=2, sort_keys=True))
                return 0
            case ExecutePlan():
                command = materialize_sbc_run(plan, backend)
                return backend.execute(command)
    except (WorkflowError, ModelLoadError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _diagnose(namespace: argparse.Namespace) -> int:
    try:
        _reject_forwarded_engine_args(tuple(cast(list[str], namespace.engine_args)))
        intent = _intent_from_namespace(namespace)
        backend = resolve_diagnose_backend(
            BackendRuntimeOptions(
                backend=str(BAYESITE),
                engine=cast(str | None, namespace.engine),
                extra_args=(),
                preflight=not _intent_skips_execution(intent),
                auto_provision=_auto_provision_enabled(namespace),
            )
        )
        plan = plan_diagnose_run(
            DiagnoseRequest(run_dir=cast(Path, namespace.run_dir)),
            backend,
        )
        return _run_directory_with_backend(plan, backend, intent=intent)
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _posterior_predictive(namespace: argparse.Namespace) -> int:
    try:
        _reject_forwarded_engine_args(tuple(cast(list[str], namespace.engine_args)))
        intent = _intent_from_namespace(namespace)
        backend = resolve_posterior_predictive_backend(
            BackendRuntimeOptions(
                backend=str(BAYESITE),
                engine=cast(str | None, namespace.engine),
                extra_args=(),
                preflight=not _intent_skips_execution(intent),
                auto_provision=_auto_provision_enabled(namespace),
            )
        )
        plan = plan_posterior_predictive_run(
            PosteriorPredictiveRequest(
                run_dir=cast(Path, namespace.run_dir),
                seed=cast(str, namespace.seed),
            ),
            backend,
        )
        return _run_directory_with_backend(plan, backend, intent=intent)
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _posterior_check(namespace: argparse.Namespace) -> int:
    try:
        intent = _intent_from_namespace(namespace)
        backend = resolve_posterior_check_backend(
            BackendRuntimeOptions(
                backend=cast(str, namespace.backend),
                engine=cast(str | None, namespace.engine),
                extra_args=tuple(cast(list[str], namespace.engine_args)),
                preflight=not _intent_skips_execution(intent),
                auto_provision=_auto_provision_enabled(namespace),
            )
        )
        plan = plan_posterior_check_run(
            PosteriorCheckRequest(
                run_dir=cast(Path, namespace.run_dir),
                seed=cast(str | None, namespace.seed),
            ),
            backend,
        )
        return _run_directory_with_backend(plan, backend, intent=intent)
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _recover_check(namespace: argparse.Namespace) -> int:
    try:
        intent = _intent_from_namespace(namespace)
        backend = resolve_recover_check_backend(
            BackendRuntimeOptions(
                backend=cast(str, namespace.backend),
                engine=cast(str | None, namespace.engine),
                extra_args=tuple(cast(list[str], namespace.engine_args)),
                preflight=not _intent_skips_execution(intent),
                auto_provision=_auto_provision_enabled(namespace),
            )
        )
        plan = plan_recover_check_run(
            RecoverCheckRequest(
                run_dir=cast(Path, namespace.run_dir),
                truth_path=cast(Path, namespace.truth),
                targets_path=cast(Path | None, namespace.targets),
                interval=cast(str | None, namespace.interval),
            ),
            backend,
        )
        return _run_directory_with_backend(plan, backend, intent=intent)
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _replay(namespace: argparse.Namespace) -> int:
    try:
        _reject_forwarded_engine_args(tuple(cast(list[str], namespace.engine_args)))
        source_run, record = load_replay_metadata(cast(Path, namespace.run_dir))
        source_checks = verify_replay_sources(record)
        output_dir = cast(Path, namespace.output).expanduser().resolve()
        check_only = bool(cast(bool, namespace.check_only))
        runtime_options = BackendRuntimeOptions(
            backend=record.backend,
            engine=cast(str | None, namespace.engine),
            extra_args=record.backend_extra_args,
            preflight=not check_only,
            auto_provision=_auto_provision_enabled(namespace),
        )
        if record.kind == "sample":
            backend = resolve_sample_backend(runtime_options)
            request = sample_request_from_replay(record, output_dir)
            return _replay_sample_with_backend(
                backend,
                request,
                source_run=source_run,
                record=record,
                source_checks=source_checks,
                check_only=check_only,
            )
        if record.kind == "prior-predictive":
            backend = resolve_prior_predictive_backend(runtime_options)
            request = prior_predictive_request_from_replay(record, output_dir)
            return _replay_prior_predictive_with_backend(
                backend,
                request,
                source_run=source_run,
                record=record,
                source_checks=source_checks,
                check_only=check_only,
            )
        if record.kind == "simulate":
            backend = resolve_simulate_backend(runtime_options)
            request = simulate_request_from_replay(record, output_dir)
            return _replay_simulate_with_backend(
                backend,
                request,
                source_run=source_run,
                record=record,
                source_checks=source_checks,
                check_only=check_only,
            )
        if record.kind == "recover":
            backend = resolve_recover_backend(runtime_options)
            request = recover_request_from_replay(record, output_dir)
            return _replay_recover_with_backend(
                backend,
                request,
                source_run=source_run,
                record=record,
                source_checks=source_checks,
                check_only=check_only,
            )
        if record.kind == "sbc":
            backend = resolve_sbc_backend(runtime_options)
            request = sbc_request_from_replay(record, output_dir)
            return _replay_sbc_with_backend(
                backend,
                request,
                source_run=source_run,
                record=record,
                source_checks=source_checks,
                check_only=check_only,
            )
        raise WorkflowError(f"replay does not support run kind: {record.kind}")
    except (InProcessBackendError, ModelLoadError, WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _replay_sample_with_backend[ActionT, CommandT](
    backend: SampleBackend[ActionT, CommandT],
    request: SampleRequest,
    *,
    source_run: Path,
    record: RecordedRunMetadata,
    source_checks: tuple[ReplaySourceCheck, ...],
    check_only: bool,
) -> int:
    plan = plan_sample_run(request, backend)
    if check_only:
        return _print_replay_plan(
            source_run=source_run,
            output_dir=request.output_dir.expanduser().resolve(),
            record=record,
            source_checks=source_checks,
            plan=sample_plan_document(plan, backend),
        )
    command = materialize_sample_run(plan, backend)
    code = backend.execute(command)
    if code != 0:
        return code
    return _print_replay_result(
        source_run=source_run,
        output_dir=request.output_dir.expanduser().resolve(),
        record=record,
        source_checks=source_checks,
    )


def _replay_prior_predictive_with_backend[ActionT, CommandT](
    backend: PriorPredictiveBackend[ActionT, CommandT],
    request: PriorPredictiveRequest,
    *,
    source_run: Path,
    record: RecordedRunMetadata,
    source_checks: tuple[ReplaySourceCheck, ...],
    check_only: bool,
) -> int:
    plan = plan_prior_predictive_run(request, backend)
    if check_only:
        return _print_replay_plan(
            source_run=source_run,
            output_dir=request.output_dir.expanduser().resolve(),
            record=record,
            source_checks=source_checks,
            plan=prior_predictive_plan_document(plan, backend),
        )
    command = materialize_prior_predictive_run(plan, backend)
    code = backend.execute(command)
    if code != 0:
        return code
    return _print_replay_result(
        source_run=source_run,
        output_dir=request.output_dir.expanduser().resolve(),
        record=record,
        source_checks=source_checks,
    )


def _replay_simulate_with_backend[ActionT, CommandT](
    backend: SimulateBackend[ActionT, CommandT],
    request: SimulateRequest,
    *,
    source_run: Path,
    record: RecordedRunMetadata,
    source_checks: tuple[ReplaySourceCheck, ...],
    check_only: bool,
) -> int:
    plan = plan_simulate_run(request, backend)
    if check_only:
        return _print_replay_plan(
            source_run=source_run,
            output_dir=request.output_dir.expanduser().resolve(),
            record=record,
            source_checks=source_checks,
            plan=simulate_plan_document(plan, backend),
        )
    command = materialize_simulate_run(plan, backend)
    code = backend.execute(command)
    if code != 0:
        return code
    return _print_replay_result(
        source_run=source_run,
        output_dir=request.output_dir.expanduser().resolve(),
        record=record,
        source_checks=source_checks,
    )


def _replay_recover_with_backend[ActionT, CommandT](
    backend: RecoverBackend[ActionT, CommandT],
    request: RecoverRequest,
    *,
    source_run: Path,
    record: RecordedRunMetadata,
    source_checks: tuple[ReplaySourceCheck, ...],
    check_only: bool,
) -> int:
    plan = plan_recover_run(request, backend)
    if check_only:
        return _print_replay_plan(
            source_run=source_run,
            output_dir=request.output_dir.expanduser().resolve(),
            record=record,
            source_checks=source_checks,
            plan=recover_plan_document(plan, backend),
        )
    command = materialize_recover_run(plan, backend)
    code = backend.execute(command)
    if code != 0:
        return code
    return _print_replay_result(
        source_run=source_run,
        output_dir=request.output_dir.expanduser().resolve(),
        record=record,
        source_checks=source_checks,
    )


def _replay_sbc_with_backend[ActionT, CommandT](
    backend: SbcBackend[ActionT, CommandT],
    request: SbcRequest,
    *,
    source_run: Path,
    record: RecordedRunMetadata,
    source_checks: tuple[ReplaySourceCheck, ...],
    check_only: bool,
) -> int:
    plan = plan_sbc_run(request, backend)
    if check_only:
        return _print_replay_plan(
            source_run=source_run,
            output_dir=request.output_dir.expanduser().resolve(),
            record=record,
            source_checks=source_checks,
            plan=sbc_plan_document(plan, backend),
        )
    command = materialize_sbc_run(plan, backend)
    code = backend.execute(command)
    if code != 0:
        return code
    return _print_replay_result(
        source_run=source_run,
        output_dir=request.output_dir.expanduser().resolve(),
        record=record,
        source_checks=source_checks,
    )


def _print_replay_plan(
    *,
    source_run: Path,
    output_dir: Path,
    record: RecordedRunMetadata,
    source_checks: tuple[ReplaySourceCheck, ...],
    plan: Mapping[str, object],
) -> int:
    print(
        json.dumps(
            replay_plan_document(
                source_run=source_run,
                output_dir=output_dir,
                record=record,
                source_checks=source_checks,
                plan=plan,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _print_replay_result(
    *,
    source_run: Path,
    output_dir: Path,
    record: RecordedRunMetadata,
    source_checks: tuple[ReplaySourceCheck, ...],
) -> int:
    comparison = compare_replay_artifacts(
        record,
        original_run_dir=source_run,
        replay_run_dir=output_dir,
    )
    print(
        json.dumps(
            replay_result_document(
                source_run=source_run,
                output_dir=output_dir,
                record=record,
                source_checks=source_checks,
                comparison=comparison,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    if comparison.byte_identical:
        return 0
    return 1


def _run_directory_with_backend[ActionT, CommandT](
    plan: RunDirectoryCommandPlan[ActionT],
    backend: ActionBackend[ActionT, CommandT],
    *,
    intent: CliIntent,
) -> int:
    match intent:
        case ShowPlan():
            print(json.dumps(run_command_plan_document(plan, backend), indent=2, sort_keys=True))
            return 0
        case ExecutePlan():
            command = materialize_run_directory_command(plan, backend)
            return backend.execute(command)


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


def _engine(namespace: argparse.Namespace) -> int:
    try:
        _reject_forwarded_engine_args(tuple(cast(list[str], namespace.engine_args)))
        subcommand = cast(str, namespace.engine_command)
        if subcommand == "ensure":
            return _engine_ensure(namespace)
        if subcommand == "path":
            return _engine_path(namespace)
        return _engine_info(namespace)
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _engine_release(base_url: str | None) -> EngineRelease:
    if base_url is None:
        return PINNED_ENGINE_RELEASE
    return replace(PINNED_ENGINE_RELEASE, base_url=base_url)


def _engine_ensure(namespace: argparse.Namespace) -> int:
    release = _engine_release(cast(str | None, namespace.base_url))
    cache_root = cast(Path | None, namespace.cache_root)
    provisioned = ensure_engine(release, cache_root=cache_root, force=bool(namespace.force))
    print(provisioned.executable)
    return 0


def _resolve_cached_or_path_engine(cache_root: Path | None) -> Path:
    found = shutil.which(str(BAYESITE))
    if found is not None:
        return Path(found).resolve(strict=False)
    candidate = cached_engine_path(cache_root=cache_root)
    if candidate.is_file():
        return candidate
    raise WorkflowError(
        "Bayesite engine is not installed: not found on PATH and not cached under "
        f"{candidate.parent}.\n\n"
        "Run `bayescycle engine ensure` to provision it, or pass --engine explicitly."
    )


def _engine_path(namespace: argparse.Namespace) -> int:
    cache_root = cast(Path | None, namespace.cache_root)
    print(_resolve_cached_or_path_engine(cache_root))
    return 0


def _engine_info(namespace: argparse.Namespace) -> int:
    engine = cast(str | None, namespace.engine)
    cache_root = cast(Path | None, namespace.cache_root)
    resolved_engine = (
        engine if engine is not None else str(_resolve_cached_or_path_engine(cache_root))
    )
    info = preflight_bayesite_engine(resolved_engine, ())
    if info.capabilities is None:
        raise WorkflowError(
            f"Bayesite engine does not support structured capabilities: {info.executable}\n\n"
            "The binary may be stale. Rebuild or reprovision a newer Bayesite release, for "
            "example: bayescycle engine ensure --force"
        )
    capabilities = info.capabilities
    document: dict[str, object] = {
        "capabilities_format": capabilities.capabilities_format,
        "commands": list(capabilities.commands),
        "version": capabilities.version,
        "ir": dict(capabilities.ir),
        "schemas": dict(capabilities.schemas),
    }
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


def _resolve_viz_source(namespace: argparse.Namespace) -> str:
    override = cast(str | None, getattr(namespace, "viz_source", None))
    return override if override is not None else BAYESITE_VIZ_SOURCE


def _parse_coords(values: list[str] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return ()
    parsed: list[tuple[str, str]] = []
    for item in values:
        if "=" not in item:
            raise WorkflowError(f"--coords expects key=value, got: {item!r}")
        key, value = item.split("=", 1)
        parsed.append((key, value))
    return tuple(parsed)


def _idata_needs_engine(validate: str | None, explicit_engine: str | None) -> bool:
    """Decide whether an idata step needs a resolved Bayesite engine path.

    ``--validate skip`` exists precisely so a run can be exported without a
    Bayesite binary: ``bayesite-idata`` never invokes ``bayesite diagnose`` in
    that mode, so resolving (and potentially auto-provisioning) an engine
    would be pointless work with real network/filesystem side effects. An
    explicit ``--engine`` is always honored and forwarded, even under skip,
    since the caller named it on purpose.
    """
    return validate != "skip" or explicit_engine is not None


def _resolve_idata_engine(namespace: argparse.Namespace, *, validate: str | None) -> str | None:
    explicit_engine = cast(str | None, namespace.engine)
    if not _idata_needs_engine(validate, explicit_engine):
        return None
    return resolve_bayesite_engine_path(explicit_engine, _auto_provision_enabled(namespace))


def _idata(namespace: argparse.Namespace) -> int:
    try:
        run_dir = cast(Path, namespace.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise WorkflowError(f"run directory does not exist: {run_dir}")
        output = cast(Path | None, namespace.output)
        resolved_output = output if output is not None else default_fit_path(run_dir)
        validate = cast(str | None, namespace.validate)
        options = IdataOptions(
            run_dir=run_dir,
            output=resolved_output,
            validate=validate,
            bayesite=_resolve_idata_engine(namespace, validate=validate),
        )
        return run_idata(options, source=_resolve_viz_source(namespace))
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _plot(namespace: argparse.Namespace) -> int:
    try:
        verb = cast(str, namespace.verb)
        run_dir = cast(Path, namespace.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise WorkflowError(f"run directory does not exist: {run_dir}")
        fit_path = cast(Path | None, namespace.fit_path)
        resolved_fit_path = fit_path if fit_path is not None else default_fit_path(run_dir)
        no_auto_idata = bool(cast(bool, namespace.no_auto_idata))
        source = _resolve_viz_source(namespace)

        if not resolved_fit_path.is_file():
            if no_auto_idata:
                raise WorkflowError(
                    f"fit file not found: {resolved_fit_path}\n\n"
                    f"Run `bayescycle idata {run_dir}` first, or pass --fit to point at an "
                    "existing fit.nc."
                )
            # `plot` exposes no --validate of its own; its auto-idata step
            # always runs bayesite-idata's own default (not "skip"), so this
            # always resolves an engine -- shared decision function kept for
            # consistency with `_idata` should that default ever change.
            engine = _resolve_idata_engine(namespace, validate=None)
            idata_code = run_idata(
                IdataOptions(run_dir=run_dir, output=resolved_fit_path, bayesite=engine),
                source=source,
            )
            if idata_code != 0:
                return idata_code

        options = PlotOptions(
            verb=verb,
            fit_path=resolved_fit_path,
            output=cast(Path | None, namespace.output),
            kind=cast(str | None, namespace.kind),
            fmt=cast(str | None, namespace.output_format),
            variables=tuple(cast("list[str] | None", namespace.variables) or ()),
            coords=_parse_coords(cast("list[str] | None", namespace.coords)),
            backend=cast(str | None, namespace.viz_backend),
            svg=bool(cast(bool, namespace.svg)),
        )
        return run_plot(options, source=source)
    except (WorkflowError, OSError) as exc:
        print(f"bayescycle: {exc}", file=sys.stderr)
        return 2


def _intent_from_namespace(namespace: argparse.Namespace) -> CliIntent:
    if bool(getattr(namespace, "show_plan", False)) or bool(getattr(namespace, "dry_run", False)):
        return ShowPlan()
    return ExecutePlan()


def _intent_skips_execution(intent: CliIntent) -> bool:
    return isinstance(intent, ShowPlan)


def _reject_forwarded_engine_args(engine_args: tuple[str, ...]) -> None:
    if engine_args:
        raise WorkflowError(
            "engine passthrough after -- is only supported for model-level backend commands"
        )


def _split_engine_args(argv: Sequence[str] | None) -> tuple[list[str], list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    if "--" not in values:
        return values, []
    separator = values.index("--")
    return values[:separator], values[separator + 1 :]
