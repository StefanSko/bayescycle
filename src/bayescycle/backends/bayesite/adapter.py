"""Bayesite CLI backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bayescycle._errors import WorkflowError
from bayescycle._integrations.descriptions import ExternalCommandPlanDescription
from bayescycle._integrations.external_command import ExternalCommand, run_external_command
from bayescycle._run_artifacts.references import BackendPrivateArtifact, CanonicalDataArtifact
from bayescycle._settings import SamplerSettings
from bayescycle._workflow.contexts import (
    DiagnoseRunContext,
    PlannedModelRunContext,
    PlannedModelScenarioContext,
    PosteriorCheckRunContext,
    PosteriorPredictiveRunContext,
    RecoverCheckRunContext,
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
from bayescycle.backends.bayesite.data_materialization import (
    CanonicalizeGeneratedData,
    MaterializeBayesiteData,
    canonicalize_generated_data,
    materialize_bayesite_data,
)


@dataclass(frozen=True)
class BayesiteAction:
    """Planned Bayesite action before backend-private files exist."""

    command: ExternalCommand
    input_materializations: tuple[MaterializeBayesiteData, ...] = ()
    postprocess: tuple[CanonicalizeGeneratedData, ...] = ()


@dataclass(frozen=True)
class BayesitePreparedCommand:
    """Materialized Bayesite command plus backend-owned postprocessing steps."""

    command: ExternalCommand
    postprocess: tuple[CanonicalizeGeneratedData, ...] = ()


@dataclass(frozen=True)
class BayesiteBackend:
    """Bayesite subprocess backend capabilities."""

    engine: str
    extra_args: tuple[str, ...] = ()

    @property
    def backend_id(self) -> str:
        """Return the stable backend identifier for run metadata."""
        return "bayesite"

    def plan_sample_action(
        self, context: PlannedModelRunContext, request: SampleRequest
    ) -> BayesiteAction:
        """Plan the Bayesite sample action for a planned run directory."""
        draws_path = context.output_dir / "posterior.ndjson"
        engine_data_path = BackendPrivateArtifact(_backend_dir(context.output_dir) / "data.json")
        extra_args = _validated_extra_args(self.extra_args, draws_path)
        return BayesiteAction(
            command=ExternalCommand(
                argv=(
                    self.engine,
                    "sample",
                    "--model",
                    str(context.ir_path.path),
                    "--data",
                    str(engine_data_path.path),
                    *_bayesite_sampler_args(request.sampler),
                    "--out",
                    str(draws_path),
                    *extra_args,
                ),
                output_paths=(draws_path,),
            ),
            input_materializations=(
                MaterializeBayesiteData(
                    canonical_path=context.data_path,
                    native_path=engine_data_path,
                ),
            ),
        )

    def plan_prior_predictive_action(
        self, context: PlannedModelRunContext, request: PriorPredictiveRequest
    ) -> BayesiteAction:
        """Plan the Bayesite prior-predictive action for a planned run directory."""
        output_path = context.output_dir / "prior_predictive.ndjson"
        engine_data_path = BackendPrivateArtifact(_backend_dir(context.output_dir) / "data.json")
        extra_args = _validated_extra_args(self.extra_args, output_path)
        return BayesiteAction(
            command=ExternalCommand(
                argv=(
                    self.engine,
                    "prior-predictive",
                    "--model",
                    str(context.ir_path.path),
                    "--data",
                    str(engine_data_path.path),
                    *_optional_arg("--seed", request.seed),
                    *_optional_arg("--draws", request.draws),
                    "--out",
                    str(output_path),
                    *extra_args,
                ),
                output_paths=(output_path,),
            ),
            input_materializations=(
                MaterializeBayesiteData(
                    canonical_path=context.data_path,
                    native_path=engine_data_path,
                ),
            ),
        )

    def plan_simulate_action(
        self, context: PlannedModelRunContext, request: SimulateRequest, truth_path: Path
    ) -> BayesiteAction:
        """Plan the Bayesite simulate action for a planned run directory."""
        canonical_output_path = CanonicalDataArtifact(context.output_dir / "simulated_data.json")
        native_output_path = BackendPrivateArtifact(
            _backend_dir(context.output_dir) / "simulated_data.json"
        )
        engine_data_path = BackendPrivateArtifact(_backend_dir(context.output_dir) / "data.json")
        extra_args = _validated_extra_args(self.extra_args, canonical_output_path.path)
        return BayesiteAction(
            command=ExternalCommand(
                argv=(
                    self.engine,
                    "simulate",
                    "--model",
                    str(context.ir_path.path),
                    "--data",
                    str(engine_data_path.path),
                    "--truth",
                    str(truth_path),
                    *_optional_arg("--seed", request.seed),
                    "--out",
                    str(native_output_path.path),
                    *extra_args,
                ),
                output_paths=(canonical_output_path.path, native_output_path.path),
            ),
            input_materializations=(
                MaterializeBayesiteData(
                    canonical_path=context.data_path,
                    native_path=engine_data_path,
                ),
            ),
            postprocess=(
                CanonicalizeGeneratedData(
                    native_path=native_output_path,
                    canonical_path=canonical_output_path,
                ),
            ),
        )

    def plan_recover_action(
        self, context: PlannedModelScenarioContext, request: RecoverRequest
    ) -> BayesiteAction:
        """Plan the Bayesite recover action for a planned run directory."""
        output_path = context.output_dir / "recovery.json"
        extra_args = _validated_extra_args(self.extra_args, output_path)
        return BayesiteAction(
            command=ExternalCommand(
                argv=(
                    self.engine,
                    "recover",
                    "--model",
                    str(context.ir_path.path),
                    "--scenario",
                    str(context.scenario_path),
                    "--out",
                    str(output_path),
                    *extra_args,
                ),
                output_paths=(output_path,),
            )
        )

    def plan_sbc_action(
        self, context: PlannedModelScenarioContext, request: SbcRequest
    ) -> BayesiteAction:
        """Plan the Bayesite SBC action for a planned run directory."""
        output_path = context.output_dir / "sbc.json"
        extra_args = _validated_extra_args(self.extra_args, output_path)
        return BayesiteAction(
            command=ExternalCommand(
                argv=(
                    self.engine,
                    "sbc",
                    "--model",
                    str(context.ir_path.path),
                    "--scenario",
                    str(context.scenario_path),
                    *_optional_arg("--replicates", request.replicates),
                    "--out",
                    str(output_path),
                    *extra_args,
                ),
                output_paths=(output_path,),
            )
        )

    def plan_diagnose_action(
        self, context: DiagnoseRunContext, request: DiagnoseRequest
    ) -> BayesiteAction:
        """Plan the Bayesite diagnose action for an existing run directory."""
        return BayesiteAction(
            command=ExternalCommand(
                argv=(
                    self.engine,
                    "diagnose",
                    "--fit",
                    str(context.fit_path),
                    "--out",
                    str(context.output_path),
                ),
                output_paths=(context.output_path,),
            )
        )

    def plan_posterior_predictive_action(
        self, context: PosteriorPredictiveRunContext, request: PosteriorPredictiveRequest
    ) -> BayesiteAction:
        """Plan the Bayesite posterior-predictive action for an existing run directory."""
        engine_data_path = BackendPrivateArtifact(_backend_dir(context.run_dir) / "data.json")
        return BayesiteAction(
            command=ExternalCommand(
                argv=(
                    self.engine,
                    "posterior-predictive",
                    "--model",
                    str(context.model_path.path),
                    "--data",
                    str(engine_data_path.path),
                    "--fit",
                    str(context.fit_path),
                    "--seed",
                    request.seed,
                    "--out",
                    str(context.output_path),
                ),
                output_paths=(context.output_path,),
            ),
            input_materializations=(
                MaterializeBayesiteData(
                    canonical_path=context.data_path,
                    native_path=engine_data_path,
                ),
            ),
        )

    def plan_posterior_check_action(
        self, context: PosteriorCheckRunContext, request: PosteriorCheckRequest
    ) -> BayesiteAction:
        """Plan the Bayesite posterior-check action for an existing run directory."""
        engine_data_path = BackendPrivateArtifact(_backend_dir(context.run_dir) / "data.json")
        extra_args = _validated_extra_args(self.extra_args, context.output_path)
        return BayesiteAction(
            command=ExternalCommand(
                argv=(
                    self.engine,
                    "posterior-check",
                    "--model",
                    str(context.model_path.path),
                    "--data",
                    str(engine_data_path.path),
                    "--fit",
                    str(context.fit_path),
                    *_optional_arg("--seed", request.seed),
                    "--out",
                    str(context.output_path),
                    *extra_args,
                ),
                output_paths=(context.output_path,),
            ),
            input_materializations=(
                MaterializeBayesiteData(
                    canonical_path=context.data_path,
                    native_path=engine_data_path,
                ),
            ),
        )

    def plan_recover_check_action(
        self, context: RecoverCheckRunContext, request: RecoverCheckRequest
    ) -> BayesiteAction:
        """Plan the Bayesite recover-check action for an existing run directory."""
        targets_args = (
            ("--targets", str(context.targets_path)) if context.targets_path is not None else ()
        )
        extra_args = _validated_extra_args(self.extra_args, context.output_path)
        return BayesiteAction(
            command=ExternalCommand(
                argv=(
                    self.engine,
                    "recover-check",
                    "--fit",
                    str(context.fit_path),
                    "--truth",
                    str(context.truth_path),
                    *targets_args,
                    *_optional_arg("--interval", request.interval),
                    "--out",
                    str(context.output_path),
                    *extra_args,
                ),
                output_paths=(context.output_path,),
            )
        )

    def describe(self, action: BayesiteAction) -> ExternalCommandPlanDescription:
        """Return Bayesite-owned plan description for a planned action."""
        backend_simulated_data = (
            action.postprocess[0].native_path.path if action.postprocess else None
        )
        return ExternalCommandPlanDescription(
            backend="bayesite",
            command=action.command.argv,
            backend_simulated_data=backend_simulated_data,
        )

    def materialize(self, action: BayesiteAction) -> BayesitePreparedCommand:
        """Materialize Bayesite-private inputs and return an executable command."""
        for materialization in action.input_materializations:
            materialize_bayesite_data(materialization)
        return BayesitePreparedCommand(
            command=action.command,
            postprocess=action.postprocess,
        )

    def execute(self, command: BayesitePreparedCommand) -> int:
        """Execute a materialized Bayesite command and backend-owned postprocessing."""
        code = run_external_command(command.command)
        if code != 0:
            return code
        for step in command.postprocess:
            canonicalize_generated_data(step)
        return 0


def _backend_dir(output_dir: Path) -> Path:
    return output_dir / ".bayesite"


def _bayesite_sampler_args(settings: SamplerSettings) -> tuple[str, ...]:
    """Render logical sampler settings as Bayesite CLI arguments."""
    args: list[str] = []
    if settings.seed is not None:
        args.extend(("--seed", settings.seed))
    if settings.chains is not None:
        args.extend(("--chains", settings.chains))
    if settings.warmup is not None:
        args.extend(("--warmup", settings.warmup))
    if settings.draws is not None:
        args.extend(("--draws", settings.draws))
    if settings.max_tree_depth is not None:
        args.extend(("--max-treedepth", settings.max_tree_depth))
    if settings.target_accept is not None:
        args.extend(("--target-accept", settings.target_accept))
    return tuple(args)


def _validated_extra_args(extra_args: tuple[str, ...], output_path: Path) -> tuple[str, ...]:
    for arg in extra_args:
        if arg == "--out" or arg.startswith("--out="):
            raise WorkflowError(
                "forwarded engine args may not include --out; "
                f"bayescycle writes output to {output_path}. "
                "Use Bayesite directly for custom output or streaming."
            )
    return extra_args


def _optional_arg(flag: str, value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (flag, value)
