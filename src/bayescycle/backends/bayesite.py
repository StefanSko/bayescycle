"""Bayesite CLI backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bayescycle._commands import BayesiteCommand
from bayescycle._engine import run_bayesite_command
from bayescycle._errors import WorkflowError
from bayescycle._run_artifacts.canonical_data import (
    DataDocError,
    read_data_doc,
    write_bayesite_data_doc,
    write_data_doc,
)
from bayescycle._run_artifacts.references import BackendPrivateArtifact, CanonicalDataArtifact
from bayescycle._workflow import (
    DiagnoseRequest,
    DiagnoseRunContext,
    EngineBackendPlanDescription,
    PlannedModelRunContext,
    PlannedModelScenarioContext,
    PosteriorCheckRequest,
    PosteriorCheckRunContext,
    PosteriorPredictiveRequest,
    PosteriorPredictiveRunContext,
    PriorPredictiveRequest,
    RecoverCheckRequest,
    RecoverCheckRunContext,
    RecoverRequest,
    SampleRequest,
    SbcRequest,
    SimulateRequest,
)


@dataclass(frozen=True)
class MaterializeBayesiteData:
    """Materialize canonical Bayescycle data into a Bayesite-native private file."""

    canonical_path: CanonicalDataArtifact
    native_path: BackendPrivateArtifact


@dataclass(frozen=True)
class CanonicalizeGeneratedData:
    """Canonicalize Bayesite-native generated data after engine execution."""

    native_path: BackendPrivateArtifact
    canonical_path: CanonicalDataArtifact


@dataclass(frozen=True)
class BayesiteAction:
    """Planned Bayesite action before backend-private files exist."""

    engine_command: BayesiteCommand
    input_materializations: tuple[MaterializeBayesiteData, ...] = ()
    postprocess: tuple[CanonicalizeGeneratedData, ...] = ()


@dataclass(frozen=True)
class BayesitePreparedCommand:
    """Materialized Bayesite command plus backend-owned postprocessing steps."""

    engine_command: BayesiteCommand
    postprocess: tuple[CanonicalizeGeneratedData, ...] = ()


@dataclass(frozen=True)
class BayesiteBackend:
    """Bayesite subprocess backend capabilities."""

    engine: str

    def plan_sample_action(
        self, context: PlannedModelRunContext, request: SampleRequest
    ) -> BayesiteAction:
        """Plan the Bayesite sample action for a planned run directory."""
        draws_path = context.output_dir / "posterior.ndjson"
        engine_data_path = BackendPrivateArtifact(_backend_dir(context.output_dir) / "data.json")
        return BayesiteAction(
            engine_command=BayesiteCommand(
                argv=(
                    self.engine,
                    "sample",
                    "--model",
                    str(context.ir_path.path),
                    "--data",
                    str(engine_data_path.path),
                    *request.sampler.to_engine_args(),
                    "--out",
                    str(draws_path),
                    *request.engine_args,
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
        return BayesiteAction(
            engine_command=BayesiteCommand(
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
                    *request.engine_args,
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
        return BayesiteAction(
            engine_command=BayesiteCommand(
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
                    *request.engine_args,
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
        return BayesiteAction(
            engine_command=BayesiteCommand(
                argv=(
                    self.engine,
                    "recover",
                    "--model",
                    str(context.ir_path.path),
                    "--scenario",
                    str(context.scenario_path),
                    "--out",
                    str(output_path),
                    *request.engine_args,
                ),
                output_paths=(output_path,),
            )
        )

    def plan_sbc_action(
        self, context: PlannedModelScenarioContext, request: SbcRequest
    ) -> BayesiteAction:
        """Plan the Bayesite SBC action for a planned run directory."""
        output_path = context.output_dir / "sbc.json"
        return BayesiteAction(
            engine_command=BayesiteCommand(
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
                    *request.engine_args,
                ),
                output_paths=(output_path,),
            )
        )

    def plan_diagnose_action(
        self, context: DiagnoseRunContext, request: DiagnoseRequest
    ) -> BayesiteAction:
        """Plan the Bayesite diagnose action for an existing run directory."""
        return BayesiteAction(
            engine_command=BayesiteCommand(
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
            engine_command=BayesiteCommand(
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
        return BayesiteAction(
            engine_command=BayesiteCommand(
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
                    *request.engine_args,
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
        return BayesiteAction(
            engine_command=BayesiteCommand(
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
                    *request.engine_args,
                ),
                output_paths=(context.output_path,),
            )
        )

    def describe(self, action: BayesiteAction) -> EngineBackendPlanDescription:
        """Return Bayesite-owned plan description for a planned action."""
        backend_simulated_data = (
            action.postprocess[0].native_path.path if action.postprocess else None
        )
        return EngineBackendPlanDescription(
            engine_command=action.engine_command.argv,
            backend_simulated_data=backend_simulated_data,
        )

    def materialize(self, action: BayesiteAction) -> BayesitePreparedCommand:
        """Materialize Bayesite-private inputs and return an executable command."""
        for materialization in action.input_materializations:
            _materialize_engine_data(materialization)
        return BayesitePreparedCommand(
            engine_command=action.engine_command,
            postprocess=action.postprocess,
        )

    def execute(self, command: BayesitePreparedCommand) -> int:
        """Execute a materialized Bayesite command and backend-owned postprocessing."""
        code = run_bayesite_command(command.engine_command)
        if code != 0:
            return code
        for step in command.postprocess:
            _canonicalize_generated_data(step)
        return 0


def _materialize_engine_data(step: MaterializeBayesiteData) -> None:
    try:
        doc = read_data_doc(step.canonical_path.path)
        write_bayesite_data_doc(step.native_path.path, doc)
    except DataDocError as exc:
        raise WorkflowError(f"invalid data artifact for Bayesite backend: {exc}") from exc


def _canonicalize_generated_data(step: CanonicalizeGeneratedData) -> None:
    try:
        generated = read_data_doc(step.native_path.path)
        write_data_doc(step.canonical_path.path, generated)
    except DataDocError as exc:
        raise WorkflowError(
            f"Bayesite simulate did not produce a valid data artifact: {exc}"
        ) from exc


def _backend_dir(output_dir: Path) -> Path:
    return output_dir / ".bayesite"


def _optional_arg(flag: str, value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (flag, value)
