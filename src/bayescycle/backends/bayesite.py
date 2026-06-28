"""Bayesite CLI backend adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bayescycle._artifacts import BackendPrivateArtifact, CanonicalDataArtifact
from bayescycle._commands import BayesiteCommand
from bayescycle._engine import run_bayesite_command
from bayescycle._errors import WorkflowError
from bayescycle._workflow import (
    PreparedModelRunContext,
    PreparedModelScenarioContext,
    PriorPredictiveRequest,
    RecoverRequest,
    SampleRequest,
    SbcRequest,
    SimulateRequest,
)
from bayescycle.data import DataDocError, read_data_doc, write_bayesite_data_doc, write_data_doc


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
        self, context: PreparedModelRunContext, request: SampleRequest
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
        self, context: PreparedModelRunContext, request: PriorPredictiveRequest
    ) -> BayesiteAction:
        """Plan the Bayesite prior-predictive action for a prepared run directory."""
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
        self, context: PreparedModelRunContext, request: SimulateRequest, truth_path: Path
    ) -> BayesiteAction:
        """Plan the Bayesite simulate action for a prepared run directory."""
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
        self, context: PreparedModelScenarioContext, request: RecoverRequest
    ) -> BayesiteAction:
        """Plan the Bayesite recover action for a prepared run directory."""
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
        self, context: PreparedModelScenarioContext, request: SbcRequest
    ) -> BayesiteAction:
        """Plan the Bayesite SBC action for a prepared run directory."""
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

    def describe(self, action: BayesiteAction) -> dict[str, object]:
        """Return Bayesite-owned dry-run fields for a planned action."""
        fields: dict[str, object] = {"engine_command": list(action.engine_command.argv)}
        if action.postprocess:
            fields["backend_simulated_data"] = str(action.postprocess[0].native_path.path)
        return fields

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


def materialize_run_data_for_bayesite(run_dir: Path) -> BackendPrivateArtifact:
    """Materialize ``run/data.json`` into Bayesite's backend-private data file."""
    engine_data_path = BackendPrivateArtifact(_backend_dir(run_dir) / "data.json")
    _materialize_engine_data(
        MaterializeBayesiteData(
            canonical_path=CanonicalDataArtifact(run_dir / "data.json"),
            native_path=engine_data_path,
        )
    )
    return engine_data_path


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
