from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from bayescycle._errors import WorkflowError
from bayescycle._run_artifacts.references import (
    BackendPrivateArtifact,
    CanonicalDataArtifact,
    IrArtifact,
)
from bayescycle._settings import SamplerSettings
from bayescycle._workflow import (
    DiagnoseRequest,
    DiagnoseRunContext,
    EngineBackendPlanDescription,
    InProcessSamplePlanDescription,
    InProcessSettingsPlanDescription,
    PlannedModelRunContext,
    PlannedModelScenarioContext,
    RecoverRequest,
    SampleRequest,
    SimulateRequest,
    backend_plan_description_fields,
    materialize_recover_run,
    materialize_run_directory_command,
    materialize_sample_run,
    materialize_simulate_run,
    plan_diagnose_run,
    plan_recover_run,
    plan_sample_run,
    plan_simulate_run,
    sample_plan_document,
)


@dataclass(frozen=True)
class FakeSampleAction:
    output_dir: Path
    draws_path: Path


@dataclass(frozen=True)
class FakeCommand:
    output_dir: Path


def _sha256_uri(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


@dataclass(frozen=True)
class FakeDiagnoseAction:
    output_path: Path


@dataclass(frozen=True)
class FakeSimulateAction:
    output_dir: Path


@dataclass(frozen=True)
class FakeRecoverAction:
    output_dir: Path


class FakeDiagnoseBackend:
    def plan_diagnose_action(
        self, context: DiagnoseRunContext, request: DiagnoseRequest
    ) -> FakeDiagnoseAction:
        return FakeDiagnoseAction(output_path=context.output_path)

    def describe(self, action: FakeDiagnoseAction) -> EngineBackendPlanDescription:
        return EngineBackendPlanDescription(
            engine_command=("fake", "diagnose", "--out", str(action.output_path))
        )

    def materialize(self, action: FakeDiagnoseAction) -> FakeCommand:
        return FakeCommand(output_dir=action.output_path.parent)

    def execute(self, command: FakeCommand) -> int:
        return 0


class FakeSimulateBackend:
    def plan_simulate_action(
        self, context: PlannedModelRunContext, request: SimulateRequest, truth_path: Path
    ) -> FakeSimulateAction:
        return FakeSimulateAction(output_dir=context.output_dir)

    def describe(self, action: FakeSimulateAction) -> EngineBackendPlanDescription:
        return EngineBackendPlanDescription(engine_command=("fake", "simulate"))

    def materialize(self, action: FakeSimulateAction) -> FakeCommand:
        return FakeCommand(output_dir=action.output_dir)

    def execute(self, command: FakeCommand) -> int:
        return 0


class FakeRecoverBackend:
    def plan_recover_action(
        self, context: PlannedModelScenarioContext, request: RecoverRequest
    ) -> FakeRecoverAction:
        return FakeRecoverAction(output_dir=context.output_dir)

    def describe(self, action: FakeRecoverAction) -> EngineBackendPlanDescription:
        return EngineBackendPlanDescription(engine_command=("fake", "recover"))

    def materialize(self, action: FakeRecoverAction) -> FakeCommand:
        return FakeCommand(output_dir=action.output_dir)

    def execute(self, command: FakeCommand) -> int:
        return 0


class FakeSampleBackend:
    def plan_sample_action(
        self, context: PlannedModelRunContext, request: SampleRequest
    ) -> FakeSampleAction:
        assert request.backend == "bayesite"
        return FakeSampleAction(
            output_dir=context.output_dir,
            draws_path=context.output_dir / "posterior.ndjson",
        )

    def describe(self, action: FakeSampleAction) -> EngineBackendPlanDescription:
        return EngineBackendPlanDescription(
            engine_command=("fake", "sample", "--out", str(action.draws_path))
        )

    def materialize(self, action: FakeSampleAction) -> FakeCommand:
        assert (action.output_dir / "model.ir.json").is_file()
        assert (action.output_dir / "data.json").is_file()
        return FakeCommand(output_dir=action.output_dir)

    def execute(self, command: FakeCommand) -> int:
        assert command.output_dir.is_dir()
        return 17


def test_backend_plan_description_is_closed_adt_lowered_by_workflow(
    tmp_path: Path,
) -> None:
    simulated_data = tmp_path / ".bayesite" / "simulated_data.json"

    assert backend_plan_description_fields(
        EngineBackendPlanDescription(
            engine_command=("bayesite", "simulate"),
            backend_simulated_data=simulated_data,
        )
    ) == {
        "engine_command": ["bayesite", "simulate"],
        "backend_simulated_data": str(simulated_data),
    }
    assert backend_plan_description_fields(
        InProcessSamplePlanDescription(
            backend="jaxstanv5",
            sampler={"seed": 1, "target_accept": 0.9},
        )
    ) == {"backend": "jaxstanv5", "sampler": {"seed": 1, "target_accept": 0.9}}
    assert backend_plan_description_fields(
        InProcessSettingsPlanDescription(
            backend="jaxstanv5",
            settings={"seed": 2, "draws": 4},
        )
    ) == {"backend": "jaxstanv5", "settings": {"seed": 2, "draws": 4}}


def test_sample_backend_protocol_is_plan_materialize_execute(tmp_path: Path) -> None:
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "from jaxstanv5 import Observed, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "@model\n"
        "class Simple:\n"
        "    y = Observed(Normal(0.0, 1.0))\n",
        encoding="utf-8",
    )
    data_file = tmp_path / "input.json"
    data_file.write_text('{"y": 0.25}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    backend = FakeSampleBackend()

    request = SampleRequest(
        model_path=model_file,
        data_path=data_file,
        output_dir=output_dir,
        model_name=None,
        backend="bayesite",
        sampler=SamplerSettings(
            seed=None,
            chains=None,
            warmup=None,
            draws=None,
            max_tree_depth=None,
            target_accept=None,
        ),
        engine_args=(),
    )

    plan = plan_sample_run(request, backend)

    assert plan.context.ir_path == IrArtifact(output_dir.resolve() / "model.ir.json")
    assert plan.context.data_path == CanonicalDataArtifact(output_dir.resolve() / "data.json")
    assert plan.action == FakeSampleAction(
        output_dir=output_dir.resolve(),
        draws_path=output_dir.resolve() / "posterior.ndjson",
    )
    assert not output_dir.exists()
    assert sample_plan_document(plan, backend)["engine_command"] == [
        "fake",
        "sample",
        "--out",
        str(output_dir.resolve() / "posterior.ndjson"),
    ]

    command = materialize_sample_run(plan, backend)

    assert command == FakeCommand(output_dir=output_dir.resolve())
    assert backend.execute(command) == 17


def test_run_directory_materialization_rechecks_output_absence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "posterior.ndjson").write_text("{}\n", encoding="utf-8")
    backend = FakeDiagnoseBackend()

    plan = plan_diagnose_run(DiagnoseRequest(run_dir=run_dir), backend)
    plan.output_path.write_text("created after planning\n", encoding="utf-8")

    with pytest.raises(WorkflowError, match="output artifact already exists"):
        materialize_run_directory_command(plan, backend)


def test_run_directory_output_guard_treats_dangling_symlink_as_existing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "posterior.ndjson").write_text("{}\n", encoding="utf-8")
    (run_dir / "diagnostics.json").symlink_to(run_dir / "missing-diagnostics.json")

    with pytest.raises(WorkflowError, match="output artifact already exists"):
        plan_diagnose_run(DiagnoseRequest(run_dir=run_dir), FakeDiagnoseBackend())


def test_materialized_run_metadata_uses_hashes_captured_at_planning(tmp_path: Path) -> None:
    model_file = tmp_path / "model.py"
    original_model = (
        "from jaxstanv5 import Observed, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "@model\n"
        "class Simple:\n"
        "    y = Observed(Normal(0.0, 1.0))\n"
    )
    model_file.write_text(original_model, encoding="utf-8")
    data_file = tmp_path / "input.json"
    data_file.write_text('{"y": 0.25}\n', encoding="utf-8")
    original_model_hash = _sha256_uri(model_file)
    original_data_hash = _sha256_uri(data_file)
    backend = FakeSampleBackend()

    plan = plan_sample_run(
        SampleRequest(
            model_path=model_file,
            data_path=data_file,
            output_dir=tmp_path / "run",
            model_name=None,
            backend="bayesite",
            sampler=SamplerSettings(
                seed=None,
                chains=None,
                warmup=None,
                draws=None,
                max_tree_depth=None,
                target_accept=None,
            ),
            engine_args=(),
        ),
        backend,
    )
    model_file.write_text(f"{original_model}\n# edited after planning\n", encoding="utf-8")
    data_file.write_text('{"y": 9.0}\n', encoding="utf-8")

    materialize_sample_run(plan, backend)

    run_metadata = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert run_metadata["model"]["sha256"] == original_model_hash
    assert run_metadata["inputs"][0]["sha256"] == original_data_hash


def test_simulate_metadata_hashes_materialized_truth(tmp_path: Path) -> None:
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "from jaxstanv5 import Observed, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "@model\n"
        "class Simple:\n"
        "    y = Observed(Normal(0.0, 1.0))\n",
        encoding="utf-8",
    )
    data_file = tmp_path / "input.json"
    data_file.write_text('{"y": 0.25}\n', encoding="utf-8")
    truth_file = tmp_path / "truth.json"
    truth_file.write_text('{"mu": 0.0}\n', encoding="utf-8")
    backend = FakeSimulateBackend()

    plan = plan_simulate_run(
        SimulateRequest(
            model_path=model_file,
            data_path=data_file,
            truth_path=truth_file,
            output_dir=tmp_path / "run",
            model_name=None,
            backend="bayesite",
            seed=None,
            engine_args=(),
        ),
        backend,
    )
    truth_file.write_text('{"mu": 9.0}\n', encoding="utf-8")

    materialize_simulate_run(plan, backend)

    run_metadata = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    truth_input = next(entry for entry in run_metadata["inputs"] if entry["role"] == "truth")
    assert truth_input["sha256"] == _sha256_uri(tmp_path / "run" / "truth.json")


def test_recover_metadata_hashes_materialized_scenario(tmp_path: Path) -> None:
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "from jaxstanv5 import Observed, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "@model\n"
        "class Simple:\n"
        "    y = Observed(Normal(0.0, 1.0))\n",
        encoding="utf-8",
    )
    scenario_file = tmp_path / "scenario.json"
    scenario_file.write_text('{"data": {"y": 0.25}}\n', encoding="utf-8")
    backend = FakeRecoverBackend()

    plan = plan_recover_run(
        RecoverRequest(
            model_path=model_file,
            scenario_path=scenario_file,
            output_dir=tmp_path / "run",
            model_name=None,
            backend="bayesite",
            engine_args=(),
        ),
        backend,
    )
    scenario_file.write_text('{"data": {"y": 9.0}}\n', encoding="utf-8")

    materialize_recover_run(plan, backend)

    run_metadata = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    scenario_input = next(entry for entry in run_metadata["inputs"] if entry["role"] == "scenario")
    assert scenario_input["sha256"] == _sha256_uri(tmp_path / "run" / "scenario.json")


def test_materialization_preserves_output_dir_guard(tmp_path: Path) -> None:
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "from jaxstanv5 import Observed, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "@model\n"
        "class Simple:\n"
        "    y = Observed(Normal(0.0, 1.0))\n",
        encoding="utf-8",
    )
    data_file = tmp_path / "input.json"
    data_file.write_text('{"y": 0.25}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    backend = FakeSampleBackend()

    plan = plan_sample_run(
        SampleRequest(
            model_path=model_file,
            data_path=data_file,
            output_dir=output_dir,
            model_name=None,
            backend="bayesite",
            sampler=SamplerSettings(
                seed=None,
                chains=None,
                warmup=None,
                draws=None,
                max_tree_depth=None,
                target_accept=None,
            ),
            engine_args=(),
        ),
        backend,
    )
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("created after planning\n", encoding="utf-8")

    with pytest.raises(WorkflowError, match="output directory is not empty"):
        materialize_sample_run(plan, backend)


def test_command_values_do_not_own_dry_run_projection_or_bayesite_adapters() -> None:
    import bayescycle._commands as commands
    import bayescycle._engine as engine
    from bayescycle._commands import (
        BayesiteCommand,
        Jaxstanv5PriorPredictiveCommand,
        Jaxstanv5SampleCommand,
    )

    assert not hasattr(commands, "DryRunCommand")
    assert not hasattr(commands, "BayesiteSimulateCommand")
    assert not hasattr(engine, "EngineCommand")
    assert not hasattr(engine, "run_engine")
    assert not hasattr(BayesiteCommand(argv=("bayesite", "sample")), "dry_run_fields")
    assert not hasattr(Jaxstanv5SampleCommand, "dry_run_fields")
    assert not hasattr(Jaxstanv5PriorPredictiveCommand, "dry_run_fields")


def test_bayesite_action_owns_data_materialization_and_postprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bayescycle._commands import BayesiteCommand
    from bayescycle.backends import bayesite

    assert not hasattr(bayesite, "BayesiteExecutableCommand")
    assert not hasattr(bayesite, "materialize_run_data_for_bayesite")

    canonical_input = tmp_path / "run" / "data.json"
    canonical_input.parent.mkdir()
    canonical_input.write_text(
        '{"format":"bayescycle.data.json.v1",'
        '"variables":{"ok":{"dtype":"bool","shape":[],"values":[true]}}}\n',
        encoding="utf-8",
    )
    native_input = BackendPrivateArtifact(tmp_path / "run" / ".bayesite" / "data.json")
    native_output = BackendPrivateArtifact(tmp_path / "run" / ".bayesite" / "simulated_data.json")
    canonical_output = CanonicalDataArtifact(tmp_path / "run" / "simulated_data.json")
    engine_command = BayesiteCommand(argv=("fake-bayesite", "simulate"))
    action = bayesite.BayesiteAction(
        engine_command=engine_command,
        input_materializations=(
            bayesite.MaterializeBayesiteData(
                canonical_path=CanonicalDataArtifact(canonical_input),
                native_path=native_input,
            ),
        ),
        postprocess=(
            bayesite.CanonicalizeGeneratedData(
                native_path=native_output,
                canonical_path=canonical_output,
            ),
        ),
    )

    prepared = bayesite.BayesiteBackend("fake-bayesite").materialize(action)

    assert prepared == bayesite.BayesitePreparedCommand(
        engine_command=engine_command,
        postprocess=action.postprocess,
    )
    assert '"ok": {' in native_input.path.read_text(encoding="utf-8")
    assert '"dtype": "int64"' in native_input.path.read_text(encoding="utf-8")

    def fake_run(command: BayesiteCommand) -> int:
        assert command == engine_command
        native_output.path.write_text(
            '{"y":{"dtype":"float64","shape":[],"values":[1.5]}}\n',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(bayesite, "run_bayesite_command", fake_run)

    assert bayesite.BayesiteBackend("fake-bayesite").execute(prepared) == 0
    assert '"format": "bayescycle.data.json.v1"' in canonical_output.path.read_text(
        encoding="utf-8"
    )


def test_run_directory_commands_execute_through_bayesite_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bayescycle._cli as cli
    from bayescycle._commands import BayesiteCommand
    from bayescycle.backends.bayesite import BayesiteAction, BayesitePreparedCommand

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "posterior.ndjson").write_text("{}\n", encoding="utf-8")
    seen: dict[str, object] = {}

    class RecordingBayesiteBackend:
        def __init__(self, engine: str) -> None:
            self.engine = engine
            seen["engine"] = engine

        def plan_diagnose_action(
            self, context: DiagnoseRunContext, request: DiagnoseRequest
        ) -> BayesiteAction:
            action = BayesiteAction(
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
            seen["action"] = action
            return action

        def describe(self, action: BayesiteAction) -> EngineBackendPlanDescription:
            return EngineBackendPlanDescription(engine_command=action.engine_command.argv)

        def materialize(self, action: BayesiteAction) -> BayesitePreparedCommand:
            return BayesitePreparedCommand(action.engine_command)

        def execute(self, command: BayesitePreparedCommand) -> int:
            seen["command"] = command
            return 23

    monkeypatch.setattr(cli, "BayesiteBackend", RecordingBayesiteBackend)
    monkeypatch.setattr(
        cli,
        "_preflight_bayesite_for_intent",
        lambda engine, _intent, _command, _requirement: engine,
    )

    code = cli.main(["diagnose", str(run_dir), "--engine", "fake-bayesite"])

    assert code == 23
    expected_action = BayesiteAction(
        engine_command=BayesiteCommand(
            argv=(
                "fake-bayesite",
                "diagnose",
                "--fit",
                str(run_dir / "posterior.ndjson"),
                "--out",
                str(run_dir / "diagnostics.json"),
            ),
            output_paths=(run_dir / "diagnostics.json",),
        ),
    )
    assert seen == {
        "engine": "fake-bayesite",
        "action": expected_action,
        "command": BayesitePreparedCommand(engine_command=expected_action.engine_command),
    }


def test_first_party_backends_do_not_expose_operation_specific_runners() -> None:
    from bayescycle.backends.bayesite import BayesiteBackend
    from bayescycle.backends.jaxstanv5 import Jaxstanv5Backend

    for backend in (BayesiteBackend("bayesite"), Jaxstanv5Backend()):
        assert hasattr(backend, "execute")
        for name in (
            "build_sample_command",
            "build_prior_predictive_command",
            "build_simulate_command",
            "build_recover_command",
            "build_sbc_command",
            "materialize_sample_run",
            "run_prior_predictive",
            "run_simulate",
            "run_recover",
            "run_sbc",
        ):
            assert not hasattr(backend, name)
