from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from bayescycle._artifacts import BackendPrivateArtifact, CanonicalDataArtifact, IrArtifact
from bayescycle._settings import SamplerSettings
from bayescycle._workflow import (
    PreparedModelRunContext,
    SampleRequest,
    materialize_sample_run,
    plan_sample_run,
    sample_plan_document,
)


@dataclass(frozen=True)
class FakeSampleAction:
    output_dir: Path
    draws_path: Path


@dataclass(frozen=True)
class FakeCommand:
    output_dir: Path


class FakeSampleBackend:
    def plan_sample_action(
        self, context: PreparedModelRunContext, request: SampleRequest
    ) -> FakeSampleAction:
        assert request.backend == "bayesite"
        return FakeSampleAction(
            output_dir=context.output_dir,
            draws_path=context.output_dir / "posterior.ndjson",
        )

    def describe(self, action: FakeSampleAction) -> dict[str, object]:
        return {
            "backend": "fake",
            "engine_command": ["fake", "sample", "--out", str(action.draws_path)],
        }

    def materialize(self, action: FakeSampleAction) -> FakeCommand:
        assert (action.output_dir / "model.ir.json").is_file()
        assert (action.output_dir / "data.json").is_file()
        return FakeCommand(output_dir=action.output_dir)

    def execute(self, command: FakeCommand) -> int:
        assert command.output_dir.is_dir()
        return 17


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
        engine="fake",
        sampler=SamplerSettings(
            seed=None,
            chains=None,
            warmup=None,
            draws=None,
            max_tree_depth=None,
            target_accept=None,
        ),
        engine_args=(),
        force=False,
    )

    plan = plan_sample_run(request, backend)

    assert plan.ir_path == IrArtifact(output_dir.resolve() / "model.ir.json")
    assert plan.data_path == CanonicalDataArtifact(output_dir.resolve() / "data.json")
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


def test_command_values_do_not_own_dry_run_projection_or_bayesite_adapters() -> None:
    import bayescycle._commands as commands
    from bayescycle._commands import (
        BayesiteCommand,
        Jaxstanv5PriorPredictiveCommand,
        Jaxstanv5SampleCommand,
    )

    assert not hasattr(commands, "DryRunCommand")
    assert not hasattr(commands, "BayesiteSimulateCommand")
    assert not hasattr(BayesiteCommand(argv=("bayesite", "sample")), "dry_run_fields")
    assert not hasattr(Jaxstanv5SampleCommand, "dry_run_fields")
    assert not hasattr(Jaxstanv5PriorPredictiveCommand, "dry_run_fields")


def test_bayesite_action_owns_data_materialization_and_postprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bayescycle._commands import BayesiteCommand
    from bayescycle.backends import bayesite

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
