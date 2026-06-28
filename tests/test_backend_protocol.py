from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def test_command_values_do_not_own_dry_run_projection() -> None:
    import bayescycle._commands as commands
    from bayescycle._commands import (
        BayesiteCommand,
        BayesiteSimulateCommand,
        Jaxstanv5PriorPredictiveCommand,
        Jaxstanv5SampleCommand,
    )

    assert not hasattr(commands, "DryRunCommand")
    assert not hasattr(BayesiteCommand(argv=("bayesite", "sample")), "dry_run_fields")
    assert not hasattr(
        BayesiteSimulateCommand(
            engine_command=BayesiteCommand(argv=("bayesite", "simulate")),
            native_data_path=Path("backend.json"),
            canonical_data_path=Path("canonical.json"),
        ),
        "dry_run_fields",
    )
    assert not hasattr(Jaxstanv5SampleCommand, "dry_run_fields")
    assert not hasattr(Jaxstanv5PriorPredictiveCommand, "dry_run_fields")


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
