from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import bayescycle._dims as dims_module
from bayescycle._cli import main

FAKE_BAYESITE_USAGE = (
    "usage: bayesite sample diagnose prior-predictive posterior-predictive "
    "posterior-check simulate recover-check recover sbc"
)


def _fake_bayesite_usage_prelude() -> str:
    return f"if not args:\n    print({FAKE_BAYESITE_USAGE!r})\n    raise SystemExit(0)\n"


def _write_simple_model(tmp_path: Path) -> Path:
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
    return model_file


def _write_normal_mean_model(tmp_path: Path) -> Path:
    model_file = tmp_path / "normal_mean.py"
    model_file.write_text(
        "from jaxstanv5 import Observed, Param, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "@model\n"
        "class NormalMean:\n"
        "    mu = Param(Normal(0.0, 1.0))\n"
        "    y = Observed(Normal(mu, 1.0))\n",
        encoding="utf-8",
    )
    return model_file


def _write_input_data(tmp_path: Path) -> Path:
    data_file = tmp_path / "input.json"
    data_file.write_text('{"y": 0.25}\n', encoding="utf-8")
    return data_file


def _write_empty_data(tmp_path: Path) -> Path:
    data_file = tmp_path / "empty_data.json"
    data_file.write_text("{}\n", encoding="utf-8")
    return data_file


def _write_json_file(tmp_path: Path, name: str, content: str = "{}\n") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _write_run_artifacts(
    run_dir: Path,
    *,
    model: bool = True,
    data: bool = True,
    posterior: bool = True,
) -> None:
    run_dir.mkdir()
    if model:
        (run_dir / "model.ir.json").write_text('{"jaxstanv5_ir": 1}\n', encoding="utf-8")
    if data:
        (run_dir / "data.json").write_text('{"y": 0.25}\n', encoding="utf-8")
    if posterior:
        (run_dir / "posterior.ndjson").write_text('{"draws_format": "test"}\n', encoding="utf-8")


def _write_fake_out_engine(tmp_path: Path) -> Path:
    fake_engine = tmp_path / "fake_bayesite.py"
    fake_engine.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "args = sys.argv[1:]\n" + _fake_bayesite_usage_prelude() + "try:\n"
        "    out_index = args.index('--out')\n"
        "except ValueError:\n"
        "    print('missing --out', file=sys.stderr)\n"
        "    raise SystemExit(11)\n"
        "Path(args[out_index + 1]).write_text(' '.join(args) + '\\n')\n",
        encoding="utf-8",
    )
    fake_engine.chmod(0o755)
    return fake_engine


def test_sample_dry_run_writes_ir_and_data_and_prints_engine_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--dry-run",
            "--",
            "--seed",
            "123",
        ]
    )

    assert code == 0
    ir_path = output_dir / "model.ir.json"
    run_data_path = output_dir / "data.json"
    assert json.loads(ir_path.read_text(encoding="utf-8"))["jaxstanv5_ir"] == 1
    assert json.loads(run_data_path.read_text(encoding="utf-8")) == {
        "format": "bayescycle.data.json.v1",
        "variables": {"y": {"dtype": "float64", "shape": [], "values": [0.25]}},
    }
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) == {
        "manifest_format": "bayescycle.run-manifest.v1",
        "artifacts": {
            "data.json": {
                "format": "bayescycle.data.json.v1",
                "path": "data.json",
            }
        },
    }
    assert not (output_dir / "dims.json").exists()

    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "data": str(run_data_path),
        "draws": str(output_dir / "posterior.ndjson"),
        "engine_command": [
            "bayesite",
            "sample",
            "--model",
            str(ir_path),
            "--data",
            str(output_dir / ".bayesite" / "data.json"),
            "--out",
            str(output_dir / "posterior.ndjson"),
            "--seed",
            "123",
        ],
        "ir": str(ir_path),
        "model": "Simple",
        "output": str(output_dir),
    }


def test_sample_dry_run_accepts_promoted_sampler_options_and_explicit_out(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_input_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--seed",
            "7",
            "--chains",
            "2",
            "--warmup",
            "100",
            "--draws",
            "100",
            "--dry-run",
            "--",
            "--experimental-engine-flag",
        ]
    )

    assert code == 0
    ir_path = output_dir / "model.ir.json"
    printed = json.loads(capsys.readouterr().out)
    assert printed["draws"] == str(output_dir / "posterior.ndjson")
    assert printed["engine_command"] == [
        "bayesite",
        "sample",
        "--model",
        str(ir_path),
        "--data",
        str(output_dir / ".bayesite" / "data.json"),
        "--seed",
        "7",
        "--chains",
        "2",
        "--warmup",
        "100",
        "--draws",
        "100",
        "--out",
        str(output_dir / "posterior.ndjson"),
        "--experimental-engine-flag",
    ]


def test_sample_dry_run_supports_jaxstanv5_backend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_normal_mean_model(tmp_path)
    data_file = _write_input_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--backend",
            "jaxstanv5",
            "--seed",
            "3",
            "--chains",
            "1",
            "--warmup",
            "5",
            "--draws",
            "4",
            "--max-treedepth",
            "3",
            "--target-accept",
            "0.75",
            "--dry-run",
        ]
    )

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    printed.pop("dims", None)
    assert printed == {
        "backend": "jaxstanv5",
        "data": str(output_dir / "data.json"),
        "draws": str(output_dir / "posterior.ndjson"),
        "ir": str(output_dir / "model.ir.json"),
        "model": "NormalMean",
        "output": str(output_dir),
        "sampler": {
            "chains": 1,
            "draws": 4,
            "max_tree_depth": 3,
            "seed": 3,
            "target_accept": 0.75,
            "warmup": 5,
        },
    }
    assert not (output_dir / "posterior.ndjson").exists()


def test_sample_jaxstanv5_backend_rejects_bayesite_engine_option(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_normal_mean_model(tmp_path)
    data_file = _write_input_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--backend",
            "jaxstanv5",
            "--engine",
            "/tmp/bayesite",
            "--dry-run",
        ]
    )

    assert code == 2
    assert "--engine configures the bayesite backend" in capsys.readouterr().err
    assert not output_dir.exists()


def test_sample_jaxstanv5_backend_rejects_engine_passthrough(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_normal_mean_model(tmp_path)
    data_file = _write_input_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--backend",
            "jaxstanv5",
            "--dry-run",
            "--",
            "--experimental-engine-flag",
        ]
    )

    assert code == 2
    assert "passthrough" in capsys.readouterr().err
    assert not output_dir.exists()


def test_sample_jaxstanv5_backend_writes_energy_posterior(
    tmp_path: Path,
) -> None:
    pytest.importorskip("blackjax")
    pytest.importorskip("jax")
    model_file = _write_normal_mean_model(tmp_path)
    data_file = _write_input_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--backend",
            "jaxstanv5",
            "--seed",
            "3",
            "--chains",
            "1",
            "--warmup",
            "5",
            "--draws",
            "4",
            "--max-treedepth",
            "3",
        ]
    )

    assert code == 0
    lines = [
        json.loads(line) for line in (output_dir / "posterior.ndjson").read_text().splitlines()
    ]
    assert lines[0]["sample_stats_mode"] == "per_draw_v2"
    assert lines[0]["model_data_fingerprint"].startswith("sha256:")
    assert "energy" in lines[1]
    assert lines[-1]["trailer"]["model_data_fingerprint"] == lines[0]["model_data_fingerprint"]


@pytest.mark.parametrize("engine_args", [("--out", "elsewhere.ndjson"), ("--out=-",)])
def test_sample_rejects_forwarded_out_engine_arg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    engine_args: tuple[str, ...],
) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_input_data(tmp_path)

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(tmp_path / "run"),
            "--dry-run",
            "--",
            *engine_args,
        ]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert "forwarded engine args may not include --out" in err
    assert "posterior.ndjson" in err


def test_sample_missing_engine_fails_before_output_dir_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_input_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--engine",
            str(tmp_path / "missing-bayesite"),
        ]
    )

    assert code == 2
    assert "Bayesite engine does not exist" in capsys.readouterr().err
    assert not output_dir.exists()


def test_sample_invokes_engine_with_explicit_out_instead_of_stdout_capture(
    tmp_path: Path,
) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_input_data(tmp_path)
    output_dir = tmp_path / "run"
    fake_engine = tmp_path / "fake_bayesite.py"
    fake_engine.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "args = sys.argv[1:]\n" + _fake_bayesite_usage_prelude() + "if args[:1] != ['sample']:\n"
        "    raise SystemExit(10)\n"
        "try:\n"
        "    out_index = args.index('--out')\n"
        "except ValueError:\n"
        "    print('missing --out', file=sys.stderr)\n"
        "    raise SystemExit(11)\n"
        "Path(args[out_index + 1]).write_text('posterior written by --out\\n')\n"
        "print('stdout is not the posterior artifact')\n",
        encoding="utf-8",
    )
    fake_engine.chmod(0o755)

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--engine",
            str(fake_engine),
            "--seed",
            "7",
        ]
    )

    assert code == 0
    assert (output_dir / "posterior.ndjson").read_text() == "posterior written by --out\n"


def test_sample_uses_preflight_resolved_engine_after_model_changes_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "other").mkdir()
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "import os\n"
        "from jaxstanv5 import Observed, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "os.chdir('other')\n"
        "@model\n"
        "class Simple:\n"
        "    y = Observed(Normal(0.0, 1.0))\n",
        encoding="utf-8",
    )
    data_file = _write_input_data(tmp_path)
    engine_dir = tmp_path / "bin"
    engine_dir.mkdir()
    fake_engine = engine_dir / "fake_bayesite.py"
    fake_engine.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "args = sys.argv[1:]\n"
        + _fake_bayesite_usage_prelude()
        + "out_index = args.index('--out')\n"
        "Path(args[out_index + 1]).write_text('posterior written by resolved engine\\n')\n",
        encoding="utf-8",
    )
    fake_engine.chmod(0o755)

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            "run",
            "--engine",
            str(Path("bin") / "fake_bayesite.py"),
        ]
    )

    assert code == 0
    assert (tmp_path / "run" / "posterior.ndjson").read_text(encoding="utf-8") == (
        "posterior written by resolved engine\n"
    )


def test_sample_force_clears_stale_posterior_when_engine_fails(tmp_path: Path) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_input_data(tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    stale_posterior = output_dir / "posterior.ndjson"
    stale_posterior.write_text("stale draws\n", encoding="utf-8")
    fake_engine = tmp_path / "failing_bayesite.py"
    fake_engine.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "\n"
        "args = sys.argv[1:]\n"
        + _fake_bayesite_usage_prelude()
        + "print('engine failed before writing --out', file=sys.stderr)\n"
        "raise SystemExit(19)\n",
        encoding="utf-8",
    )
    fake_engine.chmod(0o755)

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--force",
            "--engine",
            str(fake_engine),
        ]
    )

    assert code == 19
    assert not stale_posterior.exists()


def test_prior_predictive_dry_run_prepares_run_and_prints_engine_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_empty_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "prior-predictive",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--seed",
            "7",
            "--draws",
            "9",
            "--dry-run",
            "--",
            "--experimental-engine-flag",
        ]
    )

    assert code == 0
    ir_path = output_dir / "model.ir.json"
    run_data_path = output_dir / "data.json"
    assert json.loads(ir_path.read_text(encoding="utf-8"))["jaxstanv5_ir"] == 1
    assert json.loads(run_data_path.read_text(encoding="utf-8")) == {
        "format": "bayescycle.data.json.v1",
        "variables": {},
    }
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "data": str(run_data_path),
        "engine_command": [
            "bayesite",
            "prior-predictive",
            "--model",
            str(ir_path),
            "--data",
            str(output_dir / ".bayesite" / "data.json"),
            "--seed",
            "7",
            "--draws",
            "9",
            "--out",
            str(output_dir / "prior_predictive.ndjson"),
            "--experimental-engine-flag",
        ],
        "ir": str(ir_path),
        "model": "Simple",
        "output": str(output_dir),
        "prior_predictive": str(output_dir / "prior_predictive.ndjson"),
    }
    assert not (output_dir / "prior_predictive.ndjson").exists()


def test_prior_predictive_dry_run_supports_jaxstanv5_backend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_normal_mean_model(tmp_path)
    data_file = _write_empty_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "prior-predictive",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--backend",
            "jaxstanv5",
            "--seed",
            "3",
            "--draws",
            "4",
            "--dry-run",
        ]
    )

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    printed.pop("dims", None)
    assert printed == {
        "backend": "jaxstanv5",
        "data": str(output_dir / "data.json"),
        "ir": str(output_dir / "model.ir.json"),
        "model": "NormalMean",
        "output": str(output_dir),
        "prior_predictive": str(output_dir / "prior_predictive.ndjson"),
        "settings": {"draws": 4, "seed": 3},
    }
    assert not (output_dir / "prior_predictive.ndjson").exists()


def test_prior_predictive_jaxstanv5_backend_rejects_engine_passthrough_before_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_normal_mean_model(tmp_path)
    data_file = _write_empty_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "prior-predictive",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--backend",
            "jaxstanv5",
            "--dry-run",
            "--",
            "--experimental-engine-flag",
        ]
    )

    assert code == 2
    assert "passthrough" in capsys.readouterr().err
    assert not output_dir.exists()


def test_prior_predictive_jaxstanv5_backend_writes_v0_stream(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    model_file = _write_normal_mean_model(tmp_path)
    data_file = _write_empty_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "prior-predictive",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--backend",
            "jaxstanv5",
            "--seed",
            "3",
            "--draws",
            "4",
        ]
    )

    assert code == 0
    lines = [
        json.loads(line)
        for line in (output_dir / "prior_predictive.ndjson").read_text().splitlines()
    ]
    assert lines[0]["prior_predictive_format"] == "v0-provisional"
    assert lines[0]["artifact_kind"] == "prior_predictive_draws"
    assert lines[0]["draw_count"] == 4
    assert lines[0]["site_order"] == ["mu", "y"]
    assert lines[0]["sites"][0]["role"] == "parameter"
    assert lines[0]["sites"][1]["role"] == "observed"
    assert lines[1]["draw"] == 0
    assert set(lines[1]["values"]) == {"mu", "y"}
    assert lines[-1]["trailer"]["draw_count"] == 4


@pytest.mark.parametrize("engine_args", [("--out", "elsewhere.ndjson"), ("--out=-",)])
def test_prior_predictive_rejects_forwarded_out_engine_arg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    engine_args: tuple[str, ...],
) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_empty_data(tmp_path)

    code = main(
        [
            "prior-predictive",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(tmp_path / "run"),
            "--dry-run",
            "--",
            *engine_args,
        ]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert "forwarded engine args may not include --out" in err
    assert "prior_predictive.ndjson" in err


def test_simulate_jaxstanv5_backend_reports_unsupported_without_engine_preflight(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_empty_data(tmp_path)
    truth_file = _write_json_file(tmp_path, "truth.json", '{"mu": 0.1}\n')
    output_dir = tmp_path / "run"

    code = main(
        [
            "simulate",
            str(model_file),
            "--data",
            str(data_file),
            "--truth",
            str(truth_file),
            "-o",
            str(output_dir),
            "--backend",
            "jaxstanv5",
        ]
    )

    assert code == 2
    assert "not supported on --backend jaxstanv5" in capsys.readouterr().err
    assert not output_dir.exists()


def test_simulate_stale_engine_fails_before_output_dir_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_empty_data(tmp_path)
    truth_file = _write_json_file(tmp_path, "truth.json", '{"mu": 0.1}\n')
    output_dir = tmp_path / "run"
    stale_engine = tmp_path / "stale_bayesite.py"
    stale_engine.write_text(
        f"#!{sys.executable}\n"
        "print('usage: bayesite sample diagnose prior-predictive recover sbc')\n",
        encoding="utf-8",
    )
    stale_engine.chmod(0o755)

    code = main(
        [
            "simulate",
            str(model_file),
            "--data",
            str(data_file),
            "--truth",
            str(truth_file),
            "-o",
            str(output_dir),
            "--engine",
            str(stale_engine),
        ]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert "does not support required command: simulate" in err
    assert "required by stage: simulate" in err
    assert not output_dir.exists()


def test_simulate_invokes_engine_with_owned_truth_and_canonical_output(tmp_path: Path) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_empty_data(tmp_path)
    truth_file = _write_json_file(tmp_path, "truth.json", '{"mu": 0.1}\n')
    output_dir = tmp_path / "run"
    fake_engine = tmp_path / "fake_bayesite.py"
    fake_engine.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "args = sys.argv[1:]\n" + _fake_bayesite_usage_prelude() + "if args[:1] != ['simulate']:\n"
        "    raise SystemExit(10)\n"
        "out_index = args.index('--out')\n"
        "Path(args[out_index + 1]).write_text("
        '\'{"y": {"dtype": "float64", "shape": [], "values": [1.5]}}\\n\''
        ")\n",
        encoding="utf-8",
    )
    fake_engine.chmod(0o755)

    code = main(
        [
            "simulate",
            str(model_file),
            "--data",
            str(data_file),
            "--truth",
            str(truth_file),
            "-o",
            str(output_dir),
            "--seed",
            "1",
            "--engine",
            str(fake_engine),
        ]
    )

    assert code == 0
    assert (output_dir / "truth.json").read_text(encoding="utf-8") == '{"mu": 0.1}\n'
    assert json.loads((output_dir / "simulated_data.json").read_text(encoding="utf-8")) == {
        "format": "bayescycle.data.json.v1",
        "variables": {"y": {"dtype": "float64", "shape": [], "values": [1.5]}},
    }
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["artifacts"] == {
        "data.json": {"format": "bayescycle.data.json.v1", "path": "data.json"},
        "simulated_data.json": {
            "format": "bayescycle.data.json.v1",
            "path": "simulated_data.json",
        },
    }
    assert (output_dir / ".bayesite" / "simulated_data.json").is_file()


def test_simulate_missing_truth_does_not_create_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_empty_data(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "simulate",
            str(model_file),
            "--data",
            str(data_file),
            "--truth",
            str(tmp_path / "missing-truth.json"),
            "-o",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert code == 2
    assert "truth file does not exist" in capsys.readouterr().err
    assert not output_dir.exists()


def test_recover_missing_scenario_does_not_create_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "recover",
            str(model_file),
            "--scenario",
            str(tmp_path / "missing-scenario.json"),
            "-o",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert code == 2
    assert "scenario file does not exist" in capsys.readouterr().err
    assert not output_dir.exists()


def test_sample_missing_data_does_not_create_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    output_dir = tmp_path / "run"

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(tmp_path / "missing-data.json"),
            "-o",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert code == 2
    assert "data file does not exist" in capsys.readouterr().err
    assert not output_dir.exists()


def test_sample_invalid_canonical_data_does_not_create_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    data_file = tmp_path / "bad-data.json"
    data_file.write_text(
        '{"format":"bayescycle.data.json.v1",'
        '"variables":{"y":{"dtype":"float64","shape":[2],"values":[0.25]}}}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert code == 2
    err = capsys.readouterr().err
    assert "invalid data file" in err
    assert "values length" in err
    assert not output_dir.exists()


def test_recover_dry_run_prepares_scenario_command(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    scenario_file = _write_json_file(tmp_path, "scenario.json", '{"recover_scenario":"v0"}\n')
    output_dir = tmp_path / "run"

    code = main(
        [
            "recover",
            str(model_file),
            "--scenario",
            str(scenario_file),
            "-o",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "engine_command": [
            "bayesite",
            "recover",
            "--model",
            str(output_dir / "model.ir.json"),
            "--scenario",
            str(output_dir / "scenario.json"),
            "--out",
            str(output_dir / "recovery.json"),
        ],
        "ir": str(output_dir / "model.ir.json"),
        "model": "Simple",
        "output": str(output_dir),
        "recovery": str(output_dir / "recovery.json"),
        "scenario": str(output_dir / "scenario.json"),
    }


def test_sbc_dry_run_prepares_scenario_command_with_replicates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = _write_simple_model(tmp_path)
    scenario_file = _write_json_file(tmp_path, "scenario.json", '{"sbc_scenario":"v0"}\n')
    output_dir = tmp_path / "run"

    code = main(
        [
            "sbc",
            str(model_file),
            "--scenario",
            str(scenario_file),
            "-o",
            str(output_dir),
            "--replicates",
            "12",
            "--dry-run",
        ]
    )

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "engine_command": [
            "bayesite",
            "sbc",
            "--model",
            str(output_dir / "model.ir.json"),
            "--scenario",
            str(output_dir / "scenario.json"),
            "--replicates",
            "12",
            "--out",
            str(output_dir / "sbc.json"),
        ],
        "ir": str(output_dir / "model.ir.json"),
        "model": "Simple",
        "output": str(output_dir),
        "sbc": str(output_dir / "sbc.json"),
        "scenario": str(output_dir / "scenario.json"),
    }


def test_workflow_plan_prints_single_backend_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["workflow-plan", "--backend", "bayesite", "--engine", "/tmp/bayesite"])

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "mode": "single",
        "stages": {"recover": "bayesite", "simulate": "bayesite"},
        "backends": {"bayesite": {"engine": "/tmp/bayesite"}},
    }


def test_workflow_plan_rejects_partial_mixed_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["workflow-plan", "--simulate-backend", "bayesite"])

    assert code == 2
    err = capsys.readouterr().err
    assert "Partial backend assignment" in err
    assert "recover: inherited from default" in err


def test_workflow_plan_rejects_engine_without_bayesite(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["workflow-plan", "--backend", "jaxstanv5", "--engine", "/tmp/bayesite"])

    assert code == 2
    assert "no bayesite backend stage was selected" in capsys.readouterr().err


def test_workflow_plan_loads_mixed_toml_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "workflow.toml"
    config.write_text(
        '[workflow]\nmode = "mixed"\n\n'
        '[stages.simulate]\nbackend = "bayesite"\n\n'
        '[stages.recover]\nbackend = "jaxstanv5"\n',
        encoding="utf-8",
    )

    code = main(["workflow-plan", "--config", str(config)])

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "mode": "mixed",
        "stages": {"recover": "jaxstanv5", "simulate": "bayesite"},
        "backends": {},
    }


def test_workflow_plan_missing_config_reports_standard_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["workflow-plan", "--config", str(tmp_path / "missing.toml")])

    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("bayescycle: cannot read backend plan config")


def test_diagnose_dry_run_uses_run_directory_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir, model=False, data=False)

    code = main(["diagnose", str(run_dir), "--dry-run"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "engine_command": [
            "bayesite",
            "diagnose",
            "--fit",
            str(run_dir / "posterior.ndjson"),
            "--out",
            str(run_dir / "diagnostics.json"),
        ],
        "output": str(run_dir / "diagnostics.json"),
        "run": str(run_dir),
    }


def test_diagnose_invokes_engine_with_run_directory_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir, model=False, data=False)
    fake_engine = _write_fake_out_engine(tmp_path)

    code = main(["diagnose", str(run_dir), "--engine", str(fake_engine)])

    assert code == 0
    assert (run_dir / "diagnostics.json").read_text(encoding="utf-8") == (
        f"diagnose --fit {run_dir / 'posterior.ndjson'} --out {run_dir / 'diagnostics.json'}\n"
    )


def test_diagnose_reports_missing_posterior(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir, model=False, data=False, posterior=False)

    code = main(["diagnose", str(run_dir), "--dry-run"])

    assert code == 2
    err = capsys.readouterr().err
    assert "missing required run artifact" in err
    assert "posterior.ndjson" in err


def test_posterior_predictive_dry_run_uses_run_directory_artifacts_and_seed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir)

    code = main(["posterior-predictive", str(run_dir), "--seed", "8", "--dry-run"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "engine_command": [
            "bayesite",
            "posterior-predictive",
            "--model",
            str(run_dir / "model.ir.json"),
            "--data",
            str(run_dir / ".bayesite" / "data.json"),
            "--fit",
            str(run_dir / "posterior.ndjson"),
            "--seed",
            "8",
            "--out",
            str(run_dir / "posterior_predictive.ndjson"),
        ],
        "output": str(run_dir / "posterior_predictive.ndjson"),
        "run": str(run_dir),
    }


def test_posterior_predictive_invokes_engine_with_run_directory_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir)
    fake_engine = _write_fake_out_engine(tmp_path)

    code = main(["posterior-predictive", str(run_dir), "--seed", "8", "--engine", str(fake_engine)])

    assert code == 0
    assert (run_dir / "posterior_predictive.ndjson").read_text(encoding="utf-8") == (
        f"posterior-predictive --model {run_dir / 'model.ir.json'} "
        f"--data {run_dir / '.bayesite' / 'data.json'} --fit {run_dir / 'posterior.ndjson'} "
        f"--seed 8 --out {run_dir / 'posterior_predictive.ndjson'}\n"
    )


def test_posterior_predictive_reports_missing_required_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir, model=False, posterior=False)

    code = main(["posterior-predictive", str(run_dir), "--seed", "8", "--dry-run"])

    assert code == 2
    err = capsys.readouterr().err
    assert "missing required run artifacts" in err
    assert "model.ir.json" in err
    assert "posterior.ndjson" in err


def test_posterior_check_dry_run_uses_run_directory_artifacts_and_seed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir)

    code = main(["posterior-check", str(run_dir), "--seed", "8", "--dry-run"])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "engine_command": [
            "bayesite",
            "posterior-check",
            "--model",
            str(run_dir / "model.ir.json"),
            "--data",
            str(run_dir / ".bayesite" / "data.json"),
            "--fit",
            str(run_dir / "posterior.ndjson"),
            "--seed",
            "8",
            "--out",
            str(run_dir / "posterior_check.json"),
        ],
        "output": str(run_dir / "posterior_check.json"),
        "run": str(run_dir),
    }


def test_posterior_check_reports_missing_required_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir, model=False, posterior=False)

    code = main(["posterior-check", str(run_dir), "--seed", "8", "--dry-run"])

    assert code == 2
    err = capsys.readouterr().err
    assert "missing required run artifacts" in err
    assert "model.ir.json" in err
    assert "posterior.ndjson" in err


def test_posterior_check_jaxstanv5_backend_reports_unsupported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir)

    code = main(["posterior-check", str(run_dir), "--backend", "jaxstanv5", "--dry-run"])

    assert code == 2
    assert "not supported on --backend jaxstanv5" in capsys.readouterr().err


def test_recover_check_dry_run_uses_fit_truth_targets_and_interval(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir, model=False, data=False)
    truth_file = _write_json_file(tmp_path, "truth.json", '{"mu": 0.0}\n')
    targets_file = _write_json_file(tmp_path, "targets.json", '{"targets": []}\n')

    code = main(
        [
            "recover-check",
            str(run_dir),
            "--truth",
            str(truth_file),
            "--targets",
            str(targets_file),
            "--interval",
            "0.8",
            "--dry-run",
        ]
    )

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "engine_command": [
            "bayesite",
            "recover-check",
            "--fit",
            str(run_dir / "posterior.ndjson"),
            "--truth",
            str(truth_file.resolve()),
            "--targets",
            str(targets_file.resolve()),
            "--interval",
            "0.8",
            "--out",
            str(run_dir / "recovery_check.json"),
        ],
        "output": str(run_dir / "recovery_check.json"),
        "run": str(run_dir),
    }


def test_recover_check_reports_missing_posterior(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir, model=False, data=False, posterior=False)
    truth_file = _write_json_file(tmp_path, "truth.json", '{"mu": 0.0}\n')

    code = main(["recover-check", str(run_dir), "--truth", str(truth_file), "--dry-run"])

    assert code == 2
    err = capsys.readouterr().err
    assert "missing required run artifact" in err
    assert "posterior.ndjson" in err


def test_recover_check_clears_stale_output_when_engine_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run_artifacts(run_dir, model=False, data=False)
    stale_report = run_dir / "recovery_check.json"
    stale_report.write_text("stale report\n", encoding="utf-8")
    truth_file = _write_json_file(tmp_path, "truth.json", '{"mu": 0.0}\n')
    fake_engine = tmp_path / "failing_bayesite.py"
    fake_engine.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "\n"
        "args = sys.argv[1:]\n"
        + _fake_bayesite_usage_prelude()
        + "print('engine failed before writing --out', file=sys.stderr)\n"
        "raise SystemExit(19)\n",
        encoding="utf-8",
    )
    fake_engine.chmod(0o755)

    code = main(
        [
            "recover-check",
            str(run_dir),
            "--truth",
            str(truth_file),
            "--engine",
            str(fake_engine),
        ]
    )

    assert code == 19
    assert not stale_report.exists()


def test_sample_dry_run_writes_dims_sidecar_when_model_declares_dims(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "from jaxstanv5 import Data, Dim, Observed, Param, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "obs = Dim('obs')\n"
        "predictor = Dim('predictor', coords=['x1', 'x2'])\n"
        "\n"
        "@model\n"
        "class LinearRegression:\n"
        "    x = Data.matrix(3, 2, dims=(obs, predictor))\n"
        "    beta = Param(Normal(0.0, 1.0), size=2, dims=(predictor,))\n"
        "    alpha = Param(Normal(0.0, 1.0))\n"
        "    y = Observed(Normal(0.0, 1.0), dims=(obs,))\n",
        encoding="utf-8",
    )
    data_file = tmp_path / "input.json"
    data_file.write_text(
        '{"x": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "y": [1.0, 2.0, 3.0]}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--dry-run",
        ]
    )

    assert code == 0
    dims_path = output_dir / "dims.json"
    assert json.loads(dims_path.read_text(encoding="utf-8")) == {
        "dims_format": "bayescycle-dims-v1",
        "dims": {
            "alpha": [],
            "beta": ["predictor"],
            "x": ["obs", "predictor"],
            "y": ["obs"],
        },
        "coords": {"predictor": ["x1", "x2"]},
    }

    printed = json.loads(capsys.readouterr().out)
    assert printed["dims"] == str(dims_path)


def test_sample_force_removes_stale_dims_sidecar_when_model_has_no_dims(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dims_model_file = tmp_path / "dims_model.py"
    dims_model_file.write_text(
        "from jaxstanv5 import Dim, Observed, Param, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "predictor = Dim('predictor')\n"
        "\n"
        "@model\n"
        "class WithDims:\n"
        "    beta = Param(Normal(0.0, 1.0), size=2, dims=(predictor,))\n"
        "    y = Observed(Normal(0.0, 1.0))\n",
        encoding="utf-8",
    )
    plain_model_file = tmp_path / "plain_model.py"
    plain_model_file.write_text(
        "from jaxstanv5 import Observed, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "@model\n"
        "class Plain:\n"
        "    y = Observed(Normal(0.0, 1.0))\n",
        encoding="utf-8",
    )
    data_file = tmp_path / "input.json"
    data_file.write_text('{"y": 0.25}\n', encoding="utf-8")
    output_dir = tmp_path / "run"

    first_code = main(
        [
            "sample",
            str(dims_model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--dry-run",
        ]
    )
    assert first_code == 0
    assert (output_dir / "dims.json").exists()
    capsys.readouterr()

    second_code = main(
        [
            "sample",
            str(plain_model_file),
            "--data",
            str(data_file),
            "-o",
            str(output_dir),
            "--force",
            "--dry-run",
        ]
    )

    assert second_code == 0
    assert not (output_dir / "dims.json").exists()
    assert "dims" not in json.loads(capsys.readouterr().out)


def test_sample_rejects_dims_sidecar_rank_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "from jaxstanv5 import Dim, Observed, Param, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "predictor = Dim('predictor')\n"
        "\n"
        "@model\n"
        "class BadDims:\n"
        "    beta = Param(Normal(0.0, 1.0), size=2, dims=(predictor,))\n"
        "    y = Observed(Normal(0.0, 1.0))\n",
        encoding="utf-8",
    )
    data_file = tmp_path / "input.json"
    data_file.write_text('{"y": 0.25}\n', encoding="utf-8")

    def mismatched_dimension_metadata(_metadata: object) -> object:
        return {"dims": {"beta": ["predictor", "extra"]}, "coords": {}}

    monkeypatch.setattr(dims_module, "dimension_metadata_to_dict", mismatched_dimension_metadata)

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(tmp_path / "run"),
            "--dry-run",
        ]
    )

    assert code == 2
    assert "invalid model dimension metadata" in capsys.readouterr().err


def test_sample_requires_explicit_model_when_file_declares_multiple_models(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "from jaxstanv5 import Observed, model\n"
        "from jaxstanv5.distributions import Normal\n"
        "\n"
        "@model\n"
        "class First:\n"
        "    y = Observed(Normal(0.0, 1.0))\n"
        "\n"
        "@model\n"
        "class Second:\n"
        "    y = Observed(Normal(1.0, 2.0))\n",
        encoding="utf-8",
    )
    data_file = tmp_path / "input.json"
    data_file.write_text('{"y": 0.25}\n', encoding="utf-8")

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(tmp_path / "run"),
            "--dry-run",
        ]
    )

    assert code == 2
    assert "multiple jaxstanv5 models" in capsys.readouterr().err
