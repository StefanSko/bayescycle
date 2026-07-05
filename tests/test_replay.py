from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from bayescycle._cli import main

FAKE_BAYESITE_USAGE = "usage: bayesite sample"


def _fake_bayesite_usage_prelude() -> str:
    return (
        "if not args or args == ['--help'] or args == ['__bayescycle_capability_probe__']:\n"
        f"    print({FAKE_BAYESITE_USAGE!r})\n"
        "    raise SystemExit(0)\n"
    )


def _write_simple_model(tmp_path: Path) -> Path:
    model_file = tmp_path / "model.py"
    model_file.write_text(
        "from bayeswire import Observed, model\n"
        "from bayeswire.distributions import Normal\n"
        "\n"
        "@model\n"
        "class Simple:\n"
        "    y = Observed(Normal(0.0, 1.0))\n",
        encoding="utf-8",
    )
    return model_file


def _write_input_data(tmp_path: Path) -> Path:
    data_file = tmp_path / "input.json"
    data_file.write_text('{"y": 0.25}\n', encoding="utf-8")
    return data_file


def _write_constant_sample_engine(tmp_path: Path, content: str = "draws\n") -> Path:
    fake_engine = tmp_path / "fake_bayesite.py"
    fake_engine.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "args = sys.argv[1:]\n"
        + _fake_bayesite_usage_prelude()
        + "out_index = args.index('--out')\n"
        f"Path(args[out_index + 1]).write_text({content!r}, encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_engine.chmod(0o755)
    return fake_engine


def _write_path_sensitive_sample_engine(tmp_path: Path) -> Path:
    fake_engine = tmp_path / "fake_bayesite.py"
    fake_engine.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "args = sys.argv[1:]\n"
        + _fake_bayesite_usage_prelude()
        + "out_index = args.index('--out')\n"
        "out_path = Path(args[out_index + 1])\n"
        "content = 'replay draws\\n' if 'replay' in out_path.parts else 'original draws\\n'\n"
        "out_path.write_text(content, encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_engine.chmod(0o755)
    return fake_engine


def _create_sample_run(
    tmp_path: Path,
    *,
    fake_engine: Path,
    capsys: pytest.CaptureFixture[str],
) -> tuple[Path, Path, Path]:
    model_file = _write_simple_model(tmp_path)
    data_file = _write_input_data(tmp_path)
    run_dir = tmp_path / "run"

    code = main(
        [
            "sample",
            str(model_file),
            "--data",
            str(data_file),
            "-o",
            str(run_dir),
            "--engine",
            str(fake_engine),
            "--seed",
            "7",
            "--chains",
            "2",
            "--warmup",
            "5",
            "--draws",
            "6",
        ]
    )

    assert code == 0
    capsys.readouterr()
    return model_file, data_file, run_dir


def _artifact_by_role(document: dict[str, Any], role: str) -> dict[str, Any]:
    matches = [artifact for artifact in document["artifacts"] if artifact["role"] == role]
    assert len(matches) == 1
    return matches[0]


def test_replay_check_only_verifies_hashes_and_reconstructs_sample_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_engine = _write_constant_sample_engine(tmp_path)
    model_file, data_file, run_dir = _create_sample_run(
        tmp_path, fake_engine=fake_engine, capsys=capsys
    )
    replay_dir = tmp_path / "replay"

    run_metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_metadata["settings"] == {
        "chains": "2",
        "draws": "6",
        "seed": "7",
        "warmup": "5",
    }

    code = main(
        [
            "replay",
            str(run_dir),
            "-o",
            str(replay_dir),
            "--engine",
            str(fake_engine),
            "--check-only",
        ]
    )

    assert code == 0
    assert not replay_dir.exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["format"] == "bayescycle.replay-plan.v1"
    assert printed["source_run"] == str(run_dir.resolve())
    assert printed["output"] == str(replay_dir.resolve())
    assert printed["kind"] == "sample"
    assert printed["backend"] == "bayesite"
    assert printed["verified_sources"] == [
        {
            "role": "model",
            "path": str(model_file.resolve()),
            "sha256": run_metadata["model"]["sha256"],
        },
        {
            "role": "input data",
            "path": str(data_file.resolve()),
            "sha256": run_metadata["inputs"][0]["sha256"],
        },
    ]
    assert printed["plan"]["command"] == [
        str(fake_engine),
        "sample",
        "--model",
        str(replay_dir.resolve() / "model.ir.json"),
        "--data",
        str(replay_dir.resolve() / "data.json"),
        "--seed",
        "7",
        "--chains",
        "2",
        "--warmup",
        "5",
        "--draws",
        "6",
        "--out",
        str(replay_dir.resolve() / "posterior.ndjson"),
    ]


def test_replay_refuses_drifted_input_before_creating_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_engine = _write_constant_sample_engine(tmp_path)
    _, data_file, run_dir = _create_sample_run(tmp_path, fake_engine=fake_engine, capsys=capsys)
    data_file.write_text('{"y": 9.0}\n', encoding="utf-8")
    replay_dir = tmp_path / "replay"

    code = main(["replay", str(run_dir), "-o", str(replay_dir), "--engine", str(fake_engine)])

    assert code == 2
    assert "hash mismatch for input data" in capsys.readouterr().err
    assert not replay_dir.exists()


def test_replay_executes_and_reports_byte_identical_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_engine = _write_constant_sample_engine(tmp_path)
    _, _, run_dir = _create_sample_run(tmp_path, fake_engine=fake_engine, capsys=capsys)
    replay_dir = tmp_path / "replay"

    code = main(["replay", str(run_dir), "-o", str(replay_dir), "--engine", str(fake_engine)])

    assert code == 0
    assert (replay_dir / "posterior.ndjson").read_text(encoding="utf-8") == "draws\n"
    printed = json.loads(capsys.readouterr().out)
    assert printed["format"] == "bayescycle.replay-result.v1"
    assert printed["status"] == "byte-identical"
    assert printed["byte_identical"] is True
    assert {artifact["role"] for artifact in printed["artifacts"]} == {
        "model_ir",
        "data",
        "posterior",
    }
    assert _artifact_by_role(printed, "posterior")["byte_identical"] is True


def test_replay_returns_one_with_per_artifact_report_on_difference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_engine = _write_path_sensitive_sample_engine(tmp_path)
    _, _, run_dir = _create_sample_run(tmp_path, fake_engine=fake_engine, capsys=capsys)
    replay_dir = tmp_path / "replay"

    code = main(["replay", str(run_dir), "-o", str(replay_dir), "--engine", str(fake_engine)])

    assert code == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "different"
    assert printed["byte_identical"] is False
    posterior = _artifact_by_role(printed, "posterior")
    assert posterior["path"] == "posterior.ndjson"
    assert posterior["status"] == "different"
    assert posterior["byte_identical"] is False
