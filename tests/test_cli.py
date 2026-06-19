from __future__ import annotations

import json
from pathlib import Path

import pytest

import bayescycle._dims as dims_module
from bayescycle._cli import main


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
    assert run_data_path.read_text(encoding="utf-8") == '{"y": 0.25}\n'
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
            str(run_data_path),
            "--seed",
            "123",
        ],
        "ir": str(ir_path),
        "model": "Simple",
        "output": str(output_dir),
    }


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
