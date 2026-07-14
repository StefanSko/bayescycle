from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from bayescycle._cli import main


def _write_model(tmp_path: Path) -> Path:
    path = tmp_path / "model.py"
    path.write_text(
        "from bayeswire import Data, Observed, Param, model\n"
        "from bayeswire.distributions import Normal\n"
        "\n"
        "@model\n"
        "class Linear:\n"
        "    alpha = Param(Normal(0.0, 1.0))\n"
        "    x = Data.vector()\n"
        "    y = Observed(Normal(alpha, 1.0))\n",
        encoding="utf-8",
    )
    return path


def _write_fake_generation_engine(tmp_path: Path) -> Path:
    path = tmp_path / "fake_bayesite.py"
    path.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if args == ['capabilities']:\n"
        "    print(json.dumps({'capabilities_format':'v0-provisional','version':'0.test','commands':['generate'],'ir':{'bayeswire_ir':1},'schemas':{}}))\n"
        "    raise SystemExit(0)\n"
        "if not args or args[0] != 'generate':\n"
        "    print('usage: bayesite generate')\n"
        "    raise SystemExit(0)\n"
        "def value(flag): return args[args.index(flag)+1]\n"
        "model = Path(value('--model')).read_bytes()\n"
        "design_bytes = Path(value('--design')).read_bytes()\n"
        "parameters_bytes = Path(value('--parameters')).read_bytes()\n"
        "design = json.loads(design_bytes)\n"
        "parameters = json.loads(parameters_bytes)\n"
        "count, seed = int(value('--count')), int(value('--seed'))\n"
        "sha = lambda data: 'sha256:' + hashlib.sha256(data).hexdigest()\n"
        "common = {'generated_datasets_format':'v0-provisional','artifact_kind':'generated_dataset_pairs','artifact_scope':'parameter_and_complete_dataset_joint_draws'}\n"
        "phases = ['parse_json','decode_ir','bind_design','draw_parameters','simulate_outcomes','emit_artifact']\n"
        "source = {'kind':'fixed','parameters_hash':sha(parameters_bytes)}\n"
        "schema = lambda doc: [{'name':name,'dtype':spec['dtype'],'shape':spec['shape']} for name,spec in doc['variables'].items()]\n"
        "header = {**common,'workflow_phases':phases,'generation_model_hash':sha(model),'design_hash':sha(design_bytes),'parameter_source':source,'count':count,'seed':seed,'draw_index_base':'zero_based_generation_order','parameter_schema':schema(parameters),'dataset_schema':schema(design)}\n"
        "draws = [{**common,'draw_index':index,'draw_count':count,'parameters':parameters,'dataset':design,'source_lineage':{'kind':'fixed'}} for index in range(count)]\n"
        "trailer = {**common,'workflow_phases':phases,'generation_model_hash':sha(model),'design_hash':sha(design_bytes),'parameter_source':source,'count':count,'seed':seed,'draw_count':count,'complete':True}\n"
        "text = '\\n'.join(json.dumps(item,separators=(',',':')) for item in [header,*draws,{'trailer':trailer}]) + '\\n'\n"
        "Path(value('--out')).write_text(text, encoding='utf-8')\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _canonical(path: Path, variables: dict[str, dict[str, Any]]) -> Path:
    path.write_text(
        json.dumps({"format": "bayescycle.data.json.v1", "variables": variables}, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    return path


def test_generation_run_is_portable_and_replays_without_python_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _write_model(tmp_path)
    design = _canonical(
        tmp_path / "design.json",
        {"x": {"dtype": "float64", "shape": [3], "values": [-1.0, 0.0, 1.0]}},
    )
    parameters = _canonical(
        tmp_path / "parameters.json",
        {"alpha": {"dtype": "float64", "shape": [], "values": [0.5]}},
    )
    engine = _write_fake_generation_engine(tmp_path)
    run_dir = tmp_path / "run"

    code = main(
        [
            "generate",
            str(model),
            "--design",
            str(design),
            "--source",
            "fixed",
            "--parameters",
            str(parameters),
            "--count",
            "2",
            "--seed",
            "7",
            "--engine",
            str(engine),
            "-o",
            str(run_dir),
        ]
    )

    assert code == 0
    capsys.readouterr()
    assert {path.name for path in run_dir.iterdir()} == {
        "model.ir.json",
        "design.json",
        "fixed-parameters.json",
        "generation-plan.json",
        "generated_datasets.ndjson",
        "run.json",
    }
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["format"] == "bayescycle.generation-run.v0"
    assert metadata["kind"] == "generate"
    assert [entry["role"] for entry in metadata["inputs"]] == ["design", "fixed-parameters"]
    assert metadata["outputs"][0]["role"] == "generated-datasets"

    moved = tmp_path / "moved" / "run"
    moved.parent.mkdir()
    shutil.move(run_dir, moved)
    model.unlink()
    design.unlink()
    parameters.unlink()

    replay_dir = tmp_path / "replay"
    check_code = main(
        [
            "replay",
            str(moved),
            "-o",
            str(replay_dir),
            "--engine",
            str(engine),
            "--check-only",
        ]
    )
    assert check_code == 0
    assert not replay_dir.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["kind"] == "generate"
    assert all(str(moved) in entry["path"] for entry in plan["verified_sources"])

    replay_code = main(
        ["replay", str(moved), "-o", str(replay_dir), "--engine", str(engine)]
    )
    assert replay_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["byte_identical"] is True
    assert (replay_dir / "generated_datasets.ndjson").read_bytes() == (
        moved / "generated_datasets.ndjson"
    ).read_bytes()
