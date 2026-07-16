from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from bayescycle._cli import main
from bayescycle._errors import WorkflowError
from bayescycle._workflow.generation_plan import (
    FitArtifact,
    FitAssociation,
    Fixed,
    PosteriorOf,
    generate_datasets,
)
from bayescycle._workflow.generation_runs import (
    execute_generation_run,
    load_generation_run,
    materialize_generation_run,
)


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
    script = f"""#!{sys.executable}
import hashlib, json, sys
from pathlib import Path
args = sys.argv[1:]
if args == ["capabilities"]:
    print(json.dumps({{
        "capabilities_format": "v0-provisional",
        "version": "0.test",
        "commands": ["generate"],
        "ir": {{"bayeswire_ir": 1}},
        "schemas": {{}},
    }}))
    raise SystemExit(0)
if not args or args[0] != "generate":
    print("usage: bayesite generate")
    raise SystemExit(0)
def value(flag):
    return args[args.index(flag) + 1]
model = Path(value("--model")).read_bytes()
design_bytes = Path(value("--design")).read_bytes()
parameters_bytes = Path(value("--parameters")).read_bytes()
design = json.loads(design_bytes)
parameters = json.loads(parameters_bytes)
count, seed = int(value("--count")), int(value("--seed"))
sha = lambda data: "sha256:" + hashlib.sha256(data).hexdigest()
common = {{
    "generated_datasets_format": "v0-provisional",
    "artifact_kind": "generated_dataset_pairs",
    "artifact_scope": "parameter_and_complete_dataset_joint_draws",
}}
phases = [
    "parse_json", "decode_ir", "bind_design", "draw_parameters",
    "simulate_outcomes", "emit_artifact",
]
source = {{"kind": "fixed", "parameters_hash": sha(parameters_bytes)}}
def schema(document):
    return [
        {{"name": name, "dtype": spec["dtype"], "shape": spec["shape"]}}
        for name, spec in document["variables"].items()
    ]
header = {{
    **common,
    "workflow_phases": phases,
    "generation_model_hash": sha(model),
    "design_hash": sha(design_bytes),
    "parameter_source": source,
    "count": count,
    "seed": seed,
    "draw_index_base": "zero_based_generation_order",
    "parameter_schema": schema(parameters),
    "dataset_schema": schema(design),
}}
draws = [
    {{
        **common,
        "draw_index": index,
        "draw_count": count,
        "parameters": parameters,
        "dataset": design,
        "source_lineage": {{"kind": "fixed"}},
    }}
    for index in range(count)
]
trailer = {{
    **common,
    "workflow_phases": phases,
    "generation_model_hash": sha(model),
    "design_hash": sha(design_bytes),
    "parameter_source": source,
    "count": count,
    "seed": seed,
    "draw_count": count,
    "complete": True,
}}
items = [header, *draws, {{"trailer": trailer}}]
text = "\\n".join(json.dumps(item, separators=(",", ":")) for item in items) + "\\n"
Path(value("--out")).write_text(text, encoding="utf-8")
"""
    path.write_text(textwrap.dedent(script), encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_malformed_generation_engine(tmp_path: Path) -> Path:
    path = tmp_path / "malformed_bayesite.py"
    script = f"""#!{sys.executable}
import json, sys
from pathlib import Path
if sys.argv[1:] == ["capabilities"]:
    print(json.dumps({{
        "capabilities_format": "v0-provisional",
        "version": "0.test",
        "commands": ["generate"],
        "ir": {{"bayeswire_ir": 1}},
        "schemas": {{}},
    }}))
    raise SystemExit(0)
args = sys.argv[1:]
Path(args[args.index("--out") + 1]).write_text('{{"bad":true}}\\n')
"""
    path.write_text(textwrap.dedent(script), encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_symlink_generation_engine(tmp_path: Path) -> Path:
    delegate = _write_fake_generation_engine(tmp_path)
    path = tmp_path / "symlink_bayesite.py"
    outside = tmp_path / "external-generated.ndjson"
    script = f"""#!{sys.executable}
import subprocess, sys
from pathlib import Path
args = sys.argv[1:]
code = subprocess.run([{str(delegate)!r}, *args], check=False).returncode
if code == 0 and args != ["capabilities"]:
    output = Path(args[args.index("--out") + 1])
    external = Path({str(outside)!r})
    output.replace(external)
    output.symlink_to(external)
raise SystemExit(code)
"""
    path.write_text(textwrap.dedent(script), encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_colliding_generation_engine(tmp_path: Path, *, symlink: bool) -> Path:
    delegate = _write_fake_generation_engine(tmp_path)
    suffix = "symlink" if symlink else "regular"
    path = tmp_path / f"colliding_{suffix}_bayesite.py"
    outside = tmp_path / f"outside-{suffix}-run.json"
    script = f"""#!{sys.executable}
import subprocess, sys
from pathlib import Path
args = sys.argv[1:]
code = subprocess.run([{str(delegate)!r}, *args], check=False).returncode
if code == 0 and args != ["capabilities"]:
    output = Path(args[args.index("--out") + 1])
    metadata = output.parent / "run.json"
    outside = Path({str(outside)!r})
    if {symlink!r}:
        outside.write_text("ENGINE SENTINEL\\n")
        metadata.symlink_to(outside)
    else:
        metadata.write_text("ENGINE SENTINEL\\n")
raise SystemExit(code)
"""
    path.write_text(textwrap.dedent(script), encoding="utf-8")
    path.chmod(0o755)
    return path


def _canonical(path: Path, variables: dict[str, dict[str, Any]]) -> Path:
    path.write_text(
        json.dumps(
            {"format": "bayescycle.data.json.v1", "variables": variables}, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_portable_posterior_requires_verified_file_fingerprint(tmp_path: Path) -> None:
    model_bytes = b'{"bayeswire_ir":1,"model":{}}\n'
    data_bytes = b'{"format":"bayescycle.data.json.v1","variables":{}}\n'
    plan = generate_datasets(
        model_bytes,
        design=data_bytes,
        parameter_source=PosteriorOf(
            FitArtifact(
                model_bytes,
                data_bytes,
                b'{"draws_format":"v0-provisional"}\n',
                FitAssociation.PORTABLE,
            )
        ),
        count=1,
        seed=0,
    )
    output = tmp_path / "invalid-posterior-run"
    with pytest.raises(WorkflowError, match="fingerprint|posterior"):
        materialize_generation_run(output_dir=output, plan=plan, engine="bayesite")
    assert not output.exists()


def test_malformed_engine_output_fails_without_publishing_run_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _write_model(tmp_path)
    design = _canonical(
        tmp_path / "malformed-design.json",
        {"x": {"dtype": "float64", "shape": [1], "values": [0.0]}},
    )
    parameters = _canonical(
        tmp_path / "malformed-parameters.json",
        {"alpha": {"dtype": "float64", "shape": [], "values": [0.5]}},
    )
    output = tmp_path / "malformed-run"
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
            "1",
            "--seed",
            "0",
            "--engine",
            str(_write_malformed_generation_engine(tmp_path)),
            "-o",
            str(output),
        ]
    )
    assert code == 2
    assert "generated-dataset" in capsys.readouterr().err
    assert not (output / "run.json").exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_generation_refuses_engine_metadata_collision(tmp_path: Path, symlink: bool) -> None:
    model_bytes = b'{"bayeswire_ir":1,"model":{}}\n'
    design_bytes = (
        b'{"format":"bayescycle.data.json.v1","variables":'
        b'{"x":{"dtype":"float64","shape":[1],"values":[0.0]}}}\n'
    )
    parameter_bytes = (
        b'{"format":"bayescycle.data.json.v1","variables":'
        b'{"alpha":{"dtype":"float64","shape":[],"values":[0.5]}}}\n'
    )
    output = tmp_path / f"collision-{symlink}"
    plan = generate_datasets(
        model_bytes,
        design=design_bytes,
        parameter_source=Fixed(parameter_bytes),
        count=1,
        seed=0,
    )
    with pytest.raises(WorkflowError, match="run.json|metadata|exists"):
        execute_generation_run(
            output_dir=output,
            plan=plan,
            engine=str(_write_colliding_generation_engine(tmp_path, symlink=symlink)),
        )
    metadata = output / "run.json"
    assert metadata.is_symlink() is symlink
    assert metadata.read_text() == "ENGINE SENTINEL\n"


def test_generation_rejects_symlinked_engine_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = _write_model(tmp_path)
    design = _canonical(
        tmp_path / "symlink-design.json",
        {"x": {"dtype": "float64", "shape": [1], "values": [0.0]}},
    )
    parameters = _canonical(
        tmp_path / "symlink-parameters.json",
        {"alpha": {"dtype": "float64", "shape": [], "values": [0.5]}},
    )
    output = tmp_path / "symlink-run"
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
            "1",
            "--seed",
            "0",
            "--engine",
            str(_write_symlink_generation_engine(tmp_path)),
            "-o",
            str(output),
        ]
    )
    assert code == 2
    assert "regular" in capsys.readouterr().err
    assert not (output / "run.json").exists()


def _make_fixed_run(tmp_path: Path) -> Path:
    model = _write_model(tmp_path)
    design = _canonical(
        tmp_path / "bounded-design.json",
        {"x": {"dtype": "float64", "shape": [1], "values": [0.0]}},
    )
    parameters = _canonical(
        tmp_path / "bounded-parameters.json",
        {"alpha": {"dtype": "float64", "shape": [], "values": [0.5]}},
    )
    output = tmp_path / "bounded-run"
    assert (
        main(
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
                "1",
                "--seed",
                "0",
                "--engine",
                str(_write_fake_generation_engine(tmp_path)),
                "-o",
                str(output),
            ]
        )
        == 0
    )
    return output


def test_generation_run_routing_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    run_dir = tmp_path / "fifo-run"
    run_dir.mkdir()
    os.mkfifo(run_dir / "run.json")
    code = (
        "from pathlib import Path\n"
        "from bayescycle._errors import WorkflowError\n"
        "from bayescycle._workflow.generation_runs import is_generation_run\n"
        "try:\n"
        f"    is_generation_run(Path({str(run_dir)!r}))\n"
        "except WorkflowError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False, timeout=2)
    assert completed.returncode == 0


def test_generation_run_rejects_symlink_without_nofollow_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _make_fixed_run(tmp_path)
    design = run_dir / "design.json"
    outside = tmp_path / "outside-design.json"
    design.replace(outside)
    design.symlink_to(outside)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    with pytest.raises(WorkflowError, match="regular"):
        load_generation_run(run_dir)


def test_generation_run_interprets_the_same_bytes_it_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _make_fixed_run(tmp_path)
    output = run_dir / "generated_datasets.ndjson"
    original = output.read_bytes()
    changed = original.replace(b',"dataset":', b', "dataset":', 1)
    assert changed != original
    target_inode = output.stat().st_ino
    read_descriptor = os.read
    raced = False

    def replace_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal raced
        data = read_descriptor(descriptor, size)
        if os.fstat(descriptor).st_ino == target_inode and data and not raced:
            raced = True
            output.write_bytes(changed)
        return data

    monkeypatch.setattr(os, "read", replace_after_first_read)
    with pytest.raises(WorkflowError, match="hash mismatch"):
        load_generation_run(run_dir)
    assert output.read_bytes() == changed


def test_generation_run_rejects_external_unbounded_or_invalid_metadata(tmp_path: Path) -> None:
    run_dir = _make_fixed_run(tmp_path)
    metadata_path = run_dir / "run.json"
    outside = tmp_path / "outside-run.json"
    metadata_path.replace(outside)
    metadata_path.symlink_to(outside)
    with pytest.raises(WorkflowError, match="regular|symbolic|contained"):
        load_generation_run(run_dir)

    metadata_path.unlink()
    metadata_path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(WorkflowError, match="byte|size|MiB"):
        load_generation_run(run_dir)

    metadata = json.loads(outside.read_text(encoding="utf-8"))
    metadata["backend"] = "!" * 65
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":")) + "\n")
    with pytest.raises(WorkflowError, match="backend"):
        load_generation_run(run_dir)

    duplicate = outside.read_bytes().replace(
        b'"backend":"bayesite"', b'"backend":"wrong","backend":"bayesite"'
    )
    metadata_path.write_bytes(duplicate)
    with pytest.raises(WorkflowError, match="duplicate"):
        load_generation_run(run_dir)

    metadata = json.loads(outside.read_text(encoding="utf-8"))
    metadata["inputs"][0]["format"] = "wrong"
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":")) + "\n")
    with pytest.raises(WorkflowError, match="format"):
        load_generation_run(run_dir)

    nested = "[" * 65 + "0" + "]" * 65
    metadata_path.write_text('{"format":' + nested + "}\n")
    with pytest.raises(WorkflowError, match="depth"):
        load_generation_run(run_dir)


def test_generation_run_preserves_design_source_through_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_bytes = b'{"bayeswire_ir":1,"model":{}}\n'
    design_bytes = (
        b'{"format":"bayescycle.data.json.v1","variables":'
        b'{"x":{"dtype":"float64","shape":[1],"values":[0.0]}}}\n'
    )
    parameter_bytes = (
        b'{"format":"bayescycle.data.json.v1","variables":'
        b'{"alpha":{"dtype":"float64","shape":[],"values":[0.5]}}}\n'
    )
    plan = generate_datasets(
        model_bytes,
        design=design_bytes,
        design_source={"x": "repeat([0], 1)"},
        parameter_source=Fixed(parameter_bytes),
        count=1,
        seed=0,
    )
    engine = _write_fake_generation_engine(tmp_path)
    original = tmp_path / "design-source-run"
    assert execute_generation_run(output_dir=original, plan=plan, engine=str(engine)) == 0
    original_plan = (original / "generation-plan.json").read_bytes()
    assert json.loads(original_plan)["design_source"] == {"x": "repeat([0], 1)"}

    replay = tmp_path / "design-source-replay"
    assert (
        main(
            [
                "replay",
                str(original),
                "-o",
                str(replay),
                "--engine",
                str(engine),
                "--check-only",
            ]
        )
        == 0
    )
    assert not replay.exists()
    capsys.readouterr()

    assert main(["replay", str(original), "-o", str(replay), "--engine", str(engine)]) == 0
    assert (replay / "generation-plan.json").read_bytes() == original_plan
    assert json.loads(capsys.readouterr().out)["byte_identical"] is True


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

    replay_code = main(["replay", str(moved), "-o", str(replay_dir), "--engine", str(engine)])
    assert replay_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["byte_identical"] is True
    assert (replay_dir / "generated_datasets.ndjson").read_bytes() == (
        moved / "generated_datasets.ndjson"
    ).read_bytes()
