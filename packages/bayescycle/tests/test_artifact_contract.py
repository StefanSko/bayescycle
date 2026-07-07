from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_runs() -> list[Path]:
    return sorted(path for path in FIXTURES.iterdir() if path.is_dir())


def test_v0_fixture_runs_have_required_contract_files() -> None:
    runs = _fixture_runs()
    assert runs, "expected committed run-directory v0 fixtures"
    for run in runs:
        assert (run / "model.ir.json").is_file()
        assert (run / "data.json").is_file()
        posterior = run / "posterior.ndjson"
        assert posterior.is_file()
        header = json.loads(posterior.read_text(encoding="utf-8").splitlines()[0])
        assert header["draws_format"] == "v0-provisional"
        assert header["artifact_kind"] == "posterior_draws"


@pytest.mark.parametrize("run", _fixture_runs(), ids=lambda path: path.name)
def test_v0_fixture_posterior_is_accepted_by_bayesite_diagnose(
    run: Path,
    tmp_path: Path,
) -> None:
    bayesite = shutil.which("bayesite")
    if bayesite is None:
        pytest.skip("bayesite executable is not installed")
    out = tmp_path / f"{run.name}-diagnostics.json"
    completed = subprocess.run(  # noqa: S603
        [bayesite, "diagnose", "--fit", str(run / "posterior.ndjson"), "--out", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["diagnostics_format"] == "v0-provisional"


@pytest.mark.parametrize("run", _fixture_runs(), ids=lambda path: path.name)
def test_v0_fixture_run_is_accepted_by_bayesite_idata(run: Path, tmp_path: Path) -> None:
    bayesite_idata = shutil.which("bayesite-idata")
    if bayesite_idata is None:
        pytest.skip("bayesite-idata executable is not installed")
    out = tmp_path / f"{run.name}.nc"
    completed = subprocess.run(  # noqa: S603
        [bayesite_idata, str(run), "-o", str(out), "--validate", "skip"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert out.is_file()
