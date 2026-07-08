from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr
from bayesite_viz.io import load_fit, require_groups


def test_load_fit_returns_datatree(continuous_fit: Path) -> None:
    dt = load_fit(continuous_fit)
    assert isinstance(dt, xr.DataTree)
    assert "posterior" in {g.lstrip("/") for g in dt.groups}


def test_load_fit_directory_is_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as ei:
        load_fit(tmp_path)
    assert ei.value.code == 2


def test_load_fit_missing_file_is_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as ei:
        load_fit(tmp_path / "nope.nc")
    assert ei.value.code == 2


def test_require_groups_passes_when_present(continuous_fit: Path) -> None:
    dt = load_fit(continuous_fit)
    require_groups(dt, "posterior")  # no raise


def test_require_groups_errors_when_missing(continuous_fit: Path) -> None:
    dt = load_fit(continuous_fit)
    with pytest.raises(SystemExit) as ei:
        require_groups(dt, "posterior_predictive", "observed_data", "prior")
    assert ei.value.code == 3


def test_load_fit_ndjson_repair_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "posterior.ndjson"
    p.write_text('{"mu": 0.1}\n{"mu": 0.2}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        load_fit(p)
    assert ei.value.code == 5
    err = capsys.readouterr().err
    assert "not an InferenceData" in err
    assert "bayesite-idata" in err


def test_load_fit_json_repair_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "posterior.json"
    p.write_text('{"posterior": {"mu": [[0.1]]}}', encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        load_fit(p)
    assert ei.value.code == 5
    assert "bayesite-idata" in capsys.readouterr().err


def test_load_fit_binary_garbage_is_generic_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = tmp_path / "weird.nc"
    p.write_bytes(b"\x00\x01\x02not a netcdf\xff\xfe")
    with pytest.raises(SystemExit) as ei:
        load_fit(p)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "failed to load InferenceData" in err
