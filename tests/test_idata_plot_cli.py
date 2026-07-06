"""CLI coverage for `bayescycle idata` and `bayescycle plot`.

These tests stay offline: paths that would otherwise shell out to `uvx`
(and from there fetch bayesite-viz over the network) are exercised only up
to the point bayescycle itself would reject them -- a missing run directory,
a missing fit file with auto-idata disabled, or `uvx` absent from PATH. The
real end-to-end path (uvx actually invoked) lives in
tests/test_bayesite_viz_end_to_end.py, gated by BAYESCYCLE_TEST_UVX.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bayescycle._cli import main


@pytest.mark.parametrize("command", ("idata", "plot"))
def test_idata_and_plot_help_exit_cleanly(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([command, "--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--no-auto-provision" in out


def test_idata_missing_run_dir_reports_workflow_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "no-such-run"

    code = main(["idata", str(missing)])

    assert code == 2
    err = capsys.readouterr().err
    assert "bayescycle: " in err
    assert "run directory does not exist" in err


def test_idata_reports_missing_uvx_without_touching_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit --engine skips auto-provisioning; only uvx is missing."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))

    code = main(["idata", str(run_dir), "--engine", "/fake/bayesite"])

    assert code == 2
    err = capsys.readouterr().err
    assert "bayescycle: " in err
    assert "uvx" in err


def test_plot_missing_run_dir_reports_workflow_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "no-such-run"

    code = main(["plot", "trace", str(missing)])

    assert code == 2
    err = capsys.readouterr().err
    assert "bayescycle: " in err
    assert "run directory does not exist" in err


def test_plot_no_auto_idata_with_missing_fit_reports_actionable_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    code = main(["plot", "trace", str(run_dir), "--no-auto-idata"])

    assert code == 2
    err = capsys.readouterr().err
    assert "bayescycle: " in err
    assert "fit file not found" in err
    assert "bayescycle idata" in err


def test_plot_rejects_unknown_verb_via_argparse(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["plot", "not-a-real-verb", "run"])

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err


def test_plot_auto_idata_reports_missing_uvx_without_touching_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Auto-idata triggers on a missing fit.nc; uvx-missing still short-circuits before uvx runs."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))

    code = main(["plot", "trace", str(run_dir), "--engine", "/fake/bayesite"])

    assert code == 2
    err = capsys.readouterr().err
    assert "bayescycle: " in err
    assert "uvx" in err


def test_plot_with_existing_fit_skips_auto_idata_and_reports_missing_uvx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "fit.nc").write_bytes(b"not-a-real-netcdf-file")
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))

    code = main(["plot", "trace", str(run_dir)])

    assert code == 2
    err = capsys.readouterr().err
    assert "uvx" in err
