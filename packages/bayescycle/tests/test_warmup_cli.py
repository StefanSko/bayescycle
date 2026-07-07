"""CLI coverage for `bayescycle warmup`.

Like tests/test_idata_plot_cli.py, this stays offline: the only path
exercised is the one bayescycle itself would reject before shelling out to
`uvx` (missing on `PATH`). The real end-to-end path (uvx actually invoked)
would live alongside tests/test_bayesite_viz_end_to_end.py, gated the same
way, if this warmup surface grows one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bayescycle._cli import main


def test_warmup_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["warmup", "--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "warmup" in out.lower()


def test_warmup_reports_missing_uvx_without_touching_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))

    code = main(["warmup"])

    assert code == 2
    err = capsys.readouterr().err
    assert "bayescycle: " in err
    assert "uvx" in err
