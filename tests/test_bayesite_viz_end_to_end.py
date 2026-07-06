"""End-to-end coverage of `bayescycle idata` / `bayescycle plot` against real uvx.

This exercises the two-command story end to end: `bayescycle sample` (real
Bayesite engine) produces a run directory, `bayescycle idata` exports it to
ArviZ NetCDF through the pinned bayesite-viz uvx invocation, and
`bayescycle plot trace` renders a real PNG from that fit file. Set
``BAYESCYCLE_TEST_UVX=1`` and point ``BAYESCYCLE_TEST_BAYESITE_BIN`` at a real
Bayesite executable to enable it; both must be set because the run directory
this test plots from is produced by the real engine, mirroring
tests/test_bayesite_end_to_end.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bayescycle._cli import main

BAYESITE_BIN = os.environ.get("BAYESCYCLE_TEST_BAYESITE_BIN")
UVX_ENABLED = os.environ.get("BAYESCYCLE_TEST_UVX") == "1"

pytestmark = pytest.mark.skipif(
    BAYESITE_BIN is None or not UVX_ENABLED,
    reason=(
        "set BAYESCYCLE_TEST_BAYESITE_BIN to a bayesite binary and "
        "BAYESCYCLE_TEST_UVX=1 to run the real bayesite-viz uvx end-to-end"
    ),
)

MODEL_SOURCE = (
    "from bayeswire import Dim, Observed, Param, model\n"
    "from bayeswire.distributions import Normal\n"
    "\n"
    'obs = Dim("obs", coords=["a", "b", "c", "d"])\n'
    "\n"
    "\n"
    "@model\n"
    "class VizEndToEndNormal:\n"
    "    mu = Param(Normal(0.0, 1.0))\n"
    "    y = Observed(Normal(mu, 1.0), dims=(obs,))\n"
)


def test_idata_then_plot_trace_with_real_uvx(tmp_path: Path) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text(MODEL_SOURCE, encoding="utf-8")
    data_path = tmp_path / "data.json"
    data_path.write_text('{"y": [0.2, 0.5, 0.9, 1.4]}\n', encoding="utf-8")
    run_dir = tmp_path / "run"

    assert BAYESITE_BIN is not None
    code = main(
        [
            "sample",
            str(model_path),
            "--data",
            str(data_path),
            "-o",
            str(run_dir),
            "--engine",
            BAYESITE_BIN,
            "--seed",
            "20260706",
            "--chains",
            "2",
            "--warmup",
            "200",
            "--draws",
            "200",
        ]
    )
    assert code == 0

    code = main(["idata", str(run_dir), "--validate", "require", "--engine", BAYESITE_BIN])
    assert code == 0
    fit_path = run_dir / "fit.nc"
    assert fit_path.is_file()
    assert fit_path.stat().st_size > 0

    plot_path = tmp_path / "trace.png"
    code = main(["plot", "trace", str(run_dir), "-o", str(plot_path)])
    assert code == 0
    assert plot_path.is_file()
    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
