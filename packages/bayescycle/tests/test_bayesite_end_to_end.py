"""End-to-end workflow against a real bayesite release binary.

This is the default agent path: model.py executes with bayeswire only (no
JAX, no bayesjax), the run directory is prepared by bayescycle, and a real
bayesite binary samples and diagnoses. Point
``BAYESCYCLE_TEST_BAYESITE_BIN`` at a bayesite executable to enable it; the
no-JAX CI job always does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bayescycle._cli import main

BAYESITE_BIN = os.environ.get("BAYESCYCLE_TEST_BAYESITE_BIN")

pytestmark = pytest.mark.skipif(
    BAYESITE_BIN is None,
    reason="set BAYESCYCLE_TEST_BAYESITE_BIN to a bayesite binary to run end-to-end",
)

MODEL_SOURCE = (
    "from bayeswire import Dim, Observed, Param, model\n"
    "from bayeswire.distributions import Normal\n"
    "\n"
    'obs = Dim("obs", coords=["a", "b", "c", "d"])\n'
    "\n"
    "\n"
    "@model\n"
    "class EndToEndNormal:\n"
    "    mu = Param(Normal(0.0, 1.0))\n"
    "    y = Observed(Normal(mu, 1.0), dims=(obs,))\n"
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    model_path = tmp_path / "model.py"
    model_path.write_text(MODEL_SOURCE, encoding="utf-8")
    data_path = tmp_path / "data.json"
    data_path.write_text('{"y": [0.2, 0.5, 0.9, 1.4]}\n', encoding="utf-8")
    return model_path, data_path


def test_sample_and_diagnose_with_real_engine(tmp_path: Path) -> None:
    model_path, data_path = _write_inputs(tmp_path)
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
            "20260702",
            "--chains",
            "2",
            "--warmup",
            "200",
            "--draws",
            "200",
        ]
    )
    assert code == 0

    ir_document = json.loads((run_dir / "model.ir.json").read_text(encoding="utf-8"))
    assert ir_document["bayeswire_ir"] == 1
    dims_document = json.loads((run_dir / "dims.json").read_text(encoding="utf-8"))
    assert dims_document["dims"] == {"mu": [], "y": ["obs"]}
    draws_lines = (run_dir / "posterior.ndjson").read_text(encoding="utf-8").splitlines()
    assert len(draws_lines) > 200

    code = main(
        [
            "diagnose",
            str(run_dir),
            "--engine",
            BAYESITE_BIN,
        ]
    )
    assert code == 0
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics, "diagnostics.json must be a non-empty document"


def test_bayesjax_fit_passes_bayesite_posterior_predictive_and_check(tmp_path: Path) -> None:
    """A bayesjax-produced fit must pass the real engine's fingerprint check.

    ``sample`` runs on the in-process bayesjax backend (fingerprinting the
    canonical ``run/data.json`` it was handed); ``posterior-predictive`` and
    ``posterior-check`` then run against the real Bayesite engine, which must
    fingerprint the same bytes for the same run directory.
    """
    pytest.importorskip("bayesjax")
    model_path, data_path = _write_inputs(tmp_path)
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
            "--backend",
            "bayesjax",
            "--seed",
            "20260702",
            "--chains",
            "2",
            "--warmup",
            "200",
            "--draws",
            "200",
        ]
    )
    assert code == 0

    code = main(
        [
            "posterior-predictive",
            str(run_dir),
            "--seed",
            "20260703",
            "--engine",
            BAYESITE_BIN,
        ]
    )
    assert code == 0
    assert (run_dir / "posterior_predictive.ndjson").is_file()

    code = main(
        [
            "posterior-check",
            str(run_dir),
            "--engine",
            BAYESITE_BIN,
        ]
    )
    assert code == 0
    assert (run_dir / "posterior_check.json").is_file()
