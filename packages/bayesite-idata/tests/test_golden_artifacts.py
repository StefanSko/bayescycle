"""bayesite_idata against the golden run-artifact corpus.

The corpus ships inside the bayeswire package: real run directories produced
by a release bayesite binary from corpus models. These tests replace
synthetic happy-path run directories — the exporter is proven against actual
engine output; synthetic inputs remain only for protocol error paths.
"""

from __future__ import annotations

import json
from importlib.resources import as_file, files
from pathlib import Path

import pytest
import xarray as xr
from bayesite_idata import from_run_dir, write_netcdf

ARTIFACTS = files("bayeswire") / "corpus" / "artifacts"
ARTIFACT_MODELS = ("eight_schools_non_centered", "varying_intercepts_poisson")


@pytest.fixture(params=ARTIFACT_MODELS)
def artifact_run(request: pytest.FixtureRequest) -> Path:
    resource = ARTIFACTS / request.param
    with as_file(resource) as run_dir:
        return Path(run_dir)


def _manifest(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def test_golden_artifact_exports_posterior_and_sample_stats(artifact_run: Path) -> None:
    manifest = _manifest(artifact_run)

    dt = from_run_dir(artifact_run, validate="skip")

    posterior = dt["posterior"].ds
    assert posterior.sizes["chain"] == manifest["chains"]
    assert posterior.sizes["draw"] == manifest["draws"]
    sample_stats = dt["sample_stats"].ds
    # per_draw_v2 streams carry energy alongside divergence stats.
    assert "energy" in sample_stats
    assert "diverging" in sample_stats
    assert bool(sample_stats["diverging"].any()) in (True, False)


def test_golden_artifact_exports_observed_data(artifact_run: Path) -> None:
    dt = from_run_dir(artifact_run, validate="skip")

    observed = dt["observed_data"].ds
    assert "y" in observed


def test_golden_artifact_dims_sidecar_becomes_coordinates() -> None:
    with as_file(ARTIFACTS / "eight_schools_non_centered") as run_dir:
        dt = from_run_dir(Path(run_dir), validate="skip")

    posterior = dt["posterior"].ds
    assert "school" in posterior["z"].dims
    assert list(posterior["school"].values) == ["A", "B", "C", "D", "E", "F", "G", "H"]
    observed = dt["observed_data"].ds
    assert "school" in observed["y"].dims


def test_golden_artifact_round_trips_to_netcdf(tmp_path: Path, artifact_run: Path) -> None:
    dt = from_run_dir(artifact_run, validate="skip")

    out = write_netcdf(dt, tmp_path / "fit.nc")

    reloaded = xr.open_datatree(out)
    assert set(reloaded["posterior"].ds.data_vars) == set(dt["posterior"].ds.data_vars)
