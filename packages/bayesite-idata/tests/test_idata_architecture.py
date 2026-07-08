from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr
from bayesite_idata import assemble as assemble_module
from bayesite_idata.arviz import dense_to_datatree
from bayesite_idata.dense import CheckedDenseFit, build_checked_dense_fit, validate_dense_fit
from bayesite_idata.protocol_v0 import read_posterior_predictive_stream, read_posterior_stream

from tests.test_bayesite_idata import (
    _posterior_lines,
    _posterior_predictive_lines,
    _write_json,
    _write_ndjson,
)


def _minimal_run(tmp_path: Path) -> Path:
    _write_json(
        tmp_path / "model.ir.json",
        {
            "bayeswire_ir": 1,
            "model": {"node": "ModelMeta", "observed_nodes": [{"name": "y"}]},
        },
    )
    _write_json(
        tmp_path / "data.json",
        {
            "x": {"dtype": "float64", "shape": [2], "values": [1.0, 2.0]},
            "y": {"dtype": "int64", "shape": [2], "values": [3, 4]},
        },
    )
    _write_ndjson(tmp_path / "posterior.ndjson", _posterior_lines(chains=(2, 4)))
    return tmp_path


def test_build_checked_dense_fit_materializes_invariant_before_arviz(tmp_path: Path) -> None:
    run_root = _minimal_run(tmp_path)
    posterior = read_posterior_stream(run_root / "posterior.ndjson")

    dense = build_checked_dense_fit(
        model_ir=run_root / "model.ir.json",
        data_json=run_root / "data.json",
        posterior=posterior,
    )

    assert isinstance(dense, CheckedDenseFit)
    posterior_group = dense.require_group("posterior")
    beta = posterior_group.require_var("beta")
    assert beta.dims == ("chain", "draw", "beta_dim_0")
    assert posterior_group.coords["chain"] == (2, 4)
    assert dense.require_group("observed_data").var_names == ("y",)
    assert dense.require_group("constant_data").var_names == ("x",)


def test_dense_validation_rejects_predictive_shape_mismatch(tmp_path: Path) -> None:
    run_root = _minimal_run(tmp_path)
    _write_json(
        run_root / "data.json",
        {
            "y": {"dtype": "int64", "shape": [3], "values": [3, 4, 5]},
        },
    )
    _write_ndjson(
        run_root / "posterior_predictive.ndjson", _posterior_predictive_lines(chains=(2, 4))
    )
    posterior = read_posterior_stream(run_root / "posterior.ndjson")
    posterior_predictive = read_posterior_predictive_stream(
        run_root / "posterior_predictive.ndjson"
    )

    with pytest.raises(ValueError, match="does not match observed_data shape"):
        build_checked_dense_fit(
            model_ir=run_root / "model.ir.json",
            data_json=run_root / "data.json",
            posterior=posterior,
            posterior_predictive=posterior_predictive,
        )


def test_dense_validation_rejects_predictive_without_matching_observed_data(
    tmp_path: Path,
) -> None:
    run_root = _minimal_run(tmp_path)
    _write_ndjson(
        run_root / "posterior_predictive.ndjson", _posterior_predictive_lines(chains=(2, 4))
    )
    posterior = read_posterior_stream(run_root / "posterior.ndjson")
    posterior_predictive = read_posterior_predictive_stream(
        run_root / "posterior_predictive.ndjson"
    )
    dense = build_checked_dense_fit(
        model_ir=run_root / "model.ir.json",
        data_json=run_root / "data.json",
        posterior=posterior,
        posterior_predictive=posterior_predictive,
    )
    broken = dense.without_group("observed_data")
    with pytest.raises(ValueError, match="observed_data"):
        validate_dense_fit(broken)


def test_dense_fit_allows_zero_length_data_axes(tmp_path: Path) -> None:
    run_root = _minimal_run(tmp_path)
    _write_json(
        run_root / "data.json",
        {
            "y": {"dtype": "float64", "shape": [0], "values": []},
        },
    )
    posterior = read_posterior_stream(run_root / "posterior.ndjson")

    dense = build_checked_dense_fit(
        model_ir=run_root / "model.ir.json",
        data_json=run_root / "data.json",
        posterior=posterior,
    )

    observed = dense.require_group("observed_data").require_var("y")
    assert observed.values.shape == (0,)
    assert dense.require_group("observed_data").coords["y_dim_0"] == ()


def test_dense_to_datatree_is_the_only_arviz_boundary(tmp_path: Path) -> None:
    run_root = _minimal_run(tmp_path)
    posterior = read_posterior_stream(run_root / "posterior.ndjson")
    dense = build_checked_dense_fit(
        model_ir=run_root / "model.ir.json",
        data_json=run_root / "data.json",
        posterior=posterior,
    )

    dt = dense_to_datatree(dense)

    assert isinstance(dt, xr.DataTree)
    assert dt["/posterior"].ds["chain"].values.tolist() == [2, 4]
    assert set(dt["/constant_data"].ds.data_vars) == {"x"}


def test_assemble_module_no_longer_imports_arviz_base_or_json() -> None:
    assert not hasattr(assemble_module, "azb")
    assert not hasattr(assemble_module, "json")
