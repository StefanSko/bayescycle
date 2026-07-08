from __future__ import annotations

from pathlib import Path

import arviz_base as azb
import numpy as np
import pytest
import xarray as xr


def _make_continuous() -> xr.DataTree:
    rng = np.random.default_rng(0)
    n_chain, n_draw = 4, 200
    posterior = {
        "mu": rng.standard_normal((n_chain, n_draw)),
        "tau": np.abs(rng.standard_normal((n_chain, n_draw))),
    }
    sample_stats = {
        "energy": rng.standard_normal((n_chain, n_draw)),
        "diverging": rng.random((n_chain, n_draw)) < 0.05,
    }
    obs = rng.standard_normal(8)
    posterior_predictive = {"obs": rng.standard_normal((n_chain, n_draw, 8))}
    observed_data = {"obs": obs}
    return azb.from_dict(
        {
            "posterior": posterior,
            "sample_stats": sample_stats,
            "posterior_predictive": posterior_predictive,
            "observed_data": observed_data,
        }
    )


def _make_discrete() -> xr.DataTree:
    rng = np.random.default_rng(1)
    n_chain, n_draw = 4, 200
    posterior = {"theta": rng.random((n_chain, n_draw))}
    # poisson counts -> discrete, suitable for rootogram
    lam = 3.0
    pp = rng.poisson(lam, size=(n_chain, n_draw, 10)).astype("int64")
    obs = rng.poisson(lam, size=10).astype("int64")
    return azb.from_dict(
        {
            "posterior": posterior,
            "posterior_predictive": {"obs": pp},
            "observed_data": {"obs": obs},
        }
    )


@pytest.fixture(scope="session")
def continuous_fit(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dt = _make_continuous()
    p = tmp_path_factory.mktemp("fit") / "continuous.nc"
    dt.to_netcdf(p)
    return p


@pytest.fixture(scope="session")
def discrete_fit(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dt = _make_discrete()
    p = tmp_path_factory.mktemp("fit") / "discrete.nc"
    dt.to_netcdf(p)
    return p
