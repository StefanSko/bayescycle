"""Orchestrate run artifact assembly through the dense-fit invariant."""

from __future__ import annotations

import xarray as xr

from bayesite_idata.arviz import dense_to_datatree
from bayesite_idata.dense import build_checked_dense_fit
from bayesite_idata.dims_json import read_dims_sidecar
from bayesite_idata.errors import AssemblyError
from bayesite_idata.protocol_v0 import (
    PosteriorPredictiveStream,
    PosteriorStream,
    PriorPredictiveStream,
)
from bayesite_idata.run_dir import RunDir

__all__ = ["AssemblyError", "assemble_datatree"]


def assemble_datatree(
    run: RunDir,
    posterior: PosteriorStream,
    *,
    prior_predictive: PriorPredictiveStream | None = None,
    posterior_predictive: PosteriorPredictiveStream | None = None,
) -> xr.DataTree:
    """Build an ArviZ DataTree from parsed run-dir artifacts."""
    dims_sidecar = read_dims_sidecar(run.dims_json) if run.dims_json is not None else None
    dense = build_checked_dense_fit(
        model_ir=run.model_ir,
        data_json=run.data_json,
        posterior=posterior,
        prior_predictive=prior_predictive,
        posterior_predictive=posterior_predictive,
        dims_sidecar=dims_sidecar,
    )
    return dense_to_datatree(dense)
