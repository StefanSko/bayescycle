"""Bayescycle run-dir to ArviZ fit exporter.

This package is intentionally separate from ``bayesite_viz``. It owns
conversion; the plotting package remains fit-file-in/image-out.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from bayesite_idata.assemble import assemble_datatree
from bayesite_idata.protocol_v0 import (
    read_posterior_predictive_stream,
    read_posterior_stream,
    read_prior_predictive_stream,
)
from bayesite_idata.run_dir import discover_run_dir
from bayesite_idata.validate import ValidateMode, run_diagnose
from bayesite_idata.writer import write_netcdf

__all__ = ["from_run_dir", "write_netcdf"]


def from_run_dir(
    path: str | Path, *, validate: ValidateMode = "require", bayesite: str | Path | None = None
) -> xr.DataTree:
    """Convert a bayescycle run directory into an ArviZ DataTree."""
    run = discover_run_dir(Path(path))
    bayesite_path = Path(bayesite) if bayesite is not None else None
    run_diagnose(run.posterior, validate, bayesite=bayesite_path)
    posterior = read_posterior_stream(run.posterior)
    prior_predictive = (
        read_prior_predictive_stream(run.prior_predictive)
        if run.prior_predictive is not None
        else None
    )
    posterior_predictive = (
        read_posterior_predictive_stream(run.posterior_predictive)
        if run.posterior_predictive is not None
        else None
    )
    return assemble_datatree(
        run,
        posterior,
        prior_predictive=prior_predictive,
        posterior_predictive=posterior_predictive,
    )
