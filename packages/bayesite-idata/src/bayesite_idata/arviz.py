"""ArViZ adapter for the checked dense fit."""

from __future__ import annotations

import arviz_base as azb
import xarray as xr

from bayesite_idata.dense import CheckedDenseFit, DenseGroup


def dense_to_datatree(fit: CheckedDenseFit) -> xr.DataTree:
    """Convert a checked dense fit into an xarray DataTree via arviz-base."""
    dt: xr.DataTree | None = None
    for group in fit.groups:
        group_dt = _group_to_datatree(group)
        if dt is None:
            dt = group_dt
        else:
            dt[f"/{group.name}"] = group_dt[f"/{group.name}"]
    if dt is None:
        msg = "checked dense fit has no groups"
        raise ValueError(msg)
    if "/posterior" in dt.groups:
        for key, value in fit.posterior_attrs.items():
            dt["/posterior"].attrs[key] = value
    return dt


def _group_to_datatree(group: DenseGroup) -> xr.DataTree:
    data = {var.name: var.values for var in group.variables}
    dims = {var.name: _extra_dims(var.dims, group.sample_dims) for var in group.variables}
    dims = {name: value for name, value in dims.items() if value}
    coords = {name: list(values) for name, values in group.coords.items()}
    return azb.from_dict(
        {group.name: data},
        coords=coords,
        dims=dims,
        sample_dims=("chain", "draw"),
    )


def _extra_dims(dims: tuple[str, ...], sample_dims: tuple[str, ...]) -> list[str]:
    if sample_dims and dims[: len(sample_dims)] == sample_dims:
        return list(dims[len(sample_dims) :])
    return list(dims)
