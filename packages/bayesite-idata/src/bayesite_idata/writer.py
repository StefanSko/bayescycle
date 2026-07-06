"""Persistence for exported ArviZ fits."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


class WriteError(ValueError):
    """The requested output path/format is unsupported."""


def write_netcdf(dt: xr.DataTree, path: Path) -> Path:
    """Write a DataTree as ArviZ NetCDF and return the absolute path."""
    if path.suffix.lower() not in {".nc", ".idata"}:
        msg = f"NetCDF output path must end in .nc or .idata, got {path}"
        raise WriteError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    dt.to_netcdf(path)
    return path.resolve()
