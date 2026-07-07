"""Fit loading. Explicit file, no discovery, no stdin, no conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import xarray as xr


def _looks_like_raw_engine_output(path: Path) -> bool:
    """Heuristic: text whose first non-whitespace char is ``{`` or ``[``.

    ArviZ NetCDF is binary HDF5, so it never matches. Raw bayesite engine output
    is NDJSON/JSON (text starting with ``{`` or ``[``). Used only to give a
    repair-oriented hint, never to accept the format.
    """
    try:
        with path.open("rb") as f:
            head = f.read(512)
    except OSError:
        return False
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    stripped = text.lstrip()
    return bool(stripped) and stripped[0] in "{["


def load_fit(path: Path) -> xr.DataTree:
    """Load an ArviZ fit NetCDF as an xarray DataTree.

    The argument must be a file. A directory is an explicit error with a
    repair-oriented message; this is not discovery, it is a good error. Raw
    engine output (NDJSON/JSON) is refused with a pointer to
    ``bayesite-idata``; this tool does no format conversion.
    """
    if path.is_dir():
        sys.stderr.write(
            f"{path} is a directory; pass the InferenceData file (e.g. {path}/posterior.nc)\n"
        )
        raise SystemExit(2)
    if not path.exists():
        sys.stderr.write(f"fit file not found: {path}\n")
        raise SystemExit(2)
    try:
        return xr.open_datatree(path)
    except Exception as e:  # noqa: BLE001 - surface as a repair-oriented message
        if _looks_like_raw_engine_output(path):
            sys.stderr.write(
                f"{path} is not an InferenceData file (looks like raw engine "
                f"output: NDJSON/JSON). Run `bayesite-idata run/ -o fit.nc` "
                f"from the separate `bayesite-idata` package first to build "
                f"a NetCDF InferenceData, then pass that path.\n"
            )
            raise SystemExit(5) from e
        sys.stderr.write(f"failed to load InferenceData from {path}: {e}\n")
        raise SystemExit(2) from e


def require_groups(dt: xr.DataTree, *groups: str) -> None:
    """Exit non-zero with a repair message if any group is missing."""
    present = {g.lstrip("/") for g in dt.groups}
    missing = [g for g in groups if g not in present]
    if missing:
        sys.stderr.write(
            f"missing required InferenceData group(s): {', '.join(missing)}; "
            f"present groups: {', '.join(sorted(present)) or '(none)'}\n"
        )
        raise SystemExit(3)


def require_var(dt: xr.DataTree, group: str, var: str) -> None:
    """Exit non-zero with a repair message if a variable is missing from a group."""
    node = dt[group] if group.startswith("/") else dt[f"/{group}"]
    present = set(node.data_vars)
    if var not in present:
        sys.stderr.write(
            f"missing required variable {var!r} in group {group!r}; "
            f"present variables: {', '.join(sorted(present)) or '(none)'}\n"
        )
        raise SystemExit(3)
