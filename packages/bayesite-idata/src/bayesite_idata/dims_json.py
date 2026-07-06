"""Optional Bayescycle dims.json sidecar decoding."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from bayesite_idata.errors import AssemblyError

type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
type CoordValue = str | int | float | bool

DIMS_FORMAT = "bayescycle-dims-v1"
RESERVED_SAMPLE_DIMS = frozenset(("chain", "draw"))


@dataclass(frozen=True)
class DimsSidecar:
    """Declared variable event dims and optional coordinate values."""

    dims: Mapping[str, tuple[str, ...]]
    coords: Mapping[str, tuple[CoordValue, ...]]

    def dims_for(self, variable: str) -> tuple[str, ...] | None:
        """Return declared event dims for ``variable`` when present."""
        return self.dims.get(variable)

    def coords_for(self, dim: str) -> tuple[CoordValue, ...] | None:
        """Return declared coordinate values for ``dim`` when present."""
        return self.coords.get(dim)


def read_dims_sidecar(path: Path) -> DimsSidecar:
    """Read a Bayescycle dims sidecar into typed metadata."""
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = cast(JsonValue, json.load(f))
    except json.JSONDecodeError as exc:
        msg = f"{path}: invalid dims.json: {exc.msg}"
        raise AssemblyError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"{path}: dims.json must be an object"
        raise AssemblyError(msg)
    if raw.get("dims_format") != DIMS_FORMAT:
        msg = f"{path}: dims_format must be {DIMS_FORMAT!r}"
        raise AssemblyError(msg)
    dims = _read_dims(raw.get("dims"), path)
    coords = _read_coords(raw.get("coords"), path)
    return DimsSidecar(dims=dims, coords=coords)


def _read_dims(value: JsonValue | None, path: Path) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        msg = f"{path}: dims must be an object keyed by variable name"
        raise AssemblyError(msg)
    out: dict[str, tuple[str, ...]] = {}
    for variable, raw_dims in value.items():
        if not isinstance(raw_dims, list):
            msg = f"{path}: dims.{variable} must be an array of dimension names"
            raise AssemblyError(msg)
        dims: list[str] = []
        seen: set[str] = set()
        for index, raw_dim in enumerate(raw_dims):
            if not isinstance(raw_dim, str) or raw_dim == "":
                msg = f"{path}: dims.{variable}[{index}] must be a non-empty string"
                raise AssemblyError(msg)
            if raw_dim in RESERVED_SAMPLE_DIMS:
                msg = f"{path}: dims.{variable}[{index}] uses reserved sample dimension {raw_dim!r}"
                raise AssemblyError(msg)
            if raw_dim in seen:
                msg = f"{path}: dims.{variable} repeats dimension {raw_dim!r}"
                raise AssemblyError(msg)
            seen.add(raw_dim)
            dims.append(raw_dim)
        out[variable] = tuple(dims)
    return MappingProxyType(out)


def _read_coords(value: JsonValue | None, path: Path) -> Mapping[str, tuple[CoordValue, ...]]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, dict):
        msg = f"{path}: coords must be an object keyed by dimension name"
        raise AssemblyError(msg)
    out: dict[str, tuple[CoordValue, ...]] = {}
    for dim, raw_coords in value.items():
        if dim in RESERVED_SAMPLE_DIMS:
            msg = f"{path}: coords.{dim} conflicts with reserved sample dimension {dim!r}"
            raise AssemblyError(msg)
        if not isinstance(raw_coords, list):
            msg = f"{path}: coords.{dim} must be an array"
            raise AssemblyError(msg)
        out[dim] = tuple(
            _coord_value(item, f"{path}: coords.{dim}[{index}]")
            for index, item in enumerate(raw_coords)
        )
    return MappingProxyType(out)


def _coord_value(value: JsonValue, field: str) -> CoordValue:
    if isinstance(value, bool | int | float | str):
        return value
    msg = f"{field} must be a JSON string, number, or boolean"
    raise AssemblyError(msg)
