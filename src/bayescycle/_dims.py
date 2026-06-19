"""Run-directory dimension-label sidecar support."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from jaxstanv5 import dimension_metadata_to_dict, model_dimensions
from jaxstanv5.model import ModelMeta

DIMS_FORMAT = "bayescycle-dims-v1"

type JsonScalar = None | bool | int | float | str


class DimsJsonDocument(TypedDict):
    """JSON document written to ``run/dims.json``."""

    dims_format: str
    dims: dict[str, list[str]]
    coords: dict[str, list[JsonScalar]]


@dataclass(frozen=True)
class DimsSidecar:
    """Dimension labels and optional coordinates for a run directory."""

    dims: dict[str, tuple[str, ...]]
    coords: dict[str, tuple[JsonScalar, ...]]

    def is_empty(self) -> bool:
        """Return whether there is no metadata worth writing."""
        return not self.dims and not self.coords

    def to_json_document(self) -> DimsJsonDocument:
        """Return a JSON-ready sidecar document."""
        return {
            "dims_format": DIMS_FORMAT,
            "dims": {name: list(dims) for name, dims in self.dims.items()},
            "coords": {name: list(coords) for name, coords in self.coords.items()},
        }


def dims_sidecar_for_model(model_cls: type[object], meta: ModelMeta) -> DimsSidecar | None:
    """Build the optional dims sidecar from jaxstanv5 authoring metadata."""
    metadata = dimension_metadata_to_dict(model_dimensions(model_cls))
    sidecar = DimsSidecar(
        dims={name: tuple(dims) for name, dims in metadata["dims"].items()},
        coords={name: tuple(coords) for name, coords in metadata["coords"].items()},
    )
    _validate_sidecar(sidecar)
    _validate_known_ranks(sidecar, meta)
    if sidecar.is_empty():
        return None
    return sidecar


def write_dims_sidecar(path: Path, sidecar: DimsSidecar) -> None:
    """Write ``sidecar`` to ``path`` as deterministic JSON."""
    path.write_text(
        json.dumps(sidecar.to_json_document(), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_sidecar(sidecar: DimsSidecar) -> None:
    for variable, dims in sidecar.dims.items():
        if not isinstance(variable, str) or variable == "":
            raise ValueError("dimension metadata variable names must be non-empty strings")
        for dim in dims:
            if not isinstance(dim, str) or dim == "":
                raise ValueError("dimension labels must be non-empty strings")

    for dim, coords in sidecar.coords.items():
        if not isinstance(dim, str) or dim == "":
            raise ValueError("coordinate dimension names must be non-empty strings")
        for coord in coords:
            if not _is_json_scalar(coord):
                raise ValueError("coordinate values must be JSON scalar values")

    # Keep this as a final guard even though jaxstanv5 normalizes coordinates.
    json.dumps(sidecar.to_json_document(), allow_nan=False)


def _validate_known_ranks(sidecar: DimsSidecar, meta: ModelMeta) -> None:
    known_ranks = _known_variable_ranks(meta)
    for variable, dims in sidecar.dims.items():
        rank = known_ranks.get(variable)
        if rank is not None and len(dims) != rank:
            raise ValueError(
                f"dimension metadata for {variable!r} has rank {len(dims)}, expected {rank}"
            )


def _known_variable_ranks(meta: ModelMeta) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for name, data in meta.data.items():
        rank = _data_schema_rank(data.schema)
        if rank is not None:
            ranks[name] = rank
    for name, free_value in meta.free_values.items():
        ranks[name] = 0 if free_value.size is None else 1
    return ranks


def _data_schema_rank(schema: object) -> int | None:
    dims = getattr(schema, "dims", None)
    if isinstance(dims, tuple):
        return len(dims)
    rank = getattr(schema, "rank", None)
    if isinstance(rank, int):
        return rank
    return None


def _is_json_scalar(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None or isinstance(value, bool | int | str)
