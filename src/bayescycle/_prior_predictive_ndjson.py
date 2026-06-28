"""Prior-predictive artifact writer for the bayescycle run-directory v0 contract."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from bayescycle._settings import ResolvedPriorPredictiveSettings


class PriorPredictiveArtifactError(RuntimeError):
    """Raised when prior-predictive results cannot be serialized."""


class _ArrayLike(Protocol):
    """Small array surface used by the artifact writer."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    def __getitem__(self, index: int) -> _ArrayLike: ...

    def reshape(self, *shape: int) -> _ArrayLike: ...

    def tolist(self) -> object: ...


class _PriorPredictiveResult(Protocol):
    parameters: Mapping[str, _ArrayLike]
    observed: Mapping[str, _ArrayLike]
    data: Mapping[str, _ArrayLike]


@dataclass(frozen=True)
class _SiteSpec:
    name: str
    role: str
    shape: tuple[int, ...]
    integer: bool


JsonValue = object

WORKFLOW_PHASES = [
    "parse_json",
    "load_model",
    "bind_declared_data",
    "simulate_prior_predictive",
    "emit_artifact",
]


def write_prior_predictive_ndjson(
    path: Path,
    *,
    result: _PriorPredictiveResult,
    settings: ResolvedPriorPredictiveSettings,
) -> None:
    """Write a v0 prior-predictive draw stream for an in-process result."""
    sites = _site_specs(result)
    if not sites:
        raise PriorPredictiveArtifactError(
            "jaxstanv5 backend cannot write prior_predictive.ndjson without generated sites"
        )
    _validate_site_arrays(result, sites, settings)
    documents = _documents(result, sites, settings)
    with path.open("x", encoding="utf-8") as f:
        for document in documents:
            f.write(json.dumps(document, separators=(",", ":"), allow_nan=False))
            f.write("\n")


def _site_specs(result: _PriorPredictiveResult) -> tuple[_SiteSpec, ...]:
    specs: list[_SiteSpec] = []
    for name, array in result.parameters.items():
        specs.append(
            _SiteSpec(
                name=name,
                role="parameter",
                shape=_draw_shape(array),
                integer=_is_integer_array(array),
            )
        )
    for name, array in result.observed.items():
        specs.append(
            _SiteSpec(
                name=name,
                role="observed",
                shape=_draw_shape(array),
                integer=_is_integer_array(array),
            )
        )
    return tuple(specs)


def _validate_site_arrays(
    result: _PriorPredictiveResult,
    sites: tuple[_SiteSpec, ...],
    settings: ResolvedPriorPredictiveSettings,
) -> None:
    for site in sites:
        array = _site_array(result, site)
        if len(array.shape) < 1:
            raise PriorPredictiveArtifactError(
                f"prior-predictive site {site.name!r} is missing its draw axis"
            )
        if int(array.shape[0]) != settings.draws:
            raise PriorPredictiveArtifactError(
                f"prior-predictive site {site.name!r} has {array.shape[0]} draw(s), "
                f"expected {settings.draws}"
            )
        if _draw_shape(array) != site.shape:
            raise PriorPredictiveArtifactError(
                f"prior-predictive site {site.name!r} changed shape during serialization"
            )


def _documents(
    result: _PriorPredictiveResult,
    sites: tuple[_SiteSpec, ...],
    settings: ResolvedPriorPredictiveSettings,
) -> list[JsonValue]:
    documents: list[JsonValue] = [_header(result, sites, settings)]
    site_order = [site.name for site in sites]
    for draw in range(settings.draws):
        documents.append(
            {
                **_artifact_fields(),
                "draw_index": draw,
                "draw_index_base": "zero_based_prior_predictive_draw_order",
                "seed": settings.seed,
                "draw": draw,
                "draw_count": settings.draws,
                "declared_data_count": len(result.data),
                "declared_data_order": list(result.data),
                "site_count": len(sites),
                "site_order": site_order,
                "values": {
                    site.name: _value_at(_site_array(result, site), draw, site.shape)
                    for site in sites
                },
            }
        )
    documents.append({"trailer": _trailer(result, sites, settings)})
    return documents


def _header(
    result: _PriorPredictiveResult,
    sites: tuple[_SiteSpec, ...],
    settings: ResolvedPriorPredictiveSettings,
) -> dict[str, JsonValue]:
    declared_data = tuple(result.data.items())
    return {
        **_artifact_fields(),
        "workflow_phases": list(WORKFLOW_PHASES),
        "draws": settings.draws,
        "draw_count": settings.draws,
        "draw_index_base": "zero_based_prior_predictive_draw_order",
        "seed": settings.seed,
        "settings": {"num_draws": settings.draws},
        "site_count": len(sites),
        "site_order": [site.name for site in sites],
        "declared_data_count": len(declared_data),
        "declared_data_order": [name for name, _ in declared_data],
        "declared_data": {
            name: _array_value(array, _array_shape(array)) for name, array in declared_data
        },
        "declared_data_shapes": {name: list(_array_shape(array)) for name, array in declared_data},
        "declared_data_coordinate_order": {
            name: _coordinate_order(_array_shape(array)) for name, array in declared_data
        },
        "declared_data_integer": {name: _is_integer_array(array) for name, array in declared_data},
        "declared_data_integer_by_coordinate": {
            name: _integer_by_coordinate(array) for name, array in declared_data
        },
        "sites": [_site_document(site) for site in sites],
    }


def _trailer(
    result: _PriorPredictiveResult,
    sites: tuple[_SiteSpec, ...],
    settings: ResolvedPriorPredictiveSettings,
) -> dict[str, JsonValue]:
    return {
        **_artifact_fields(),
        "workflow_phases": list(WORKFLOW_PHASES),
        "draws": settings.draws,
        "draw_count": settings.draws,
        "draw_index_base": "zero_based_prior_predictive_draw_order",
        "seed": settings.seed,
        "settings": {"num_draws": settings.draws},
        "site_count": len(sites),
        "site_order": [site.name for site in sites],
        "declared_data_count": len(result.data),
        "declared_data_order": list(result.data),
        "sites": len(sites),
    }


def _artifact_fields() -> dict[str, str]:
    return {
        "prior_predictive_format": "v0-provisional",
        "artifact_kind": "prior_predictive_draws",
        "artifact_scope": "declared_data_conditioned_site_draws",
    }


def _site_document(site: _SiteSpec) -> dict[str, JsonValue]:
    return {
        "name": site.name,
        "stochastic_site": site.name,
        "role": site.role,
        "shape": list(site.shape),
        "integer": site.integer,
        "integer_by_coordinate": _integer_flags(site.shape, site.integer),
        "coordinate_order": _coordinate_order(site.shape),
    }


def _site_array(result: _PriorPredictiveResult, site: _SiteSpec) -> _ArrayLike:
    if site.role == "parameter":
        return result.parameters[site.name]
    return result.observed[site.name]


def _draw_shape(array: _ArrayLike) -> tuple[int, ...]:
    shape = _array_shape(array)
    if not shape:
        return ()
    return shape[1:]


def _array_shape(array: _ArrayLike) -> tuple[int, ...]:
    return tuple(int(dim) for dim in array.shape)


def _value_at(array: _ArrayLike, draw: int, shape: tuple[int, ...]) -> JsonValue:
    return _array_value(array[draw], shape)


def _array_value(array: _ArrayLike, shape: tuple[int, ...]) -> JsonValue:
    if shape == ():
        return _scalar_value(array.tolist())
    raw = array.reshape(-1).tolist()
    if not isinstance(raw, list):
        raise PriorPredictiveArtifactError("non-scalar array did not flatten to a JSON array")
    return [_scalar_value(value) for value in raw]


def _scalar_value(value: object) -> int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PriorPredictiveArtifactError("artifact values must be finite")
        return value
    raise PriorPredictiveArtifactError(f"artifact value is not numeric: {value!r}")


def _is_integer_array(array: _ArrayLike) -> bool:
    dtype = getattr(array, "dtype", None)
    kind = getattr(dtype, "kind", None)
    return kind in {"b", "i", "u"}


def _integer_by_coordinate(array: _ArrayLike) -> JsonValue:
    shape = _array_shape(array)
    integer = _is_integer_array(array)
    return _integer_flags(shape, integer)


def _integer_flags(shape: tuple[int, ...], integer: bool) -> JsonValue:
    size = _size(shape)
    if size == 1 and shape == ():
        return integer
    return [integer for _ in range(size)]


def _coordinate_order(shape: tuple[int, ...]) -> list[list[int]]:
    if not shape:
        return []
    ranges = [range(dim) for dim in shape]
    coordinates: list[list[int]] = [[]]
    for values in ranges:
        coordinates = [prefix + [value] for prefix in coordinates for value in values]
    return coordinates


def _size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size
