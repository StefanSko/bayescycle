"""Lenient readers for bayescycle v0-provisional NDJSON streams.

These readers assemble arrays for ArviZ. They are not a replacement for
``bayesite diagnose`` and intentionally check only cheap shape/order invariants
needed to avoid silently wrong plots.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type AttrValue = str | int | float | bool
type NumberValue = int | float

V0_MARKER = "v0-provisional"
PER_DRAW_SAMPLE_STATS_V1 = "per_draw_v1"
PER_DRAW_SAMPLE_STATS_V2 = "per_draw_v2"
SUPPORTED_PER_DRAW_SAMPLE_STATS = frozenset({PER_DRAW_SAMPLE_STATS_V1, PER_DRAW_SAMPLE_STATS_V2})


class ProtocolError(ValueError):
    """The v0-provisional stream cannot be assembled safely."""


@dataclass(frozen=True)
class ParamSpec:
    """One posterior parameter's constrained shape."""

    name: str
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        out = 1
        for dim in self.shape:
            out *= dim
        return out


@dataclass(frozen=True)
class DrawRecord:
    """One retained posterior draw."""

    chain: int
    draw: int
    values: Mapping[str, tuple[NumberValue, ...]]
    diverging: bool | None
    tree_depth: int | None
    tree_accept: float | None
    energy: float | None


@dataclass(frozen=True)
class PosteriorStream:
    """Parsed posterior stream in chain/draw form."""

    params: tuple[ParamSpec, ...]
    chains: tuple[int, ...]
    draws_per_chain: int
    sample_stats_mode: str | None
    records: tuple[DrawRecord, ...]
    attrs: Mapping[str, AttrValue]


@dataclass(frozen=True)
class SiteSpec:
    """One generated-site shape in a predictive stream."""

    name: str
    shape: tuple[int, ...]
    role: str | None
    integer: bool

    @property
    def size(self) -> int:
        out = 1
        for dim in self.shape:
            out *= dim
        return out


@dataclass(frozen=True)
class PriorPredictiveRecord:
    """One prior-predictive generated draw."""

    draw: int
    values: Mapping[str, tuple[NumberValue, ...]]


@dataclass(frozen=True)
class PriorPredictiveStream:
    """Parsed prior-predictive stream."""

    sites: tuple[SiteSpec, ...]
    draws: int
    records: tuple[PriorPredictiveRecord, ...]


@dataclass(frozen=True)
class PosteriorPredictiveRecord:
    """One posterior-predictive generated draw with source posterior provenance."""

    source_chain: int
    source_draw: int
    values: Mapping[str, tuple[NumberValue, ...]]


@dataclass(frozen=True)
class PosteriorPredictiveStream:
    """Parsed posterior-predictive stream."""

    sites: tuple[SiteSpec, ...]
    records: tuple[PosteriorPredictiveRecord, ...]
    source_fit_seed: int | None


def _load_lines(path: Path) -> list[Mapping[str, JsonValue]]:
    docs: list[Mapping[str, JsonValue]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = cast(JsonValue, json.loads(stripped))
            except json.JSONDecodeError as exc:
                msg = f"{path}:{line_no}: invalid JSON: {exc.msg}"
                raise ProtocolError(msg) from exc
            if not isinstance(value, dict):
                msg = f"{path}:{line_no}: expected a JSON object"
                raise ProtocolError(msg)
            docs.append(value)
    if len(docs) < 2:
        msg = f"{path}: expected header, at least one draw, and trailer"
        raise ProtocolError(msg)
    return docs


def _obj(value: JsonValue | None, field: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        msg = f"expected object field {field!r}"
        raise ProtocolError(msg)
    return value


def _array(value: JsonValue | None, field: str) -> Sequence[JsonValue]:
    if not isinstance(value, list):
        msg = f"expected array field {field!r}"
        raise ProtocolError(msg)
    return value


def _str(value: JsonValue | None, field: str) -> str:
    if not isinstance(value, str):
        msg = f"expected string field {field!r}"
        raise ProtocolError(msg)
    return value


def _int(value: JsonValue | None, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"expected integer field {field!r}"
        raise ProtocolError(msg)
    return value


def _bool(value: JsonValue | None, field: str) -> bool:
    if not isinstance(value, bool):
        msg = f"expected boolean field {field!r}"
        raise ProtocolError(msg)
    return value


def _optional_int(value: JsonValue | None, field: str) -> int | None:
    if value is None:
        return None
    return _int(value, field)


def _numeric(value: JsonValue | None, field: str) -> NumberValue:
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = f"expected numeric field {field!r}"
        raise ProtocolError(msg)
    return value


def _number(value: JsonValue | None, field: str) -> float:
    return float(_numeric(value, field))


def _optional_attr(value: JsonValue | None) -> AttrValue | None:
    if isinstance(value, bool | int | float | str):
        return value
    return None


def _shape(value: JsonValue | None, field: str) -> tuple[int, ...]:
    raw = _array(value, field)
    shape: list[int] = []
    for i, item in enumerate(raw):
        dim = _int(item, f"{field}[{i}]")
        if dim < 0:
            msg = f"shape field {field!r} has negative dimension {dim}"
            raise ProtocolError(msg)
        shape.append(dim)
    return tuple(shape)


def _flat_values(
    value: JsonValue | None, spec: ParamSpec | SiteSpec, field: str
) -> tuple[NumberValue, ...]:
    if spec.shape == ():
        if isinstance(value, list):
            if len(value) != 1:
                msg = f"{field} for scalar {spec.name!r} has shape mismatch"
                raise ProtocolError(msg)
            return (_numeric(value[0], field),)
        return (_numeric(value, field),)
    raw = _array(value, field)
    if len(raw) != spec.size:
        msg = (
            f"{field} for {spec.name!r} has shape mismatch: "
            f"got {len(raw)} value(s), expected {spec.size} for shape {spec.shape}"
        )
        raise ProtocolError(msg)
    return tuple(_numeric(item, f"{field}[{i}]") for i, item in enumerate(raw))


def _params(header: Mapping[str, JsonValue]) -> tuple[ParamSpec, ...]:
    raw_params = _array(header.get("params"), "params")
    params: list[ParamSpec] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_params):
        obj = _obj(raw, f"params[{i}]")
        name = _str(obj.get("name"), f"params[{i}].name")
        if name in seen:
            msg = f"duplicate parameter {name!r} in header params"
            raise ProtocolError(msg)
        seen.add(name)
        params.append(ParamSpec(name=name, shape=_shape(obj.get("shape"), f"params[{i}].shape")))
    return tuple(params)


def _sites(header: Mapping[str, JsonValue], marker_field: str) -> tuple[SiteSpec, ...]:
    if header.get(marker_field) != V0_MARKER:
        msg = f"stream header needs {marker_field} {V0_MARKER!r}"
        raise ProtocolError(msg)
    raw_sites = _array(header.get("sites"), "sites")
    sites: list[SiteSpec] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_sites):
        obj = _obj(raw, f"sites[{i}]")
        name = _str(obj.get("name"), f"sites[{i}].name")
        if name in seen:
            msg = f"duplicate generated site {name!r}"
            raise ProtocolError(msg)
        seen.add(name)
        role_value = obj.get("role")
        role = _str(role_value, f"sites[{i}].role") if role_value is not None else None
        integer_value = obj.get("integer")
        integer = (
            _bool(integer_value, f"sites[{i}].integer") if integer_value is not None else False
        )
        sites.append(
            SiteSpec(
                name=name,
                shape=_shape(obj.get("shape"), f"sites[{i}].shape"),
                role=role,
                integer=integer,
            )
        )
    return tuple(sites)


def _chain_order(header: Mapping[str, JsonValue]) -> tuple[int, ...]:
    raw = header.get("chain_order")
    if raw is None:
        count = _int(header.get("chain_count"), "chain_count")
        return tuple(range(count))
    chains = tuple(_int(item, "chain_order[]") for item in _array(raw, "chain_order"))
    if len(set(chains)) != len(chains):
        msg = f"duplicate chain_order label(s): {chains}"
        raise ProtocolError(msg)
    return chains


def _posterior_attrs(header: Mapping[str, JsonValue]) -> Mapping[str, AttrValue]:
    out: dict[str, AttrValue] = {}
    for key in ("posterior_identity_hash", "seed", "draw_count", "chain_count"):
        value = _optional_attr(header.get(key))
        if value is not None:
            out[key] = value
    settings = header.get("settings")
    if isinstance(settings, dict):
        for key, value in settings.items():
            attr_value = _optional_attr(value)
            if attr_value is not None:
                out[f"settings.{key}"] = attr_value
    return MappingProxyType(out)


def read_posterior_stream(path: Path) -> PosteriorStream:
    """Read enough v0 posterior NDJSON to assemble ArviZ arrays."""
    docs = _load_lines(path)
    header = docs[0]
    if header.get("draws_format") != V0_MARKER:
        msg = f"fit header needs draws_format {V0_MARKER!r}; got {header.get('draws_format')!r}"
        raise ProtocolError(msg)
    params = _params(header)
    params_by_name = {param.name: param for param in params}
    chains = _chain_order(header)
    sample_stats_mode_value = header.get("sample_stats_mode")
    sample_stats_mode = (
        _str(sample_stats_mode_value, "sample_stats_mode")
        if sample_stats_mode_value is not None
        else None
    )
    if sample_stats_mode is not None and sample_stats_mode not in SUPPORTED_PER_DRAW_SAMPLE_STATS:
        msg = (
            f"{path}: sample_stats_mode must be "
            f"{PER_DRAW_SAMPLE_STATS_V1!r} or {PER_DRAW_SAMPLE_STATS_V2!r}"
        )
        raise ProtocolError(msg)

    trailer_doc = docs[-1]
    if "trailer" not in trailer_doc:
        msg = f"{path}: final line must be a trailer object"
        raise ProtocolError(msg)
    trailer = _obj(trailer_doc.get("trailer"), "trailer")

    records: list[DrawRecord] = []
    seen: set[tuple[int, int]] = set()
    for line_index, doc in enumerate(docs[1:-1], start=2):
        if doc.get("draws_format") != V0_MARKER:
            msg = f"{path}:{line_index}: draw line needs draws_format {V0_MARKER!r}"
            raise ProtocolError(msg)
        chain = _int(doc.get("chain"), "chain")
        draw = _int(doc.get("draw"), "draw")
        key = (chain, draw)
        if key in seen:
            msg = f"{path}:{line_index}: duplicate chain/draw cell {key}"
            raise ProtocolError(msg)
        seen.add(key)
        values_obj = _obj(doc.get("values"), "values")
        values: dict[str, tuple[NumberValue, ...]] = {}
        for param in params:
            if param.name not in values_obj:
                msg = f"{path}:{line_index}: missing value for parameter {param.name!r}"
                raise ProtocolError(msg)
            values[param.name] = _flat_values(
                values_obj.get(param.name), param, f"values.{param.name}"
            )
        unknown = sorted(set(values_obj) - set(params_by_name))
        if unknown:
            msg = f"{path}:{line_index}: values contain unknown parameter(s): {unknown}"
            raise ProtocolError(msg)

        diverging: bool | None = None
        tree_depth: int | None = None
        tree_accept: float | None = None
        energy: float | None = None
        if sample_stats_mode in SUPPORTED_PER_DRAW_SAMPLE_STATS:
            if doc.get("sample_stats_mode") != sample_stats_mode:
                msg = (
                    f"{path}:{line_index}: draw line missing sample_stats_mode "
                    f"{sample_stats_mode!r}"
                )
                raise ProtocolError(msg)
            diverging = _bool(doc.get("diverging"), "diverging")
            tree_depth = _int(doc.get("tree_depth"), "tree_depth")
            tree_accept = _number(doc.get("tree_accept"), "tree_accept")
            if sample_stats_mode == PER_DRAW_SAMPLE_STATS_V2:
                energy = _number(doc.get("energy"), "energy")

        records.append(
            DrawRecord(
                chain=chain,
                draw=draw,
                values=MappingProxyType(values),
                diverging=diverging,
                tree_depth=tree_depth,
                tree_accept=tree_accept,
                energy=energy,
            )
        )

    if not records:
        msg = f"{path}: posterior stream has no draw records"
        raise ProtocolError(msg)
    header_chain_count = header.get("chain_count")
    if header_chain_count is not None and _int(header_chain_count, "chain_count") != len(chains):
        msg = f"{path}: header chain_count does not match chain_order length"
        raise ProtocolError(msg)
    header_draw_count = header.get("draw_count")
    if header_draw_count is not None and _int(header_draw_count, "draw_count") != len(records):
        msg = f"{path}: header draw_count does not match draw records"
        raise ProtocolError(msg)
    trailer_draw_count = trailer.get("draw_count")
    if trailer_draw_count is not None and _int(trailer_draw_count, "trailer.draw_count") != len(
        records
    ):
        msg = f"{path}: trailer draw_count does not match draw records"
        raise ProtocolError(msg)
    unknown_chains = sorted({record.chain for record in records} - set(chains))
    if unknown_chains:
        msg = f"{path}: draw records contain chain(s) not in header chain_order: {unknown_chains}"
        raise ProtocolError(msg)
    per_chain_counts = {chain: 0 for chain in chains}
    for record in records:
        per_chain_counts[record.chain] += 1
    count_values = set(per_chain_counts.values())
    if len(count_values) != 1:
        msg = f"{path}: uneven draw counts by chain: {per_chain_counts}"
        raise ProtocolError(msg)
    draws_per_chain = next(iter(count_values))
    trailer_draws_per_chain = trailer.get("draws_per_chain")
    if (
        trailer_draws_per_chain is not None
        and _int(trailer_draws_per_chain, "trailer.draws_per_chain") != draws_per_chain
    ):
        msg = f"{path}: trailer draws_per_chain does not match draw records"
        raise ProtocolError(msg)
    expected = {(chain, draw) for chain in chains for draw in range(draws_per_chain)}
    missing = sorted(expected - seen)
    if missing:
        msg = f"{path}: missing chain/draw cell(s): {missing[:5]}"
        raise ProtocolError(msg)
    extra_draws = sorted(seen - expected)
    if extra_draws:
        msg = f"{path}: draw index outside expected range: {extra_draws[:5]}"
        raise ProtocolError(msg)

    return PosteriorStream(
        params=params,
        chains=chains,
        draws_per_chain=draws_per_chain,
        sample_stats_mode=sample_stats_mode,
        records=tuple(records),
        attrs=_posterior_attrs(header),
    )


def _check_announced_count(
    path: Path,
    doc: Mapping[str, JsonValue],
    field: str,
    actual: int,
    *,
    prefix: str = "",
) -> None:
    value = doc.get(field)
    if value is None:
        return
    if _int(value, f"{prefix}{field}") != actual:
        msg = f"{path}: {prefix}{field} does not match parsed draw records"
        raise ProtocolError(msg)


def read_prior_predictive_stream(path: Path) -> PriorPredictiveStream:
    """Read a v0 prior-predictive stream."""
    docs = _load_lines(path)
    header = docs[0]
    sites = _sites(header, "prior_predictive_format")
    sites_by_name = {site.name: site for site in sites}
    trailer_doc = docs[-1]
    if "trailer" not in trailer_doc:
        msg = f"{path}: final line must be a trailer object"
        raise ProtocolError(msg)
    trailer = _obj(trailer_doc.get("trailer"), "trailer")
    records: list[PriorPredictiveRecord] = []
    seen: set[int] = set()
    for line_index, doc in enumerate(docs[1:-1], start=2):
        if doc.get("prior_predictive_format") != V0_MARKER:
            msg = f"{path}:{line_index}: draw line needs prior_predictive_format {V0_MARKER!r}"
            raise ProtocolError(msg)
        draw = _int(doc.get("draw"), "draw")
        if draw in seen:
            msg = f"{path}:{line_index}: duplicate prior-predictive draw {draw}"
            raise ProtocolError(msg)
        seen.add(draw)
        values_obj = _obj(doc.get("values"), "values")
        values: dict[str, tuple[NumberValue, ...]] = {}
        for site in sites:
            values[site.name] = _flat_values(values_obj.get(site.name), site, f"values.{site.name}")
        unknown = sorted(set(values_obj) - set(sites_by_name))
        if unknown:
            msg = f"{path}:{line_index}: values contain unknown generated site(s): {unknown}"
            raise ProtocolError(msg)
        records.append(PriorPredictiveRecord(draw=draw, values=MappingProxyType(values)))
    _check_announced_count(path, header, "draw_count", len(records))
    _check_announced_count(path, header, "draws", len(records))
    _check_announced_count(path, trailer, "draw_count", len(records), prefix="trailer.")
    _check_announced_count(path, trailer, "draws", len(records), prefix="trailer.")
    expected = set(range(len(records)))
    if seen != expected:
        msg = f"{path}: prior-predictive draws must be contiguous from zero"
        raise ProtocolError(msg)
    return PriorPredictiveStream(sites=sites, draws=len(records), records=tuple(records))


def read_posterior_predictive_stream(path: Path) -> PosteriorPredictiveStream:
    """Read a v0 posterior-predictive stream."""
    docs = _load_lines(path)
    header = docs[0]
    sites = _sites(header, "posterior_predictive_format")
    sites_by_name = {site.name: site for site in sites}
    trailer_doc = docs[-1]
    if "trailer" not in trailer_doc:
        msg = f"{path}: final line must be a trailer object"
        raise ProtocolError(msg)
    trailer = _obj(trailer_doc.get("trailer"), "trailer")
    records: list[PosteriorPredictiveRecord] = []
    seen: set[tuple[int, int]] = set()
    for line_index, doc in enumerate(docs[1:-1], start=2):
        if doc.get("posterior_predictive_format") != V0_MARKER:
            msg = f"{path}:{line_index}: draw line needs posterior_predictive_format {V0_MARKER!r}"
            raise ProtocolError(msg)
        source_chain = _int(doc.get("source_chain"), "source_chain")
        source_draw = _int(doc.get("source_draw"), "source_draw")
        key = (source_chain, source_draw)
        if key in seen:
            msg = f"{path}:{line_index}: duplicate posterior-predictive source cell {key}"
            raise ProtocolError(msg)
        seen.add(key)
        values_obj = _obj(doc.get("values"), "values")
        values: dict[str, tuple[NumberValue, ...]] = {}
        for site in sites:
            values[site.name] = _flat_values(values_obj.get(site.name), site, f"values.{site.name}")
        unknown = sorted(set(values_obj) - set(sites_by_name))
        if unknown:
            msg = f"{path}:{line_index}: values contain unknown generated site(s): {unknown}"
            raise ProtocolError(msg)
        records.append(
            PosteriorPredictiveRecord(
                source_chain=source_chain,
                source_draw=source_draw,
                values=MappingProxyType(values),
            )
        )
    _check_announced_count(path, header, "draw_count", len(records))
    _check_announced_count(path, trailer, "draw_count", len(records), prefix="trailer.")
    source_fit_seed = _optional_int(header.get("source_fit_seed"), "source_fit_seed")
    trailer_source_fit_seed = _optional_int(
        trailer.get("source_fit_seed"), "trailer.source_fit_seed"
    )
    if source_fit_seed is not None and trailer_source_fit_seed is not None:
        if source_fit_seed != trailer_source_fit_seed:
            msg = f"{path}: source_fit_seed differs between header and trailer"
            raise ProtocolError(msg)
    return PosteriorPredictiveStream(
        sites=sites,
        records=tuple(records),
        source_fit_seed=source_fit_seed if source_fit_seed is not None else trailer_source_fit_seed,
    )
