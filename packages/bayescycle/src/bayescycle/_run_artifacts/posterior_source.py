"""Validation of portable posterior sources used by functional generation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import cast

from bayescycle._run_artifacts.canonical_data import JsonValue

MAX_POSTERIOR_SOURCE_BYTES = 8 * 1024 * 1024
MAX_POSTERIOR_LINE_BYTES = 8 * 1024 * 1024
_MAX_DEPTH = 64
_MAX_SAFE_INTEGER = 9_007_199_254_740_991


class PortablePosteriorError(ValueError):
    """Raised when posterior bytes do not carry portable fit authority."""


class _DuplicateKey(ValueError):
    pass


@dataclass(frozen=True)
class PosteriorParameter:
    name: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class PosteriorSourceDraw:
    source_draw_index: int
    chain: int
    draw: int
    values: tuple[tuple[str, tuple[float, ...]], ...]


@dataclass(frozen=True)
class PortablePosterior:
    parameters: tuple[PosteriorParameter, ...]
    draws: tuple[PosteriorSourceDraw, ...]


def validate_portable_posterior(
    *, model_bytes: bytes, data_bytes: bytes, posterior_bytes: bytes
) -> PortablePosterior:
    """Validate a complete fit stream and its exact model/data fingerprint."""
    if not posterior_bytes or len(posterior_bytes) > MAX_POSTERIOR_SOURCE_BYTES:
        raise PortablePosteriorError(
            f"posterior source must contain 1..{MAX_POSTERIOR_SOURCE_BYTES} bytes"
        )
    if not posterior_bytes.endswith(b"\n"):
        raise PortablePosteriorError("posterior source must end in LF")
    lines = posterior_bytes.split(b"\n")[:-1]
    if len(lines) < 3:
        raise PortablePosteriorError("posterior source needs header, draws, and trailer")
    documents: list[JsonValue] = []
    for line_number, line in enumerate(lines, start=1):
        if not line or len(line) + 1 > MAX_POSTERIOR_LINE_BYTES:
            raise PortablePosteriorError(
                f"posterior source line {line_number} is empty or oversized"
            )
        _validate_depth(line, line_number)
        try:
            document = cast(
                JsonValue,
                json.loads(
                    line.decode("utf-8"),
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                ),
            )
            _validate_finite(document)
            documents.append(document)
        except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey) as exc:
            raise PortablePosteriorError(
                f"posterior source line {line_number} is not strict JSON: {exc}"
            ) from exc

    header = _object(documents[0], "posterior header")
    trailer_envelope = _object(documents[-1], "posterior trailer envelope")
    if tuple(trailer_envelope) != ("trailer",):
        raise PortablePosteriorError("posterior trailer envelope is invalid")
    trailer = _object(trailer_envelope["trailer"], "posterior trailer")
    _marker(header, "posterior header")
    _marker(trailer, "posterior trailer")
    _kind_scope(header, "posterior header")
    _kind_scope(trailer, "posterior trailer")
    expected_fingerprint = _fingerprint(model_bytes, data_bytes)
    for document, label in ((header, "header"), (trailer, "trailer")):
        if document.get("model_data_fingerprint") != expected_fingerprint:
            raise PortablePosteriorError(
                f"posterior {label} fingerprint does not match exact model/data bytes"
            )

    raw_parameters = header.get("params")
    raw_order = header.get("parameter_order")
    if not isinstance(raw_parameters, list) or not isinstance(raw_order, list):
        raise PortablePosteriorError("posterior header parameter metadata is missing")
    parameters: list[PosteriorParameter] = []
    for index, raw in enumerate(raw_parameters):
        parameter = _object(raw, f"posterior params[{index}]")
        name = parameter.get("name")
        shape = parameter.get("shape")
        if not isinstance(name, str) or not name or not isinstance(shape, list):
            raise PortablePosteriorError(f"posterior params[{index}] is invalid")
        dimensions = tuple(_integer(value, "posterior parameter shape") for value in shape)
        parameters.append(PosteriorParameter(name, dimensions))
    names = tuple(parameter.name for parameter in parameters)
    if (
        len(set(names)) != len(names)
        or raw_order != list(names)
        or header.get("packing") != list(names)
    ):
        raise PortablePosteriorError("posterior parameter order or packing is invalid")
    if _integer(header.get("parameter_count"), "posterior parameter_count") != len(names):
        raise PortablePosteriorError("posterior parameter count is invalid")
    header_seed = _integer(header.get("seed"), "posterior seed")
    settings = _object(header.get("settings"), "posterior settings")
    _integer(settings.get("num_warmup"), "posterior settings.num_warmup")
    target_accept = _finite_number(
        settings.get("target_accept"), "posterior settings.target_accept"
    )
    if not 0.0 < target_accept < 1.0:
        raise PortablePosteriorError("posterior settings.target_accept must be in (0, 1)")
    max_treedepth = _positive_integer(
        settings.get("max_treedepth"), "posterior settings.max_treedepth"
    )
    if max_treedepth > 20:
        raise PortablePosteriorError("posterior max_treedepth must be at most 20")
    sample_stats_mode = header.get("sample_stats_mode")
    if sample_stats_mode not in ("per_draw_v1", "per_draw_v2"):
        raise PortablePosteriorError("posterior sample_stats_mode is invalid")
    draws_per_chain = _positive_integer(settings.get("num_draws"), "posterior settings.num_draws")
    chain_order = _integer_array(header.get("chain_order"), "posterior chain_order")
    if not chain_order or len(set(chain_order)) != len(chain_order):
        raise PortablePosteriorError("posterior chain_order must be non-empty and unique")
    chain_count = _positive_integer(header.get("chain_count"), "posterior chain_count")
    if _positive_integer(header.get("chains"), "posterior chains") != chain_count:
        raise PortablePosteriorError("posterior chains disagrees with chain_count")
    if chain_count != len(chain_order):
        raise PortablePosteriorError("posterior chain_count disagrees with chain_order")
    draw_documents = documents[1:-1]
    draw_count = _positive_integer(header.get("draw_count"), "posterior draw_count")
    if draw_count != chain_count * draws_per_chain or draw_count != len(draw_documents):
        raise PortablePosteriorError("posterior draw count is incomplete")
    if _integer(trailer.get("seed"), "posterior trailer seed") != header_seed:
        raise PortablePosteriorError("posterior trailer seed disagrees with header")
    if (
        trailer.get("parameter_order") != list(names)
        or _integer(trailer.get("parameter_count"), "posterior trailer parameter_count")
        != len(names)
        or _integer(trailer.get("params"), "posterior trailer params") != len(names)
        or _integer(trailer.get("draw_count"), "posterior trailer draw_count") != draw_count
        or _integer(trailer.get("draws_per_chain"), "posterior trailer draws_per_chain")
        != draws_per_chain
        or _integer(trailer.get("chain_count"), "posterior trailer chain_count") != chain_count
        or _integer_array(trailer.get("chain_order"), "posterior trailer chain_order")
        != chain_order
    ):
        raise PortablePosteriorError("posterior trailer metadata disagrees with header")
    header_identity = header.get("posterior_identity_hash")
    trailer_identity = trailer.get("posterior_identity_hash")
    if (header_identity is None) != (trailer_identity is None) or (
        header_identity is not None and header_identity != trailer_identity
    ):
        raise PortablePosteriorError("posterior identity disagrees between header and trailer")
    raw_chain_stats = trailer.get("chains")
    if not isinstance(raw_chain_stats, list) or len(raw_chain_stats) != chain_count:
        raise PortablePosteriorError("posterior trailer chain statistics are incomplete")
    declared_chain_stats: list[tuple[int, tuple[int, ...], float]] = []
    for index, raw_stat in enumerate(raw_chain_stats):
        statistic = _object(raw_stat, f"posterior trailer chains[{index}]")
        if (
            _integer(statistic.get("chain"), "posterior trailer chain") != chain_order[index]
            or _integer(statistic.get("draw_count"), "posterior trailer chain draw_count")
            != draws_per_chain
        ):
            raise PortablePosteriorError("posterior trailer chain statistics are out of order")
        step_size = _finite_number(statistic.get("step_size"), "posterior trailer step_size")
        if step_size <= 0.0:
            raise PortablePosteriorError("posterior trailer step_size must be positive")
        mean_accept = _finite_number(statistic.get("mean_accept"), "posterior trailer mean_accept")
        if not 0.0 <= mean_accept <= 1.0:
            raise PortablePosteriorError("posterior trailer mean_accept must be in [0, 1]")
        declared_divergences = _integer(
            statistic.get("divergences"), "posterior trailer divergences"
        )
        declared_histogram = _integer_array(
            statistic.get("treedepth_histogram"), "posterior trailer treedepth_histogram"
        )
        if len(declared_histogram) != max_treedepth + 1:
            raise PortablePosteriorError("posterior treedepth histogram length is invalid")
        declared_chain_stats.append((declared_divergences, declared_histogram, mean_accept))
    draws: list[PosteriorSourceDraw] = []
    seen_coordinates: set[tuple[int, int]] = set()
    actual_divergences = [0] * chain_count
    actual_histograms = [[0] * (max_treedepth + 1) for _ in range(chain_count)]
    actual_accept_sums = [0.0] * chain_count
    for source_index, raw in enumerate(draw_documents):
        document = _object(raw, f"posterior draw {source_index}")
        _marker(document, f"posterior draw {source_index}")
        _kind_scope(document, f"posterior draw {source_index}")
        raw_index = document.get("draw_index")
        if _integer(raw_index, "posterior draw_index") != source_index:
            raise PortablePosteriorError("posterior draw indices are not contiguous")
        chain = _integer(document.get("chain"), "posterior chain")
        draw = _integer(document.get("draw"), "posterior draw")
        if _integer(document.get("seed"), "posterior draw seed") != header_seed:
            raise PortablePosteriorError("posterior draw seed disagrees with header")
        if _integer(document.get("draw_count"), "posterior draw_count") != draw_count:
            raise PortablePosteriorError("posterior draw_count disagrees with header")
        if _integer(document.get("chain_count"), "posterior draw chain_count") != chain_count:
            raise PortablePosteriorError("posterior draw chain_count disagrees with header")
        if _integer_array(document.get("chain_order"), "posterior draw chain_order") != chain_order:
            raise PortablePosteriorError("posterior draw chain_order disagrees with header")
        expected_chain = chain_order[source_index // draws_per_chain]
        expected_draw = source_index % draws_per_chain
        if chain != expected_chain or draw != expected_draw:
            raise PortablePosteriorError("posterior draws are not grouped in chain order")
        tree_depth = _integer(document.get("tree_depth"), "posterior tree_depth")
        if tree_depth > max_treedepth:
            raise PortablePosteriorError("posterior tree_depth exceeds declared max_treedepth")
        if document.get("sample_stats_mode") != sample_stats_mode:
            raise PortablePosteriorError("posterior draw sample_stats_mode disagrees with header")
        diverging = document.get("diverging")
        if not isinstance(diverging, bool):
            raise PortablePosteriorError("posterior diverging must be a boolean")
        tree_accept = _finite_number(document.get("tree_accept"), "posterior tree_accept")
        if not 0.0 <= tree_accept <= 1.0:
            raise PortablePosteriorError("posterior tree_accept must be in [0, 1]")
        if sample_stats_mode == "per_draw_v2":
            _finite_number(document.get("energy"), "posterior energy")
        elif "energy" in document:
            raise PortablePosteriorError("posterior per_draw_v1 must not contain energy")
        if (chain, draw) in seen_coordinates:
            raise PortablePosteriorError("posterior chain/draw coordinates are duplicated")
        seen_coordinates.add((chain, draw))
        chain_position = source_index // draws_per_chain
        actual_divergences[chain_position] += int(diverging)
        actual_histograms[chain_position][tree_depth] += 1
        actual_accept_sums[chain_position] += tree_accept
        if document.get("parameter_order") != list(names) or _integer(
            document.get("parameter_count"), "posterior draw parameter_count"
        ) != len(names):
            raise PortablePosteriorError("posterior draw parameter order is invalid")
        values = _object(document.get("values"), f"posterior draw {source_index} values")
        if tuple(values) != names:
            raise PortablePosteriorError("posterior draw values do not match parameter order")
        flattened: list[tuple[str, tuple[float, ...]]] = []
        for parameter in parameters:
            numbers = tuple(
                _values_for_shape(values[parameter.name], parameter.shape, parameter.name)
            )
            flattened.append((parameter.name, numbers))
        draws.append(PosteriorSourceDraw(source_index, chain, draw, tuple(flattened)))
    for index, (declared_divergences, declared_histogram, declared_mean_accept) in enumerate(
        declared_chain_stats
    ):
        actual_mean_accept = actual_accept_sums[index] / draws_per_chain
        if (
            actual_divergences[index] != declared_divergences
            or tuple(actual_histograms[index]) != declared_histogram
            or abs(actual_mean_accept - declared_mean_accept) > 1e-9
        ):
            raise PortablePosteriorError(
                "posterior trailer sampler statistics disagree with retained draws"
            )
    return PortablePosterior(tuple(parameters), tuple(draws))


def _fingerprint(model_bytes: bytes, data_bytes: bytes) -> str:
    framed = b"bayescycle-model-data-v1\n" + model_bytes + b"\n" + data_bytes
    return f"sha256:{hashlib.sha256(framed).hexdigest()}"


def _values_for_shape(value: JsonValue, shape: tuple[int, ...], label: str) -> list[float]:
    if shape:
        if not isinstance(value, list) or len(value) != shape[0]:
            raise PortablePosteriorError(f"posterior value {label} does not match declared shape")
        result: list[float] = []
        for item in value:
            result.extend(_values_for_shape(item, shape[1:], label))
        return result
    if isinstance(value, list):
        raise PortablePosteriorError(f"posterior value {label} does not match declared shape")
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise PortablePosteriorError(f"posterior value {label} must contain finite numbers")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise PortablePosteriorError(
            f"posterior value {label} is outside the finite float range"
        ) from exc
    if not math.isfinite(converted):
        raise PortablePosteriorError(f"posterior value {label} must contain finite numbers")
    return [converted]


def _validate_finite(value: JsonValue) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _validate_finite(item)
    elif isinstance(value, list):
        for item in value:
            _validate_finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise PortablePosteriorError("posterior source contains a non-finite number")
    elif isinstance(value, int) and not isinstance(value, bool) and abs(value) > _MAX_SAFE_INTEGER:
        raise PortablePosteriorError("posterior source contains an unsafe integer")


def _finite_number(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise PortablePosteriorError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise PortablePosteriorError(f"{label} must be a finite number")
    return converted


def _positive_integer(value: object, label: str) -> int:
    result = _integer(value, label)
    if result < 1:
        raise PortablePosteriorError(f"{label} must be positive")
    return result


def _integer_array(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise PortablePosteriorError(f"{label} must be an array")
    return tuple(_integer(item, label) for item in value)


def _integer(value: object, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > _MAX_SAFE_INTEGER
    ):
        raise PortablePosteriorError(f"{label} must be a nonnegative safe integer")
    return value


def _kind_scope(document: dict[str, JsonValue], label: str) -> None:
    if (
        document.get("artifact_kind") != "posterior_draws"
        or document.get("artifact_scope") != "observed_data_conditioned_parameter_draws"
    ):
        raise PortablePosteriorError(f"{label} kind or scope is invalid")


def _marker(document: dict[str, JsonValue], label: str) -> None:
    if document.get("draws_format") != "v0-provisional":
        raise PortablePosteriorError(f"{label} format is invalid")


def _object(value: object, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PortablePosteriorError(f"{label} must be an object")
    return cast(dict[str, JsonValue], value)


def _reject_constant(value: str) -> JsonValue:
    raise PortablePosteriorError(f"posterior source contains non-JSON number {value}")


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _validate_depth(line: bytes, line_number: int) -> None:
    depth = 0
    in_string = False
    escaped = False
    for byte in line:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
            if depth > _MAX_DEPTH:
                raise PortablePosteriorError(
                    f"posterior source line {line_number} exceeds nesting depth"
                )
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise PortablePosteriorError(
                    f"posterior source line {line_number} has malformed nesting"
                )
    if in_string or depth != 0:
        raise PortablePosteriorError(f"posterior source line {line_number} has malformed nesting")
