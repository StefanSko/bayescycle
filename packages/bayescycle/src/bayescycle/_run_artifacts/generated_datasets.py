"""Paired generated-dataset artifact contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import cast

from bayescycle._run_artifacts.canonical_data import (
    DataDoc,
    DataDocError,
    JsonValue,
    parse_data_doc,
)

MAX_GENERATED_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_GENERATED_LINE_BYTES = 8 * 1024 * 1024
_MAX_DEPTH = 64
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_MAX_COUNT = 1000
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MARKER_KEYS = (
    "generated_datasets_format",
    "artifact_kind",
    "artifact_scope",
)
_MARKER_VALUES = (
    "v0-provisional",
    "generated_dataset_pairs",
    "parameter_and_complete_dataset_joint_draws",
)
_PHASES = (
    "parse_json",
    "decode_ir",
    "bind_design",
    "draw_parameters",
    "simulate_outcomes",
    "emit_artifact",
)
_HEADER_KEYS = (
    *_MARKER_KEYS,
    "workflow_phases",
    "generation_model_hash",
    "design_hash",
    "parameter_source",
    "count",
    "seed",
    "draw_index_base",
    "parameter_schema",
    "dataset_schema",
)
_DRAW_KEYS = (
    *_MARKER_KEYS,
    "draw_index",
    "draw_count",
    "parameters",
    "dataset",
    "source_lineage",
)
_TRAILER_KEYS = (
    *_MARKER_KEYS,
    "workflow_phases",
    "generation_model_hash",
    "design_hash",
    "parameter_source",
    "count",
    "seed",
    "draw_count",
    "complete",
)


class GeneratedDatasetsArtifactError(ValueError):
    """Raised when a paired generated-dataset artifact is invalid."""


class _DuplicateJsonKey(ValueError):
    """Internal strict-JSON duplicate-key signal."""


@dataclass(frozen=True)
class _SchemaEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class GeneratedDatasetDraw:
    """One validated parameter/dataset pair."""

    draw_index: int
    parameters: DataDoc
    dataset: DataDoc
    source_kind: str
    _parameters_bytes: bytes = field(repr=False)
    _dataset_bytes: bytes = field(repr=False)


@dataclass(frozen=True)
class GeneratedDatasetSelection:
    """Exact standalone bytes selected from one paired record."""

    draw_index: int
    parameters_bytes: bytes
    dataset_bytes: bytes


@dataclass(frozen=True)
class GeneratedDatasetsArtifact:
    """Validated immutable paired generated-dataset stream."""

    generation_model_hash: str
    design_hash: str
    source_kind: str
    count: int
    seed: int
    draws: tuple[GeneratedDatasetDraw, ...]
    bytes: bytes
    _source_hash: str | None = field(repr=False)

    def select(self, draw_index: int) -> GeneratedDatasetSelection:
        """Select exact parameter and dataset bytes for one draw."""
        if draw_index < 0 or draw_index >= self.count:
            raise GeneratedDatasetsArtifactError(
                f"generated dataset index {draw_index} is outside range 0..{self.count - 1}"
            )
        draw = self.draws[draw_index]
        return GeneratedDatasetSelection(
            draw_index=draw_index,
            parameters_bytes=bytes(draw._parameters_bytes),
            dataset_bytes=bytes(draw._dataset_bytes),
        )


def parse_generated_datasets(data: bytes) -> GeneratedDatasetsArtifact:
    """Parse a bounded paired generated-dataset NDJSON artifact."""
    source_bytes = bytes(data)
    if len(source_bytes) > MAX_GENERATED_ARTIFACT_BYTES:
        raise GeneratedDatasetsArtifactError("generated-dataset artifact exceeds byte limit")
    if not source_bytes.endswith(b"\n"):
        raise GeneratedDatasetsArtifactError(
            "generated-dataset artifact needs a trailer and final LF"
        )
    raw_lines = source_bytes.split(b"\n")[:-1]
    if len(raw_lines) < 3:
        raise GeneratedDatasetsArtifactError("generated-dataset artifact is missing its trailer")
    for index, line in enumerate(raw_lines, start=1):
        if not line:
            raise GeneratedDatasetsArtifactError(f"generated-dataset line {index} is empty")
        if len(line) + 1 > MAX_GENERATED_LINE_BYTES:
            raise GeneratedDatasetsArtifactError(
                f"generated-dataset line {index} exceeds line limit"
            )
        _validate_depth(line, index)

    documents = tuple(_parse_line(line, index) for index, line in enumerate(raw_lines, start=1))
    header = _object(documents[0], "header")
    _exact_keys(header, _HEADER_KEYS, "header")
    _marker(header, "header")
    _phases(header["workflow_phases"], "header")
    generation_model_hash = _hash(header["generation_model_hash"], "generation_model_hash")
    design_hash = _hash(header["design_hash"], "design_hash")
    source_kind, source_hash = _parameter_source(header["parameter_source"])
    count = _integer(header["count"], "count", minimum=1, maximum=_MAX_COUNT)
    seed = _integer(header["seed"], "seed", minimum=0, maximum=_MAX_SAFE_INTEGER)
    if header["draw_index_base"] != "zero_based_generation_order":
        raise GeneratedDatasetsArtifactError("header draw_index_base is invalid")
    parameter_schema = _schema(header["parameter_schema"], "parameter_schema")
    dataset_schema = _schema(header["dataset_schema"], "dataset_schema")

    trailer_envelope = _object(documents[-1], "trailer envelope")
    _exact_keys(trailer_envelope, ("trailer",), "trailer envelope")
    trailer = _object(trailer_envelope["trailer"], "trailer")
    _exact_keys(trailer, _TRAILER_KEYS, "trailer")
    _marker(trailer, "trailer")
    _phases(trailer["workflow_phases"], "trailer")
    if (
        trailer["generation_model_hash"] != generation_model_hash
        or trailer["design_hash"] != design_hash
        or trailer["parameter_source"] != header["parameter_source"]
        or trailer["count"] != count
        or trailer["seed"] != seed
        or trailer["draw_count"] != count
        or trailer["complete"] is not True
    ):
        raise GeneratedDatasetsArtifactError(
            "header and trailer count, identity, or source disagree"
        )

    draw_documents = documents[1:-1]
    if len(draw_documents) != count:
        raise GeneratedDatasetsArtifactError(
            f"generated-dataset count declares {count}, stream contains {len(draw_documents)} draws"
        )
    draws: list[GeneratedDatasetDraw] = []
    for draw_index, (value, raw_line) in enumerate(
        zip(draw_documents, raw_lines[1:-1], strict=True)
    ):
        draw = _object(value, f"draw {draw_index}")
        _exact_keys(draw, _DRAW_KEYS, f"draw {draw_index}")
        _marker(draw, f"draw {draw_index}")
        if draw["draw_index"] != draw_index:
            raise GeneratedDatasetsArtifactError(
                f"draw_index must be contiguous; expected {draw_index}, got {draw['draw_index']!r}"
            )
        if draw["draw_count"] != count:
            raise GeneratedDatasetsArtifactError(
                f"draw {draw_index} draw_count disagrees with header"
            )
        try:
            parameters = parse_data_doc(draw["parameters"], label=f"draw {draw_index}.parameters")
            dataset = parse_data_doc(draw["dataset"], label=f"draw {draw_index}.dataset")
        except DataDocError as exc:
            raise GeneratedDatasetsArtifactError(str(exc)) from exc
        _validate_generation_doc(parameters, f"draw {draw_index}.parameters")
        _validate_generation_doc(dataset, f"draw {draw_index}.dataset")
        _matches_schema(parameters, parameter_schema, f"draw {draw_index}.parameters")
        _matches_schema(dataset, dataset_schema, f"draw {draw_index}.dataset")
        lineage_kind = _lineage(draw["source_lineage"], source_kind, draw_index)
        parameters_span, after_parameters = _top_level_value_span(raw_line, b"parameters", 0)
        dataset_span, _ = _top_level_value_span(raw_line, b"dataset", after_parameters)
        draws.append(
            GeneratedDatasetDraw(
                draw_index=draw_index,
                parameters=parameters,
                dataset=dataset,
                source_kind=lineage_kind,
                _parameters_bytes=parameters_span + b"\n",
                _dataset_bytes=dataset_span + b"\n",
            )
        )

    return GeneratedDatasetsArtifact(
        generation_model_hash=generation_model_hash,
        design_hash=design_hash,
        source_kind=source_kind,
        count=count,
        seed=seed,
        draws=tuple(draws),
        bytes=source_bytes,
        _source_hash=source_hash,
    )


def verify_generated_datasets(
    artifact: GeneratedDatasetsArtifact,
    *,
    model_bytes: bytes,
    design_bytes: bytes,
    fixed_parameters_bytes: bytes | None = None,
) -> None:
    """Verify hash-resolved model, design, and fixed-source lineage."""
    if _sha256(model_bytes) != artifact.generation_model_hash:
        raise GeneratedDatasetsArtifactError("generation model hash does not match resolved bytes")
    if _sha256(design_bytes) != artifact.design_hash:
        raise GeneratedDatasetsArtifactError("design hash does not match resolved bytes")
    design = _strict_data_bytes(design_bytes, "design")
    for draw in artifact.draws:
        prefix = draw.dataset.variables[: len(design.variables)]
        if prefix != design.variables:
            raise GeneratedDatasetsArtifactError(
                f"draw {draw.draw_index} dataset does not preserve the resolved design prefix"
            )
    if artifact.source_kind != "fixed":
        return
    if fixed_parameters_bytes is None:
        raise GeneratedDatasetsArtifactError("fixed parameters bytes are required for verification")
    if _sha256(fixed_parameters_bytes) != artifact._source_hash:
        raise GeneratedDatasetsArtifactError("fixed parameters hash does not match resolved bytes")
    fixed = _strict_data_bytes(fixed_parameters_bytes, "fixed parameters")
    for draw in artifact.draws:
        if draw.parameters != fixed:
            raise GeneratedDatasetsArtifactError(
                f"draw {draw.draw_index} parameters differ from fixed parameters"
            )


def _strict_data_bytes(data: bytes, label: str) -> DataDoc:
    if len(data) > MAX_GENERATED_LINE_BYTES:
        raise GeneratedDatasetsArtifactError(f"{label} exceeds byte limit")
    try:
        value = cast(
            JsonValue,
            json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object),
        )
        document = parse_data_doc(value, label=label)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKey, DataDocError) as exc:
        raise GeneratedDatasetsArtifactError(
            f"{label} is not a canonical data document: {exc}"
        ) from exc
    _validate_generation_doc(document, label)
    return document


def _parse_line(line: bytes, index: int) -> JsonValue:
    try:
        return cast(
            JsonValue,
            json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object),
        )
    except UnicodeDecodeError as exc:
        raise GeneratedDatasetsArtifactError(
            f"generated-dataset line {index} is not UTF-8"
        ) from exc
    except _DuplicateJsonKey as exc:
        raise GeneratedDatasetsArtifactError(
            f"generated-dataset line {index} has duplicate object key: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GeneratedDatasetsArtifactError(
            f"generated-dataset line {index} is not finite valid JSON: {exc.msg}"
        ) from exc


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _object(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise GeneratedDatasetsArtifactError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: dict[str, JsonValue], expected: tuple[str, ...], label: str) -> None:
    actual = tuple(value)
    if actual != expected:
        raise GeneratedDatasetsArtifactError(
            f"{label} has unknown, missing, or out-of-order keys: {list(actual)!r}"
        )


def _marker(value: dict[str, JsonValue], label: str) -> None:
    actual = tuple(value[name] for name in _MARKER_KEYS)
    if actual != _MARKER_VALUES:
        raise GeneratedDatasetsArtifactError(f"{label} generated-dataset marker is invalid")


def _phases(value: JsonValue, label: str) -> None:
    if value != list(_PHASES):
        raise GeneratedDatasetsArtifactError(f"{label} workflow phases are invalid")


def _hash(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise GeneratedDatasetsArtifactError(f"{label} must be a lowercase sha256 hash")
    return value


def _parameter_source(value: JsonValue) -> tuple[str, str | None]:
    source = _object(value, "parameter_source")
    kind = source.get("kind")
    if kind == "fixed":
        _exact_keys(source, ("kind", "parameters_hash"), "fixed parameter_source")
        return "fixed", _hash(source["parameters_hash"], "parameters_hash")
    if kind == "model-prior":
        _exact_keys(
            source,
            ("kind", "model_hash", "authored_provenance"),
            "model-prior parameter_source",
        )
        _hash(source["model_hash"], "model_hash")
        provenance = source["authored_provenance"]
        if provenance is not None:
            claims = _object(provenance, "authored_provenance")
            _exact_keys(
                claims,
                ("claimed_source_model_hash", "claimed_outcome_model_hash"),
                "authored_provenance",
            )
            _hash(claims["claimed_source_model_hash"], "claimed_source_model_hash")
            _hash(claims["claimed_outcome_model_hash"], "claimed_outcome_model_hash")
        return "model-prior", None
    if kind == "posterior":
        _exact_keys(
            source,
            ("kind", "fit_hash", "fit_model_hash", "fit_data_hash"),
            "posterior parameter_source",
        )
        for name in ("fit_hash", "fit_model_hash", "fit_data_hash"):
            _hash(source[name], name)
        return "posterior", None
    raise GeneratedDatasetsArtifactError(f"parameter_source has unknown kind {kind!r}")


def _schema(value: JsonValue, label: str) -> tuple[_SchemaEntry, ...]:
    if not isinstance(value, list):
        raise GeneratedDatasetsArtifactError(f"{label} must be an array")
    entries: list[_SchemaEntry] = []
    names: set[str] = set()
    for index, raw in enumerate(value):
        entry = _object(raw, f"{label}[{index}]")
        _exact_keys(entry, ("name", "dtype", "shape"), f"{label}[{index}]")
        name = entry["name"]
        dtype = entry["dtype"]
        shape_value = entry["shape"]
        if not isinstance(name, str) or not name or name in names:
            raise GeneratedDatasetsArtifactError(f"{label}[{index}] has invalid or duplicate name")
        if dtype not in {"bool", "int32", "int64", "float32", "float64"}:
            raise GeneratedDatasetsArtifactError(f"{label}[{index}] has invalid dtype")
        if not isinstance(shape_value, list):
            raise GeneratedDatasetsArtifactError(f"{label}[{index}] shape must be an array")
        shape = tuple(
            _integer(item, f"{label}[{index}] shape", minimum=0, maximum=_MAX_SAFE_INTEGER)
            for item in shape_value
        )
        names.add(name)
        entries.append(_SchemaEntry(name=name, dtype=cast(str, dtype), shape=shape))
    return tuple(entries)


def _validate_generation_doc(document: DataDoc, label: str) -> None:
    for entry in document.variables:
        for dim in entry.variable.shape:
            if dim > _MAX_SAFE_INTEGER:
                raise GeneratedDatasetsArtifactError(
                    f"{label}.{entry.name} shape is not a safe integer"
                )
        if entry.variable.dtype in {"int32", "int64"}:
            for value in entry.variable.values:
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or abs(value) > _MAX_SAFE_INTEGER
                ):
                    raise GeneratedDatasetsArtifactError(
                        f"{label}.{entry.name} contains an unsafe integer value"
                    )


def _matches_schema(document: DataDoc, schema: tuple[_SchemaEntry, ...], label: str) -> None:
    actual = tuple(
        _SchemaEntry(entry.name, entry.variable.dtype, entry.variable.shape)
        for entry in document.variables
    )
    if actual != schema:
        raise GeneratedDatasetsArtifactError(f"{label} does not match declared schema")


def _lineage(value: JsonValue, source_kind: str, draw_index: int) -> str:
    lineage = _object(value, f"draw {draw_index} source_lineage")
    if source_kind == "fixed":
        _exact_keys(lineage, ("kind",), f"draw {draw_index} fixed source_lineage")
    elif source_kind == "model-prior":
        _exact_keys(
            lineage,
            ("kind", "source_draw_index"),
            f"draw {draw_index} model-prior source_lineage",
        )
        if lineage["source_draw_index"] != draw_index:
            raise GeneratedDatasetsArtifactError(
                f"draw {draw_index} model-prior source_draw_index must equal draw_index"
            )
    else:
        _exact_keys(
            lineage,
            ("kind", "source_draw_index", "chain", "draw"),
            f"draw {draw_index} posterior source_lineage",
        )
        for name in ("source_draw_index", "chain", "draw"):
            _integer(lineage[name], name, minimum=0, maximum=_MAX_SAFE_INTEGER)
    if lineage.get("kind") != source_kind:
        raise GeneratedDatasetsArtifactError(
            f"draw {draw_index} source lineage kind disagrees with parameter source"
        )
    return source_kind


def _integer(value: JsonValue, label: str, *, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise GeneratedDatasetsArtifactError(
            f"{label} must be an integer in range {minimum}..{maximum}"
        )
    return value


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _validate_depth(line: bytes, line_index: int) -> None:
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
                raise GeneratedDatasetsArtifactError(
                    f"generated-dataset line {line_index} exceeds nesting depth"
                )
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth < 0:
                raise GeneratedDatasetsArtifactError(
                    f"generated-dataset line {line_index} has malformed nesting"
                )
    if in_string or depth != 0:
        raise GeneratedDatasetsArtifactError(
            f"generated-dataset line {line_index} has malformed finite JSON"
        )


def _top_level_value_span(line: bytes, key: bytes, offset: int) -> tuple[bytes, int]:
    marker = b'"' + key + b'"'
    key_start = line.find(marker, offset)
    if key_start < 0:
        raise GeneratedDatasetsArtifactError(f"draw is missing raw {key.decode()} bytes")
    colon = line.find(b":", key_start + len(marker))
    if colon < 0:
        raise GeneratedDatasetsArtifactError(f"draw {key.decode()} field is malformed")
    start = colon + 1
    while start < len(line) and line[start] in b" \t\r":
        start += 1
    if start >= len(line) or line[start] not in (0x7B, 0x5B):
        raise GeneratedDatasetsArtifactError(f"draw {key.decode()} must be a JSON container")
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(line)):
        byte = line[position]
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):
            depth += 1
        elif byte in (0x7D, 0x5D):
            depth -= 1
            if depth == 0:
                return line[start : position + 1], position + 1
    raise GeneratedDatasetsArtifactError(f"draw {key.decode()} has no closing delimiter")
