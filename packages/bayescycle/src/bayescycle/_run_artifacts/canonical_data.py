"""Canonical Bayescycle JSON data artifacts.

The ``bayescycle.data.json.v1`` document is the workflow boundary for concrete
model inputs and generated observed data.  Backend adapters may materialize this
stable document into backend-native runtime inputs, but backend-native data files
must not be passed between workflow stages as the durable artifact.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

DATA_DOC_FORMAT = "bayescycle.data.json.v1"
SUPPORTED_DTYPES = frozenset({"bool", "int32", "int64", "float32", "float64"})

_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type DataValue = bool | int | float


class DataDocError(ValueError):
    """Raised when a data artifact cannot be parsed or validated."""


@dataclass(frozen=True)
class DataVariable:
    """One typed, flattened data array in a canonical data document."""

    dtype: str
    shape: tuple[int, ...]
    values: tuple[DataValue, ...]

    def to_json(self) -> dict[str, JsonValue]:
        """Return the JSON object for this variable."""
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "values": list(self.values),
        }


@dataclass(frozen=True)
class NamedDataVariable:
    """A named variable entry, preserving JSON object insertion order."""

    name: str
    variable: DataVariable


@dataclass(frozen=True)
class DataDoc:
    """Canonical Bayescycle data artifact."""

    variables: tuple[NamedDataVariable, ...]

    def to_json(self) -> dict[str, JsonValue]:
        """Return a JSON-serializable canonical data document."""
        return {
            "format": DATA_DOC_FORMAT,
            "variables": {entry.name: entry.variable.to_json() for entry in self.variables},
        }

    def variable_map(self) -> dict[str, DataVariable]:
        """Return variables keyed by name for adapter materialization."""
        return {entry.name: entry.variable for entry in self.variables}


def read_data_doc(path: Path) -> DataDoc:
    """Read a canonical or legacy JSON data file and return a canonical document.

    Legacy plain-array/Bayesite typed input is accepted at CLI boundaries so
    existing data files remain usable.  The returned value is always the
    canonical Bayescycle representation.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            document = cast(JsonValue, json.load(f))
    except json.JSONDecodeError as exc:
        raise DataDocError(f"{path}: invalid JSON: {exc.msg}") from exc
    return normalize_data_doc(document, label=str(path))


def write_data_doc(path: Path, doc: DataDoc) -> None:
    """Write a canonical data document with strict JSON encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc.to_json(), f, indent=2, allow_nan=False)
        f.write("\n")


def parse_data_doc(document: JsonValue, *, label: str = "data document") -> DataDoc:
    """Parse a strict ``bayescycle.data.json.v1`` document."""
    if not isinstance(document, dict):
        raise DataDocError(f"{label}: data document must be a JSON object")
    keys = set(document)
    expected = {"format", "variables"}
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        parts: list[str] = []
        if missing:
            parts.append(f"missing {missing}")
        if extra:
            parts.append(f"unexpected {extra}")
        detail = "; ".join(parts)
        raise DataDocError(f"{label}: canonical data document keys are invalid ({detail})")
    if document["format"] != DATA_DOC_FORMAT:
        raise DataDocError(
            f"{label}: unsupported data format {document['format']!r}; expected {DATA_DOC_FORMAT!r}"
        )
    variables_value = document["variables"]
    if not isinstance(variables_value, dict):
        raise DataDocError(f"{label}: variables must be an object keyed by variable name")
    entries = tuple(
        NamedDataVariable(name=name, variable=_parse_canonical_variable(value, f"{label}.{name}"))
        for name, value in variables_value.items()
    )
    return DataDoc(entries)


def normalize_data_doc(document: JsonValue, *, label: str = "data document") -> DataDoc:
    """Normalize canonical or legacy JSON data into a canonical ``DataDoc``."""
    if not isinstance(document, dict):
        raise DataDocError(f"{label}: data document must be a JSON object keyed by variable name")
    if document.get("format") == DATA_DOC_FORMAT:
        return parse_data_doc(document, label=label)
    return DataDoc(
        tuple(
            NamedDataVariable(
                name=name, variable=_legacy_value_to_variable(value, f"{label}.{name}")
            )
            for name, value in document.items()
        )
    )


def data_doc_to_plain_json(doc: DataDoc) -> dict[str, JsonValue]:
    """Materialize a canonical data document as plain Python JSON values.

    This is the bayesjax adapter boundary: scalars become Python scalars and
    arrays become rectangular nested lists in row-major order.
    """
    return {
        entry.name: _reshape_flat_values(entry.variable.values, entry.variable.shape)
        for entry in doc.variables
    }


def _parse_canonical_variable(value: JsonValue, label: str) -> DataVariable:
    if not isinstance(value, dict):
        raise DataDocError(f"{label}: variable must be an object")
    keys = set(value)
    expected = {"dtype", "shape", "values"}
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        parts: list[str] = []
        if missing:
            parts.append(f"missing {missing}")
        if extra:
            parts.append(f"unexpected {extra}")
        raise DataDocError(f"{label}: variable keys are invalid ({'; '.join(parts)})")
    dtype = _parse_dtype(value["dtype"], label)
    shape = _parse_shape(value["shape"], label)
    values_field = value["values"]
    if not isinstance(values_field, list):
        raise DataDocError(f"{label}.values must be a flat array")
    values = tuple(_parse_typed_scalar(item, dtype, f"{label}.values") for item in values_field)
    _validate_value_count(shape, values, label)
    return DataVariable(dtype=dtype, shape=shape, values=values)


def _legacy_value_to_variable(value: JsonValue, label: str) -> DataVariable:
    if _is_typed_legacy_value(value):
        return _legacy_typed_value_to_variable(cast(dict[str, JsonValue], value), label)
    shape, values = _flatten_legacy_value(value, label)
    dtype = _infer_dtype(values)
    typed_values = tuple(_parse_typed_scalar(item, dtype, label) for item in values)
    return DataVariable(dtype=dtype, shape=shape, values=typed_values)


def _is_typed_legacy_value(value: JsonValue) -> bool:
    if not isinstance(value, dict):
        return False
    keys = set(value)
    return {"dtype", "shape", "values"}.issubset(keys)


def _legacy_typed_value_to_variable(value: dict[str, JsonValue], label: str) -> DataVariable:
    keys = set(value)
    expected = {"dtype", "shape", "values"}
    if keys != expected:
        extra = sorted(keys - expected)
        raise DataDocError(f"{label}: legacy typed value has unsupported metadata keys {extra}")
    raw_dtype = value["dtype"]
    if not isinstance(raw_dtype, str):
        raise DataDocError(f"{label}.dtype must be a string")
    dtype = _canonical_dtype_for_legacy(raw_dtype, label)
    shape = _parse_shape(value["shape"], label)
    raw_values = value["values"]
    flat_values = raw_values if isinstance(raw_values, list) else [raw_values]
    values = tuple(_parse_typed_scalar(item, dtype, f"{label}.values") for item in flat_values)
    _validate_value_count(shape, values, label)
    return DataVariable(dtype=dtype, shape=shape, values=values)


def _canonical_dtype_for_legacy(dtype: str, label: str) -> str:
    if dtype in SUPPORTED_DTYPES:
        return dtype
    if dtype in {"int8", "int16", "uint8", "uint16", "uint32", "uint64"}:
        return "int64"
    raise DataDocError(f"{label}.dtype {dtype!r} is unsupported")


def _parse_dtype(value: JsonValue, label: str) -> str:
    if not isinstance(value, str):
        raise DataDocError(f"{label}.dtype must be a string")
    if value not in SUPPORTED_DTYPES:
        raise DataDocError(f"{label}.dtype {value!r} is unsupported")
    return value


def _parse_shape(value: JsonValue, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise DataDocError(f"{label}.shape must be an array")
    shape: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise DataDocError(f"{label}.shape entries must be non-negative integers")
        shape.append(item)
    return tuple(shape)


def _parse_typed_scalar(value: JsonValue, dtype: str, label: str) -> DataValue:
    if dtype == "bool":
        if not isinstance(value, bool):
            raise DataDocError(f"{label} for dtype 'bool' must contain only JSON booleans")
        return value
    if dtype in {"int32", "int64"}:
        if not isinstance(value, int) or isinstance(value, bool):
            raise DataDocError(
                f"{label} for integer dtype {dtype!r} must contain only JSON integers"
            )
        if dtype == "int32" and not (_INT32_MIN <= value <= _INT32_MAX):
            raise DataDocError(f"{label} value {value} is outside int32 range")
        if dtype == "int64" and not (_INT64_MIN <= value <= _INT64_MAX):
            raise DataDocError(f"{label} value {value} is outside int64 range")
        return value
    if dtype in {"float32", "float64"}:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise DataDocError(f"{label} for float dtype {dtype!r} must contain only JSON numbers")
        as_float = float(value)
        if not math.isfinite(as_float):
            raise DataDocError(f"{label} contains non-finite value {value!r}")
        return as_float
    raise AssertionError(f"unhandled dtype {dtype!r}")


def _validate_value_count(
    shape: tuple[int, ...], values: tuple[DataValue, ...], label: str
) -> None:
    expected = math.prod(shape) if shape else 1
    if len(values) != expected:
        raise DataDocError(
            f"{label}.values length {len(values)} does not match shape {list(shape)} "
            f"(expected {expected})"
        )


def _flatten_legacy_value(
    value: JsonValue, label: str
) -> tuple[tuple[int, ...], tuple[DataValue, ...]]:
    if isinstance(value, bool):
        return (), (value,)
    if isinstance(value, int | float) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise DataDocError(f"{label} contains non-finite value {value!r}")
        return (), (value,)
    if isinstance(value, list):
        shape, flat = _flatten_legacy_array(value, label)
        return shape, tuple(flat)
    raise DataDocError(f"{label}: data values must be scalars or rectangular arrays")


def _flatten_legacy_array(
    value: list[JsonValue], label: str
) -> tuple[tuple[int, ...], list[DataValue]]:
    if not value:
        return (0,), []
    first = value[0]
    if isinstance(first, list):
        child_shape, flat = _flatten_legacy_array(first, label)
        out = list(flat)
        for item in value[1:]:
            if not isinstance(item, list):
                raise DataDocError(f"{label}: arrays must be rectangular and non-ragged")
            item_shape, item_flat = _flatten_legacy_array(item, label)
            if item_shape != child_shape:
                raise DataDocError(f"{label}: arrays must be rectangular and non-ragged")
            out.extend(item_flat)
        return (len(value), *child_shape), out
    out: list[DataValue] = []
    for item in value:
        if isinstance(item, list):
            raise DataDocError(f"{label}: arrays must be rectangular and non-ragged")
        if isinstance(item, bool):
            out.append(item)
        elif isinstance(item, int | float):
            if isinstance(item, float) and not math.isfinite(item):
                raise DataDocError(f"{label} contains non-finite value {item!r}")
            out.append(item)
        else:
            raise DataDocError(f"{label}: arrays may only contain JSON numbers or booleans")
    return (len(value),), out


def _infer_dtype(values: tuple[DataValue, ...]) -> str:
    if not values:
        return "float64"
    if all(isinstance(value, bool) for value in values):
        return "bool"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "int64"
    if all(isinstance(value, int | float) and not isinstance(value, bool) for value in values):
        return "float64"
    raise DataDocError("data arrays must not mix boolean and numeric values")


def _reshape_flat_values(values: tuple[DataValue, ...], shape: tuple[int, ...]) -> JsonValue:
    if not shape:
        return cast(JsonValue, values[0])
    stride = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows: list[JsonValue] = []
    for index in range(shape[0]):
        start = index * stride
        stop = start + stride
        rows.append(_reshape_flat_values(values[start:stop], shape[1:]))
    return rows
