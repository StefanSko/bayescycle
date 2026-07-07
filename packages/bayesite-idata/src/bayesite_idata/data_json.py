"""Bayesite ``data.json`` typed-value decoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np

from bayesite_idata.errors import AssemblyError

type NumericArray = np.ndarray[tuple[int, ...], np.dtype[np.integer | np.floating | np.bool_]]

type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

_WRAPPER_FORMAT = "bayescycle.data.json.v1"


def read_data_arrays(path: Path) -> dict[str, NumericArray]:
    """Read Bayesite data.json into dtype-preserving NumPy arrays.

    Accepts two document layouts: the bare top-level map keyed by data
    variable name, and the canonical bayescycle wrapper
    ``{"format": "bayescycle.data.json.v1", "variables": {...}}``. The
    ``format`` field is the discriminator; a document carrying any other
    ``format`` value fails explicitly instead of being read as a bare map.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = cast(JsonValue, json.load(f))
    except json.JSONDecodeError as exc:
        msg = f"{path}: invalid data.json: {exc.msg}"
        raise AssemblyError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"{path}: data.json must be an object keyed by data variable name"
        raise AssemblyError(msg)
    variables = _unwrap_variables(raw, path)
    return {name: _data_value_to_array(value, f"data.{name}") for name, value in variables.items()}


def _unwrap_variables(raw: dict[str, JsonValue], path: Path) -> dict[str, JsonValue]:
    if "format" not in raw:
        return raw
    format_value = raw["format"]
    if format_value != _WRAPPER_FORMAT:
        msg = (
            f"{path}: data.json format {format_value!r} is unsupported; "
            f"expected {_WRAPPER_FORMAT!r} or a bare map keyed by data variable name"
        )
        raise AssemblyError(msg)
    variables = raw.get("variables")
    if not isinstance(variables, dict):
        msg = f'{path}: data.json with format {_WRAPPER_FORMAT!r} needs a "variables" object'
        raise AssemblyError(msg)
    return variables


def _data_value_to_array(value: JsonValue, field: str) -> NumericArray:
    if isinstance(value, dict) and "shape" in value and "values" in value:
        shape_value = value["shape"]
        values_value = value["values"]
        if not isinstance(shape_value, list):
            msg = f"{field}.shape must be an array"
            raise AssemblyError(msg)
        shape = tuple(_shape_dim(item, f"{field}.shape") for item in shape_value)
        if not isinstance(values_value, list):
            msg = f"{field}.values must be an array"
            raise AssemblyError(msg)
        dtype_value = value.get("dtype")
        dtype = _data_dtype(dtype_value, field)
        _validate_typed_data_values(values_value, dtype_value, field)
        arr = np.asarray(values_value, dtype=dtype)
        expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if arr.size != expected:
            msg = f"{field}.values length {arr.size} does not match shape {shape}"
            raise AssemblyError(msg)
        return arr.reshape(shape)
    if isinstance(value, list):
        return np.asarray(value)
    if isinstance(value, bool):
        return np.asarray(value, dtype=np.bool_)
    if isinstance(value, int):
        return np.asarray(value, dtype=np.int64)
    if isinstance(value, float):
        return np.asarray(value, dtype=np.float64)
    msg = f"{field} has unsupported data value shape"
    raise AssemblyError(msg)


def _validate_typed_data_values(
    values: list[JsonValue], dtype: JsonValue | None, field: str
) -> None:
    if dtype is None:
        return
    if not isinstance(dtype, str):
        msg = f"{field}.dtype must be a string when present"
        raise AssemblyError(msg)
    if dtype == "bool":
        if any(not isinstance(value, bool) for value in values):
            msg = f"{field}.values for dtype 'bool' must contain only JSON booleans"
            raise AssemblyError(msg)
        return
    if dtype in {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}:
        for value in values:
            if not isinstance(value, int) or isinstance(value, bool):
                msg = f"{field}.values for integer dtype {dtype!r} must contain only JSON integers"
                raise AssemblyError(msg)
            if dtype.startswith("uint") and value < 0:
                msg = f"{field}.values for unsigned dtype {dtype!r} must be non-negative"
                raise AssemblyError(msg)
        return
    if dtype in {"float32", "float64"} and any(
        not isinstance(value, int | float) or isinstance(value, bool) for value in values
    ):
        msg = f"{field}.values for float dtype {dtype!r} must contain only JSON numbers"
        raise AssemblyError(msg)


def _data_dtype(
    value: JsonValue | None, field: str
) -> type[np.bool_] | type[np.int64] | type[np.uint64] | type[np.float64]:
    if value is None:
        return np.float64
    if not isinstance(value, str):
        msg = f"{field}.dtype must be a string when present"
        raise AssemblyError(msg)
    if value == "bool":
        return np.bool_
    if value in {"int8", "int16", "int32", "int64"}:
        return np.int64
    if value in {"uint8", "uint16", "uint32", "uint64"}:
        return np.uint64
    if value in {"float32", "float64"}:
        return np.float64
    msg = f"{field}.dtype {value!r} is unsupported"
    raise AssemblyError(msg)


def _shape_dim(value: JsonValue, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"{field} must contain non-negative integers"
        raise AssemblyError(msg)
    return value
