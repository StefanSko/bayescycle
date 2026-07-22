"""A strict RFC 6902 JSON Patch boundary."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from bayescycle_study._documents import JsonObject, JsonValue, StudyDocumentError


class StudyPatchError(StudyDocumentError):
    """Raised when a JSON Patch document is malformed or cannot be applied."""


@dataclass(frozen=True)
class JsonPatch:
    """A validated immutable sequence of JSON Patch operations."""

    operations: tuple[JsonObject, ...]

    @classmethod
    def parse(cls, value: JsonValue) -> JsonPatch:
        if not isinstance(value, list) or not value:
            raise StudyPatchError("patch must be a non-empty JSON array")
        operations: list[JsonObject] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                raise StudyPatchError(f"patch operation {index} must be an object")
            operation = item.get("op")
            path = item.get("path")
            if not isinstance(operation, str):
                raise StudyPatchError(f"patch operation {index}.op must be a string")
            if operation not in {"add", "remove", "replace", "move", "copy", "test"}:
                raise StudyPatchError(f"unsupported patch operation: {operation}")
            if not isinstance(path, str):
                raise StudyPatchError(f"patch operation {index}.path must be a string")
            _tokens(path)
            required = {"op", "path"}
            if operation in {"add", "replace", "test"}:
                required.add("value")
            if operation in {"move", "copy"}:
                required.add("from")
                source = item.get("from")
                if not isinstance(source, str):
                    raise StudyPatchError(f"patch operation {index}.from must be a string")
                _tokens(source)
            missing = required - item.keys()
            if missing:
                raise StudyPatchError(
                    f"patch operation {index} is missing {', '.join(sorted(missing))}"
                )
            extra = item.keys() - required
            if extra:
                raise StudyPatchError(
                    f"patch operation {index} has unknown fields: {', '.join(sorted(extra))}"
                )
            operations.append(item)
        return cls(tuple(operations))

    def apply(self, document: JsonValue) -> JsonValue:
        """Apply all operations to a defensive copy of *document*."""

        result = copy.deepcopy(document)
        for index, operation in enumerate(self.operations):
            try:
                result = _apply_one(result, operation)
            except StudyPatchError as exc:
                raise StudyPatchError(f"patch operation {index} failed: {exc}") from exc
        return result


def _apply_one(document: JsonValue, operation: JsonObject) -> JsonValue:
    name = operation["op"]
    path = operation["path"]
    assert isinstance(name, str)
    assert isinstance(path, str)
    if name == "add":
        return _add(document, path, copy.deepcopy(operation["value"]))
    if name == "remove":
        result, _ = _remove(document, path)
        return result
    if name == "replace":
        _get(document, path)
        result, _ = _remove(document, path)
        return _add(result, path, copy.deepcopy(operation["value"]))
    if name == "test":
        actual = _get(document, path)
        if not _json_equal(actual, operation["value"]):
            raise StudyPatchError(f"test failed at {path}")
        return document
    source = operation["from"]
    assert isinstance(source, str)
    if name == "copy":
        return _add(document, path, copy.deepcopy(_get(document, source)))
    if path != source and _is_child(path, source):
        raise StudyPatchError("cannot move a value into one of its children")
    without_source, moved = _remove(document, source)
    return _add(without_source, path, moved)


def _get(document: JsonValue, pointer: str) -> JsonValue:
    current = document
    for token in _tokens(pointer):
        if isinstance(current, dict):
            if token not in current:
                raise StudyPatchError(f"path does not exist: {pointer}")
            current = current[token]
        elif isinstance(current, list):
            current = current[_array_index(token, len(current), allow_end=False)]
        else:
            raise StudyPatchError(f"path traverses a scalar: {pointer}")
    return current


def _add(document: JsonValue, pointer: str, value: JsonValue) -> JsonValue:
    tokens = _tokens(pointer)
    if not tokens:
        return value
    parent = _get_tokens(document, tokens[:-1], pointer)
    token = tokens[-1]
    if isinstance(parent, dict):
        parent[token] = value
        return document
    if isinstance(parent, list):
        if token == "-":
            parent.append(value)
        else:
            parent.insert(_array_index(token, len(parent), allow_end=True), value)
        return document
    raise StudyPatchError(f"add parent is a scalar: {pointer}")


def _remove(document: JsonValue, pointer: str) -> tuple[JsonValue, JsonValue]:
    tokens = _tokens(pointer)
    if not tokens:
        return None, document
    parent = _get_tokens(document, tokens[:-1], pointer)
    token = tokens[-1]
    if isinstance(parent, dict):
        if token not in parent:
            raise StudyPatchError(f"path does not exist: {pointer}")
        return document, parent.pop(token)
    if isinstance(parent, list):
        index = _array_index(token, len(parent), allow_end=False)
        return document, parent.pop(index)
    raise StudyPatchError(f"remove parent is a scalar: {pointer}")


def _get_tokens(document: JsonValue, tokens: tuple[str, ...], pointer: str) -> JsonValue:
    current = document
    for token in tokens:
        if isinstance(current, dict):
            if token not in current:
                raise StudyPatchError(f"path does not exist: {pointer}")
            current = current[token]
        elif isinstance(current, list):
            current = current[_array_index(token, len(current), allow_end=False)]
        else:
            raise StudyPatchError(f"path traverses a scalar: {pointer}")
    return current


def _tokens(pointer: str) -> tuple[str, ...]:
    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise StudyPatchError("JSON Pointer must be empty or start with '/'")
    result: list[str] = []
    for raw in pointer[1:].split("/"):
        index = 0
        decoded = ""
        while index < len(raw):
            if raw[index] != "~":
                decoded += raw[index]
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                raise StudyPatchError(f"invalid JSON Pointer escape in {pointer}")
            decoded += "~" if raw[index + 1] == "0" else "/"
            index += 2
        result.append(decoded)
    return tuple(result)


def _array_index(token: str, length: int, *, allow_end: bool) -> int:
    if not token or (len(token) > 1 and token.startswith("0")) or not token.isascii():
        raise StudyPatchError(f"invalid array index: {token}")
    if not token.isdigit():
        raise StudyPatchError(f"invalid array index: {token}")
    index = int(token)
    upper = length if allow_end else length - 1
    if index > upper:
        raise StudyPatchError(f"array index out of bounds: {token}")
    return index


def _is_child(path: str, parent: str) -> bool:
    parent_tokens = _tokens(parent)
    path_tokens = _tokens(path)
    return (
        len(path_tokens) > len(parent_tokens) and path_tokens[: len(parent_tokens)] == parent_tokens
    )


def _json_equal(left: JsonValue, right: JsonValue) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, int | float) and isinstance(right, int | float):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal(left[key], right[key]) for key in left
        )
    return left == right
