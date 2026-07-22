"""Filesystem ownership for initialized and patched study directories."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from bayescycle_study._documents import (
    JsonObject,
    JsonValue,
    StudyDocumentError,
    ValidatedEvent,
    ValidatedState,
    event_number,
    initial_event,
    initial_state,
    load_json,
    validate_event,
    validate_relative_path,
    validate_state,
)
from bayescycle_study._patches import JsonPatch


class StudyStorageError(StudyDocumentError):
    """Raised when a study directory cannot be safely read or changed."""


@dataclass(frozen=True)
class ValidatedStudy:
    """A validated state snapshot and its append-only events."""

    root: Path
    state: ValidatedState
    events: tuple[ValidatedEvent, ...]


def initialize_study(
    root: Path,
    *,
    study_id: str,
    title: str,
    actor: str,
    toolchain_profile: str,
) -> ValidatedStudy:
    """Create a minimal study around any unrelated files already in *root*."""

    if not study_id:
        raise StudyStorageError("study ID must not be empty")
    if not actor:
        raise StudyStorageError("actor must not be empty")
    state_path = root / "state.json"
    events_path = root / "events.jsonl"
    if state_path.exists() or events_path.exists():
        raise StudyStorageError("study already has state.json or events.jsonl")
    for name in ("artifacts", "patches", "runs"):
        path = root / name
        if path.exists() and not path.is_dir():
            raise StudyStorageError(f"study path is not a directory: {path}")

    state = initial_state(
        study_id=study_id,
        title=title,
        toolchain_profile=toolchain_profile,
    )
    event = initial_event(
        study_id=study_id,
        title=title,
        actor=actor,
        created_at=_timestamp(),
        toolchain_profile=toolchain_profile,
    )
    root.mkdir(parents=True, exist_ok=True)
    for name in ("artifacts", "patches", "runs"):
        (root / name).mkdir(exist_ok=True)

    created: list[Path] = []
    try:
        _write_new_json(state_path, state)
        created.append(state_path)
        _write_new_events(events_path, (event,))
        created.append(events_path)
        _ensure_runs_ignored(root / ".gitignore")
    except OSError as exc:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise StudyStorageError(f"cannot initialize study: {exc}") from exc
    return validate_study(root)


def validate_study(root: Path) -> ValidatedStudy:
    """Validate canonical state, all events, and registered local paths."""

    if not root.is_dir():
        raise StudyStorageError(f"study directory does not exist: {root}")
    state = validate_state(load_json(root / "state.json"))
    events = _load_events(root / "events.jsonl")
    if not events:
        raise StudyStorageError("events.jsonl must contain at least one event")
    if events[0].document["type"] != "study_created":
        raise StudyStorageError("the first event must be study_created")
    seen_event_ids: set[str] = set()
    previous_number = 0
    for event in events:
        if event.study_id != state.study_id:
            raise StudyStorageError(
                f"event {event.event_id} belongs to {event.study_id}, expected {state.study_id}"
            )
        if event.event_id in seen_event_ids:
            raise StudyStorageError(f"duplicate event ID: {event.event_id}")
        number = event_number(event.event_id)
        if number <= previous_number:
            raise StudyStorageError("event IDs must increase monotonically")
        seen_event_ids.add(event.event_id)
        previous_number = number
    _validate_registered_paths(root, state.document)
    _validate_approved_references(state.document)
    return ValidatedStudy(root=root, state=state, events=events)


def apply_patch(root: Path, patch_path: Path, *, actor: str) -> ValidatedStudy:
    """Apply one validated patch and append a transition event."""

    if not actor:
        raise StudyStorageError("actor must not be empty")
    patch_bytes = _read_bytes(patch_path)
    patch_sha = f"sha256:{hashlib.sha256(patch_bytes).hexdigest()}"
    with _study_lock(root):
        study = validate_study(root)
        for event in study.events:
            payload = event.document["payload"]
            assert isinstance(payload, dict)
            if payload.get("patch_sha256") == patch_sha:
                raise StudyStorageError("patch has already been applied")
        patch = JsonPatch.parse(load_json(patch_path))
        changed_value = patch.apply(study.state.document)
        changed = validate_state(changed_value)
        before_bytes = _json_bytes(study.state.document)
        after_bytes = _json_bytes(changed.document)
        event = _patch_event(
            study=study,
            changed=changed,
            actor=actor,
            patch_path=patch_path,
            patch_sha=patch_sha,
            before_bytes=before_bytes,
            after_bytes=after_bytes,
        )
        state_path = root / "state.json"
        events_path = root / "events.jsonl"
        original_state = state_path.read_bytes()
        _atomic_write(state_path, after_bytes)
        try:
            _append_event(events_path, event)
        except OSError as exc:
            _atomic_write(state_path, original_state)
            raise StudyStorageError(f"cannot append patch event: {exc}") from exc
    return validate_study(root)


def _patch_event(
    *,
    study: ValidatedStudy,
    changed: ValidatedState,
    actor: str,
    patch_path: Path,
    patch_sha: str,
    before_bytes: bytes,
    after_bytes: bytes,
) -> JsonObject:
    number = event_number(study.events[-1].event_id) + 1
    try:
        display_path = patch_path.resolve().relative_to(study.root.resolve()).as_posix()
    except ValueError:
        display_path = patch_path.name
    event: JsonObject = {
        "schema_version": 1,
        "event_id": f"E{number:04d}",
        "type": "state_patch_applied",
        "study_id": changed.study_id,
        "cycle": changed.cycle,
        "phase": changed.phase,
        "created_at": _timestamp(),
        "actor": actor,
        "payload": {
            "patch_path": display_path,
            "patch_sha256": patch_sha,
            "state_before_sha256": f"sha256:{hashlib.sha256(before_bytes).hexdigest()}",
            "state_after_sha256": f"sha256:{hashlib.sha256(after_bytes).hexdigest()}",
        },
    }
    validate_event(event)
    return event


def _load_events(path: Path) -> tuple[ValidatedEvent, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise StudyStorageError(f"cannot read {path}: {exc}") from exc
    events: list[ValidatedEvent] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise StudyStorageError(f"events.jsonl line {line_number} is empty")
        try:
            value = cast(JsonValue, json.loads(line, parse_constant=_reject_event_constant))
        except json.JSONDecodeError as exc:
            raise StudyStorageError(
                f"invalid JSON on events.jsonl line {line_number}: {exc.msg}"
            ) from exc
        try:
            events.append(validate_event(value))
        except StudyDocumentError as exc:
            raise StudyStorageError(f"invalid event on line {line_number}: {exc}") from exc
    return tuple(events)


def _validate_registered_paths(root: Path, state: JsonObject) -> None:
    for collection_name in ("artifacts", "runs"):
        records = state[collection_name]
        assert isinstance(records, list)
        for record_value in records:
            assert isinstance(record_value, dict)
            relative = validate_relative_path(record_value["path"], f"state.{collection_name}.path")
            status = record_value["status"]
            if collection_name == "runs" and status == "planned":
                continue
            candidate = root / relative
            if not candidate.exists():
                raise StudyStorageError(f"registered path does not exist: {relative}")


def _validate_approved_references(state: JsonObject) -> None:
    artifacts_value = state["artifacts"]
    approved_value = state["approved"]
    assert isinstance(artifacts_value, list)
    assert isinstance(approved_value, dict)
    artifacts: dict[str, JsonObject] = {}
    for value in artifacts_value:
        assert isinstance(value, dict)
        identifier = value["id"]
        assert isinstance(identifier, str)
        artifacts[identifier] = value
    for role, reference in approved_value.items():
        if reference is None:
            continue
        assert isinstance(reference, dict)
        identifier = reference["id"]
        path = reference["path"]
        assert isinstance(identifier, str)
        artifact = artifacts.get(identifier)
        if artifact is None:
            raise StudyStorageError(f"approved.{role} references unknown artifact {identifier}")
        if artifact["status"] != "approved" or artifact["path"] != path:
            raise StudyStorageError(
                f"approved.{role} must match an approved artifact with the same path"
            )


def _write_new_json(path: Path, value: JsonObject) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json_bytes(value).decode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _write_new_events(path: Path, events: tuple[JsonObject, ...]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append_event(path: Path, event: JsonObject) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(value: JsonObject) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise StudyStorageError(f"cannot read {path}: {exc}") from exc


def _ensure_runs_ignored(path: Path) -> None:
    line = "/runs/"
    if not path.exists():
        path.write_text(line + "\n", encoding="utf-8")
        return
    existing = path.read_text(encoding="utf-8")
    if line in existing.splitlines():
        return
    separator = "" if not existing or existing.endswith("\n") else "\n"
    path.write_text(existing + separator + line + "\n", encoding="utf-8")


@contextmanager
def _study_lock(root: Path) -> Iterator[None]:
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - supported platforms provide fcntl
        raise StudyStorageError("study mutation requires POSIX file locking") from exc
    lock_path = root / ".study.lock"
    try:
        with lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise StudyStorageError(f"cannot lock study: {exc}") from exc


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _reject_event_constant(value: str) -> None:
    raise StudyStorageError(f"non-finite JSON number is not allowed: {value}")
