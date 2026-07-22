"""Typed boundaries and validation for study JSON documents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

PHASES = (
    "estimand",
    "generative_model",
    "estimator_plan",
    "simulation",
    "fit",
    "critique",
    "report",
    "revision",
)
GATES = ("open", "awaiting_human", "approved", "blocked")
APPROVED_ROLES = (
    "scientific_question",
    "estimand",
    "generative_model",
    "estimator_plan",
    "simulation_tests",
    "real_fit",
    "model_criticism",
    "report",
)
_EVENT_ID = re.compile(r"E([0-9]{4,})")


class StudyDocumentError(ValueError):
    """Raised when a study document violates its public contract."""


@dataclass(frozen=True)
class ValidatedState:
    """A schema-valid study state document."""

    document: JsonObject
    study_id: str
    cycle: int
    phase: str


@dataclass(frozen=True)
class ValidatedEvent:
    """A schema-valid study event document."""

    document: JsonObject
    event_id: str
    study_id: str


def load_json(path: Path) -> JsonValue:
    """Load strict finite JSON from *path*."""

    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream, parse_constant=_reject_constant)
    except OSError as exc:
        raise StudyDocumentError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise StudyDocumentError(f"invalid JSON in {path}: {exc.msg}") from exc
    return cast(JsonValue, value)


def validate_state(value: JsonValue) -> ValidatedState:
    """Validate and normalize a v1 study state document."""

    state = _object(value, "state")
    _exact_keys(
        state,
        required={
            "schema_version",
            "study_id",
            "title",
            "cycle",
            "phase",
            "gate",
            "toolchain",
            "approved",
            "open_questions",
            "decisions",
            "artifacts",
            "runs",
            "invalidated",
            "notes",
        },
        optional={"thresholds"},
        label="state",
    )
    _const_int(state["schema_version"], 1, "state.schema_version")
    study_id = _nonempty_string(state["study_id"], "state.study_id")
    _string(state["title"], "state.title")
    cycle = _positive_int(state["cycle"], "state.cycle")
    phase = _choice(state["phase"], PHASES, "state.phase")
    _choice(state["gate"], GATES, "state.gate")
    _validate_toolchain(state["toolchain"])
    _validate_approved(state["approved"])
    _validate_questions(state["open_questions"])
    _validate_decisions(state["decisions"])
    _validate_artifacts(state["artifacts"])
    _validate_runs(state["runs"])
    _object_list(state["invalidated"], "state.invalidated")
    _object_list(state["notes"], "state.notes")
    if "thresholds" in state:
        _validate_thresholds(state["thresholds"])
    _validate_unique_record_ids(state)
    return ValidatedState(document=state, study_id=study_id, cycle=cycle, phase=phase)


def validate_event(value: JsonValue) -> ValidatedEvent:
    """Validate one v1 study event."""

    event = _object(value, "event")
    _exact_keys(
        event,
        required={
            "schema_version",
            "event_id",
            "type",
            "study_id",
            "cycle",
            "phase",
            "created_at",
            "actor",
            "payload",
        },
        optional=set(),
        label="event",
    )
    _const_int(event["schema_version"], 1, "event.schema_version")
    event_id = _nonempty_string(event["event_id"], "event.event_id")
    if _EVENT_ID.fullmatch(event_id) is None:
        raise StudyDocumentError("event.event_id must have form E0001")
    _nonempty_string(event["type"], "event.type")
    study_id = _nonempty_string(event["study_id"], "event.study_id")
    _positive_int(event["cycle"], "event.cycle")
    _choice(event["phase"], PHASES, "event.phase")
    _nonempty_string(event["created_at"], "event.created_at")
    _nonempty_string(event["actor"], "event.actor")
    _object(event["payload"], "event.payload")
    return ValidatedEvent(document=event, event_id=event_id, study_id=study_id)


def initial_state(*, study_id: str, title: str, toolchain_profile: str) -> JsonObject:
    """Construct the canonical initial v1 state."""

    state: JsonObject = {
        "schema_version": 1,
        "study_id": study_id,
        "title": title,
        "cycle": 1,
        "phase": "estimand",
        "gate": "awaiting_human",
        "toolchain": {
            "profile": toolchain_profile,
            "model_authoring": "bayesjax",
            "engine": "bayesite",
            "visualization": "bayesite-viz",
        },
        "approved": {role: None for role in APPROVED_ROLES},
        "thresholds": {
            "rhat_max": 1.01,
            "ess_bulk_min": 400,
            "ess_tail_min": 400,
            "divergences_max": 0,
        },
        "open_questions": [],
        "decisions": [],
        "artifacts": [],
        "runs": [],
        "invalidated": [],
        "notes": [],
    }
    validate_state(state)
    return state


def initial_event(
    *, study_id: str, title: str, actor: str, created_at: str, toolchain_profile: str
) -> JsonObject:
    """Construct the first append-only event."""

    event: JsonObject = {
        "schema_version": 1,
        "event_id": "E0001",
        "type": "study_created",
        "study_id": study_id,
        "cycle": 1,
        "phase": "estimand",
        "created_at": created_at,
        "actor": actor,
        "payload": {"title": title, "toolchain_profile": toolchain_profile},
    }
    validate_event(event)
    return event


def event_number(event_id: str) -> int:
    """Return the numeric suffix of a validated event ID."""

    match = _EVENT_ID.fullmatch(event_id)
    if match is None:
        raise StudyDocumentError(f"invalid event ID: {event_id}")
    return int(match.group(1))


def validate_relative_path(value: JsonValue, label: str) -> str:
    """Validate a portable study-relative artifact path."""

    path = _nonempty_string(value, label)
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or "\\" in path:
        raise StudyDocumentError(f"{label} must be a safe study-relative POSIX path")
    return path


def _validate_toolchain(value: JsonValue) -> None:
    toolchain = _object(value, "state.toolchain")
    if "profile" not in toolchain:
        raise StudyDocumentError("state.toolchain is missing profile")
    for key, item in toolchain.items():
        _string(item, f"state.toolchain.{key}")
    _nonempty_string(toolchain["profile"], "state.toolchain.profile")


def _validate_approved(value: JsonValue) -> None:
    approved = _object(value, "state.approved")
    _exact_keys(
        approved,
        required=set(APPROVED_ROLES[:-1]),
        optional={APPROVED_ROLES[-1]},
        label="state.approved",
    )
    for role, reference in approved.items():
        if reference is None:
            continue
        ref = _object(reference, f"state.approved.{role}")
        for required in ("id", "path"):
            if required not in ref:
                raise StudyDocumentError(f"state.approved.{role} is missing {required}")
        _string(ref["id"], f"state.approved.{role}.id")
        validate_relative_path(ref["path"], f"state.approved.{role}.path")
        if "approved_at" in ref:
            _string(ref["approved_at"], f"state.approved.{role}.approved_at")


def _validate_questions(value: JsonValue) -> None:
    for index, question in enumerate(_object_list(value, "state.open_questions")):
        label = f"state.open_questions[{index}]"
        _required(question, ("id", "phase", "text", "status"), label)
        _string(question["id"], f"{label}.id")
        _string(question["phase"], f"{label}.phase")
        _string(question["text"], f"{label}.text")
        _choice(question["status"], ("open", "answered", "closed"), f"{label}.status")
        if "blocks_phase" in question:
            _string(question["blocks_phase"], f"{label}.blocks_phase")
        if "options" in question:
            _string_list(question["options"], f"{label}.options")


def _validate_decisions(value: JsonValue) -> None:
    for index, decision in enumerate(_object_list(value, "state.decisions")):
        label = f"state.decisions[{index}]"
        _required(decision, ("id", "phase", "status", "text"), label)
        _string(decision["id"], f"{label}.id")
        _string(decision["phase"], f"{label}.phase")
        _choice(
            decision["status"],
            ("accepted", "superseded", "rejected"),
            f"{label}.status",
        )
        _string(decision["text"], f"{label}.text")
        if "supersedes" in decision:
            _string_list(decision["supersedes"], f"{label}.supersedes")


def _validate_artifacts(value: JsonValue) -> None:
    for index, artifact in enumerate(_object_list(value, "state.artifacts")):
        label = f"state.artifacts[{index}]"
        _required(artifact, ("id", "kind", "phase", "path", "status"), label)
        for key in ("id", "kind", "phase"):
            _string(artifact[key], f"{label}.{key}")
        validate_relative_path(artifact["path"], f"{label}.path")
        _choice(
            artifact["status"],
            ("proposed", "approved", "rejected", "superseded", "invalidated"),
            f"{label}.status",
        )
        if "producer_profile" in artifact:
            _string(artifact["producer_profile"], f"{label}.producer_profile")


def _validate_runs(value: JsonValue) -> None:
    for index, run in enumerate(_object_list(value, "state.runs")):
        label = f"state.runs[{index}]"
        _required(run, ("id", "kind", "path", "status"), label)
        _string(run["id"], f"{label}.id")
        _string(run["kind"], f"{label}.kind")
        validate_relative_path(run["path"], f"{label}.path")
        _choice(
            run["status"],
            (
                "planned",
                "running",
                "completed",
                "diagnostics_pending",
                "diagnostics_passed",
                "diagnostics_failed",
                "invalidated",
            ),
            f"{label}.status",
        )


def _validate_thresholds(value: JsonValue) -> None:
    thresholds = _object(value, "state.thresholds")
    for key in ("rhat_max", "ess_bulk_min", "ess_tail_min"):
        if key in thresholds and not _is_number(thresholds[key]):
            raise StudyDocumentError(f"state.thresholds.{key} must be a number")
    if "divergences_max" in thresholds:
        _integer(thresholds["divergences_max"], "state.thresholds.divergences_max")


def _validate_unique_record_ids(state: JsonObject) -> None:
    seen: set[str] = set()
    for collection_name in ("open_questions", "decisions", "artifacts", "runs"):
        records = _object_list(state[collection_name], f"state.{collection_name}")
        for record in records:
            identifier = _string(record["id"], f"state.{collection_name}.id")
            if identifier in seen:
                raise StudyDocumentError(f"duplicate study record ID: {identifier}")
            seen.add(identifier)


def _object(value: JsonValue, label: str) -> JsonObject:
    if not isinstance(value, dict):
        raise StudyDocumentError(f"{label} must be an object")
    return value


def _object_list(value: JsonValue, label: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise StudyDocumentError(f"{label} must be an array")
    return [_object(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _string_list(value: JsonValue, label: str) -> list[str]:
    if not isinstance(value, list):
        raise StudyDocumentError(f"{label} must be an array")
    return [_string(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str):
        raise StudyDocumentError(f"{label} must be a string")
    return value


def _nonempty_string(value: JsonValue, label: str) -> str:
    result = _string(value, label)
    if not result:
        raise StudyDocumentError(f"{label} must not be empty")
    return result


def _integer(value: JsonValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StudyDocumentError(f"{label} must be an integer")
    return value


def _positive_int(value: JsonValue, label: str) -> int:
    result = _integer(value, label)
    if result < 1:
        raise StudyDocumentError(f"{label} must be at least 1")
    return result


def _const_int(value: JsonValue, expected: int, label: str) -> None:
    if _integer(value, label) != expected:
        raise StudyDocumentError(f"{label} must equal {expected}")


def _choice(value: JsonValue, choices: tuple[str, ...], label: str) -> str:
    result = _string(value, label)
    if result not in choices:
        raise StudyDocumentError(f"{label} must be one of {', '.join(choices)}")
    return result


def _required(document: JsonObject, keys: tuple[str, ...], label: str) -> None:
    missing = set(keys) - document.keys()
    if missing:
        raise StudyDocumentError(f"{label} is missing {', '.join(sorted(missing))}")


def _exact_keys(
    document: JsonObject, *, required: set[str], optional: set[str], label: str
) -> None:
    missing = required - document.keys()
    extra = document.keys() - required - optional
    if missing:
        raise StudyDocumentError(f"{label} is missing {', '.join(sorted(missing))}")
    if extra:
        raise StudyDocumentError(f"{label} has unknown fields: {', '.join(sorted(extra))}")


def _is_number(value: JsonValue) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float)


def _reject_constant(value: str) -> None:
    raise StudyDocumentError(f"non-finite JSON number is not allowed: {value}")
