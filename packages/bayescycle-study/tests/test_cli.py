from __future__ import annotations

import json
from pathlib import Path

from bayescycle_study._cli import main


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _init(study: Path) -> None:
    assert (
        main(
            [
                "init",
                str(study),
                "--study-id",
                "normal-mean",
                "--title",
                "Normal mean study",
                "--actor",
                "tester",
            ]
        )
        == 0
    )


def test_init_creates_a_valid_minimal_study(tmp_path: Path) -> None:
    study = tmp_path / "study"

    _init(study)

    state = _read_json(study / "state.json")
    assert isinstance(state, dict)
    assert state["study_id"] == "normal-mean"
    assert state["title"] == "Normal mean study"
    assert state["phase"] == "estimand"
    assert state["gate"] == "awaiting_human"
    assert state["approved"]["report"] is None
    assert (study / "artifacts").is_dir()
    assert (study / "patches").is_dir()
    assert (study / "runs").is_dir()
    assert (study / ".gitignore").read_text(encoding="utf-8") == "/runs/\n"

    events = (study / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 1
    event = json.loads(events[0])
    assert event["event_id"] == "E0001"
    assert event["type"] == "study_created"
    assert event["actor"] == "tester"
    assert main(["validate", str(study)]) == 0


def test_init_preserves_unrelated_files_but_refuses_existing_state(tmp_path: Path) -> None:
    study = tmp_path / "study"
    study.mkdir()
    model = study / "model.py"
    model.write_text("# keep me\n", encoding="utf-8")

    _init(study)

    assert model.read_text(encoding="utf-8") == "# keep me\n"
    assert main(["init", str(study), "--study-id", "again", "--title", "Again"]) == 2


def test_validate_rejects_invalid_state_and_event_stream(tmp_path: Path) -> None:
    study = tmp_path / "study"
    _init(study)
    state_path = study / "state.json"
    state = _read_json(state_path)
    assert isinstance(state, dict)
    state["phase"] = "invented"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    assert main(["validate", str(study)]) == 2

    _init(tmp_path / "other")
    other = tmp_path / "other"
    with (other / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"event_id":"broken"}\n')
    assert main(["validate", str(other)]) == 2


def test_apply_validates_patch_updates_state_and_appends_audit_event(tmp_path: Path) -> None:
    study = tmp_path / "study"
    _init(study)
    patch = study / "patches" / "0001-open-gate.json"
    patch.write_text(
        json.dumps(
            [
                {"op": "test", "path": "/phase", "value": "estimand"},
                {"op": "replace", "path": "/gate", "value": "open"},
                {
                    "op": "add",
                    "path": "/notes/-",
                    "value": {"id": "N0001", "text": "Ready for manual work."},
                },
            ]
        ),
        encoding="utf-8",
    )

    assert main(["apply", str(study), str(patch), "--actor", "tester"]) == 0

    state = _read_json(study / "state.json")
    assert isinstance(state, dict)
    assert state["gate"] == "open"
    assert state["notes"] == [{"id": "N0001", "text": "Ready for manual work."}]
    events = [
        json.loads(line)
        for line in (study / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_id"] for event in events] == ["E0001", "E0002"]
    assert events[-1]["type"] == "state_patch_applied"
    assert events[-1]["payload"]["patch_sha256"].startswith("sha256:")
    assert main(["validate", str(study)]) == 0

    before = (study / "state.json").read_bytes()
    assert main(["apply", str(study), str(patch), "--actor", "tester"]) == 2
    assert (study / "state.json").read_bytes() == before


def test_apply_rejects_a_patch_that_would_make_state_invalid(tmp_path: Path) -> None:
    study = tmp_path / "study"
    _init(study)
    patch = study / "bad.json"
    patch.write_text(
        json.dumps([{"op": "replace", "path": "/phase", "value": "invented"}]),
        encoding="utf-8",
    )
    original_state = (study / "state.json").read_bytes()
    original_events = (study / "events.jsonl").read_bytes()

    assert main(["apply", str(study), str(patch), "--actor", "tester"]) == 2

    assert (study / "state.json").read_bytes() == original_state
    assert (study / "events.jsonl").read_bytes() == original_events
