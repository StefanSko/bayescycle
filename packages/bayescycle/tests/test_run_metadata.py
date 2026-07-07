"""Unit coverage for the optional engine-provenance block in run.json."""

from __future__ import annotations

from pathlib import Path

from bayescycle._run_artifacts.run_metadata import (
    RunMetadata,
    RunMetadataEngine,
    RunMetadataModel,
    read_run_metadata,
    write_run_metadata,
)


def _model(run_dir: Path) -> RunMetadataModel:
    return RunMetadataModel(
        name="Simple",
        source_path=run_dir / "model.py",
        source_sha256="sha256:deadbeef",
        ir_path=run_dir / "model.ir.json",
    )


def test_engine_as_json_omits_none_version_and_sha256() -> None:
    engine = RunMetadataEngine(kind="system", path="/usr/local/bin/bayesite")

    assert engine.as_json() == {"kind": "system", "path": "/usr/local/bin/bayesite"}


def test_engine_as_json_includes_version_and_sha256_when_set() -> None:
    engine = RunMetadataEngine(
        kind="provisioned",
        path="/cache/bayesite",
        version="v0.2.0",
        sha256="a" * 64,
    )

    assert engine.as_json() == {
        "kind": "provisioned",
        "path": "/cache/bayesite",
        "version": "v0.2.0",
        "sha256": "a" * 64,
    }


def test_run_metadata_as_json_omits_engine_key_when_unset(tmp_path: Path) -> None:
    metadata = RunMetadata(
        kind="sample",
        backend="jaxstanv5",
        model=_model(tmp_path),
        inputs=(),
        outputs=(),
    )

    document = metadata.as_json(tmp_path)

    assert "engine" not in document


def test_run_metadata_as_json_includes_engine_block_when_set(tmp_path: Path) -> None:
    metadata = RunMetadata(
        kind="sample",
        backend="bayesite",
        model=_model(tmp_path),
        inputs=(),
        outputs=(),
        engine=RunMetadataEngine(kind="explicit", path="/opt/bayesite/bayesite"),
    )

    document = metadata.as_json(tmp_path)

    assert document["engine"] == {"kind": "explicit", "path": "/opt/bayesite/bayesite"}


def test_run_metadata_engine_block_round_trips(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata = RunMetadata(
        kind="sample",
        backend="bayesite",
        model=_model(run_dir),
        inputs=(),
        outputs=(),
        engine=RunMetadataEngine(
            kind="provisioned", path="/cache/bayesite", version="v0.2.0", sha256="b" * 64
        ),
    )
    write_run_metadata(run_dir, metadata)

    recorded = read_run_metadata(run_dir)

    assert recorded.engine is not None
    assert recorded.engine.kind == "provisioned"
    assert recorded.engine.path == "/cache/bayesite"
    assert recorded.engine.version == "v0.2.0"
    assert recorded.engine.sha256 == "b" * 64


def test_run_metadata_without_engine_round_trips_as_none(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata = RunMetadata(
        kind="sample",
        backend="jaxstanv5",
        model=_model(run_dir),
        inputs=(),
        outputs=(),
    )
    write_run_metadata(run_dir, metadata)

    recorded = read_run_metadata(run_dir)

    assert recorded.engine is None
