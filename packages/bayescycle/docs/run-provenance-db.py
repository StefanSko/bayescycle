# ruff: noqa: E501
"""Serialize bayescycle run-directory provenance into a SQLite database.

Every fresh model-level bayescycle run writes an append-only
``run.json`` (``bayescycle.run.v1``) recording the model source + hash, the
materialized inputs + hashes, and the declared outputs. Those records are flat
and self-describing, so the filesystem workflow serializes directly into a small
relational schema:

    runs          one row per run directory (kind, backend, model, hashes)
    run_inputs    one row per materialized input (role, source, sha256)
    run_outputs   one row per declared output (role, path)
    workflow_edges  derived: an input of one run IS an output of another run
                    (matched by absolute path on disk) -- this is the
                    cross-run, cross-backend, non-linear workflow graph

Usage::

    python docs/run-provenance-db.py DEMO_ROOT OUT.sqlite

``DEMO_ROOT`` is scanned recursively for ``run.json`` files.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

SCHEMA = """
CREATE TABLE runs (
    id              INTEGER PRIMARY KEY,
    workflow        TEXT NOT NULL,   -- 'complete' | 'mixed' | demo subtree name
    run_dir         TEXT NOT NULL UNIQUE,
    format          TEXT NOT NULL,
    kind            TEXT NOT NULL,
    backend         TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    model_source    TEXT NOT NULL,
    model_sha256    TEXT NOT NULL,
    model_ir_path   TEXT NOT NULL
);
CREATE TABLE run_inputs (
    id           INTEGER PRIMARY KEY,
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    role         TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    path         TEXT NOT NULL,
    format       TEXT
);
CREATE TABLE run_outputs (
    id      INTEGER PRIMARY KEY,
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    role    TEXT NOT NULL,
    path    TEXT NOT NULL,
    format  TEXT
);
CREATE TABLE workflow_edges (
    id              INTEGER PRIMARY KEY,
    producer_run_id INTEGER NOT NULL REFERENCES runs(id),
    producer_role   TEXT NOT NULL,
    consumer_run_id INTEGER NOT NULL REFERENCES runs(id),
    consumer_role   TEXT NOT NULL,
    artifact_path   TEXT NOT NULL
);
"""


def workflow_name(demo_root: Path, run_dir: Path) -> str:
    rel = run_dir.relative_to(demo_root)
    return rel.parts[0] if len(rel.parts) > 1 else "."


def build(demo_root: Path, out_path: Path) -> None:
    demo_root = demo_root.resolve()
    run_files = sorted(demo_root.rglob("run.json"))
    if not run_files:
        raise SystemExit(f"no run.json files under {demo_root}")

    if out_path.exists():
        out_path.unlink()
    con = sqlite3.connect(out_path)
    con.executescript(SCHEMA)

    # rows keyed by resolved run directory so we can resolve edges afterwards.
    run_id_by_dir: dict[Path, int] = {}
    output_index: dict[Path, tuple[int, str]] = {}  # resolved artifact path -> (run_id, role)

    for run_file in run_files:
        run_dir = run_file.parent.resolve()
        doc = json.loads(run_file.read_text())
        model = doc["model"]
        cur = con.execute(
            "INSERT INTO runs (workflow, run_dir, format, kind, backend, model_name, model_source, model_sha256, model_ir_path)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                workflow_name(demo_root, run_dir),
                str(run_dir.relative_to(demo_root)),
                doc["format"],
                doc["kind"],
                doc["backend"],
                model["name"],
                model["source_path"],
                model["sha256"],
                model["ir_path"],
            ),
        )
        run_id = cur.lastrowid
        assert run_id is not None
        run_id_by_dir[run_dir] = run_id
        for entry in doc["inputs"]:
            con.execute(
                "INSERT INTO run_inputs (run_id, role, source_path, sha256, path, format) VALUES (?,?,?,?,?,?)",
                (
                    run_id,
                    entry["role"],
                    entry["source_path"],
                    entry["sha256"],
                    entry["path"],
                    entry.get("format"),
                ),
            )
        for entry in doc["outputs"]:
            con.execute(
                "INSERT INTO run_outputs (run_id, role, path, format) VALUES (?,?,?,?)",
                (run_id, entry["role"], entry["path"], entry.get("format")),
            )
            output_index[(run_dir / entry["path"]).resolve()] = (run_id, entry["role"])

    # Derive cross-run edges: an input whose absolute source path is another
    # run's declared output. This recovers the workflow DAG, including the
    # bayesite-simulate -> bayesjax-fit cross-backend handoff in the mixed run.
    for run_file in run_files:
        run_dir = run_file.parent.resolve()
        consumer_id = run_id_by_dir[run_dir]
        doc = json.loads(run_file.read_text())
        for entry in doc["inputs"]:
            source = Path(entry["source_path"]).resolve()
            producer = output_index.get(source)
            if producer is not None and producer[0] != consumer_id:
                con.execute(
                    "INSERT INTO workflow_edges (producer_run_id, producer_role, consumer_run_id, consumer_role, artifact_path)"
                    " VALUES (?,?,?,?,?)",
                    (
                        producer[0],
                        producer[1],
                        consumer_id,
                        entry["role"],
                        str(source.relative_to(demo_root)),
                    ),
                )

    con.commit()
    runs = con.execute("SELECT count(*) FROM runs").fetchone()[0]
    inputs = con.execute("SELECT count(*) FROM run_inputs").fetchone()[0]
    outputs = con.execute("SELECT count(*) FROM run_outputs").fetchone()[0]
    edges = con.execute("SELECT count(*) FROM workflow_edges").fetchone()[0]
    con.close()
    print(
        f"wrote {out_path}: {runs} runs, {inputs} inputs, {outputs} outputs, {edges} workflow edges"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python docs/run-provenance-db.py DEMO_ROOT OUT.sqlite")
    build(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
