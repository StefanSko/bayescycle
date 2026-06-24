# State schema v1

The machine-readable contracts are `schemas/state.v1.schema.json` and
`schemas/event.v1.schema.json`. This document explains the intent of those
schemas.

`state.json` is the canonical machine-readable snapshot for a study. It is small,
mutable, and derived from human-approved decisions plus applied patches.

`events.jsonl` is the append-only audit log. Each line is one JSON object. The
snapshot may be reconstructed from events later, but the MVP treats
`state.json` as the current source of truth.

## Top-level fields

- `schema_version`: integer schema version. Current value: `1`.
- `study_id`: stable slug for the study.
- `title`: human-readable title.
- `cycle`: integer workflow cycle, incremented when a major revision loop begins.
- `phase`: current phase. One of `estimand`, `generative_model`,
  `estimator_plan`, `simulation`, `fit`, `critique`, `revision`.
- `gate`: current gate status. One of `open`, `awaiting_human`, `approved`,
  `blocked`.
- `toolchain`: current implementation profile.
- `approved`: approved scientific artifacts by phase role.
- `open_questions`: unresolved questions that may block progress.
- `decisions`: accepted or superseded human decisions.
- `artifacts`: registered files produced by humans, agents, or tools.
- `runs`: registered execution directories.
- `invalidated`: artifacts or decisions that must not be used as current support.
- `notes`: non-blocking notes.

## Approved references

Values under `approved` should be either `null` or an object with at least:

```json
{
  "id": "A0001",
  "path": "artifacts/0001-estimand.md",
  "approved_at": "2026-06-24T00:00:00Z"
}
```

Do not store long prose directly in `approved`; store prose in artifacts and keep
state as references.

## Questions

Questions should have:

```json
{
  "id": "Q0001",
  "phase": "estimand",
  "text": "What counts as a black cat?",
  "status": "open",
  "blocks_phase": "estimand",
  "options": ["exact_black", "contains_black", "domain_review"]
}
```

A phase is blocked if any question with `status: "open"` has
`blocks_phase` equal to the current or next phase.

## Decisions

Decisions should be append-only in spirit. To revise a decision, add a new
decision with `supersedes` instead of editing the old decision out of history.

```json
{
  "id": "D0001",
  "phase": "estimand",
  "status": "accepted",
  "text": "Use color == Black for cycle 1.",
  "supersedes": []
}
```

## Artifacts

Artifacts are files with typed roles. Use `references/artifact-contracts.md` for
allowed `kind` values.

```json
{
  "id": "A0001",
  "kind": "estimand_proposal",
  "phase": "estimand",
  "path": "artifacts/0001-estimand-proposals.md",
  "status": "proposed",
  "producer_profile": "bayescycle-study"
}
```

Statuses: `proposed`, `approved`, `rejected`, `superseded`, `invalidated`.

## Runs

Runs are execution directories, usually under `<study>/runs/`.

```json
{
  "id": "R0001",
  "kind": "real_fit",
  "path": "runs/fit-0001",
  "status": "diagnostics_pending",
  "model_artifact": "A0004",
  "data_artifact": "A0005"
}
```

Statuses: `planned`, `running`, `completed`, `diagnostics_pending`,
`diagnostics_passed`, `diagnostics_failed`, `invalidated`.

## Patches

State changes should be proposed as RFC 6902-style JSON Patch arrays and saved
under `<study>/patches/`. Apply only after human approval.
