from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bayescycle._run_artifacts.posterior_source import (
    PortablePosteriorError,
    validate_portable_posterior,
)

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "playground/tests/fixtures/engine/eight_schools_non_centered"
)


def _documents() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (FIXTURE / "posterior.ndjson").read_text().splitlines()]


def _encode(documents: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(document, separators=(",", ":")).encode() + b"\n" for document in documents
    )


def _validate(posterior: bytes) -> None:
    validate_portable_posterior(
        model_bytes=(FIXTURE / "model.ir.json").read_bytes(),
        data_bytes=(FIXTURE / "data.json").read_bytes(),
        posterior_bytes=posterior,
    )


def _wrong_kind(documents: list[dict[str, Any]]) -> None:
    documents[0]["artifact_kind"] = "wrong"


def _missing_settings(documents: list[dict[str, Any]]) -> None:
    del documents[0]["settings"]


def _wrong_chain_topology(documents: list[dict[str, Any]]) -> None:
    documents[0]["chain_count"] = 99


def _ungrouped_draw(documents: list[dict[str, Any]]) -> None:
    documents[1]["chain"] = 99
    documents[1]["draw"] = 42


def _wrong_trailer_parameters(documents: list[dict[str, Any]]) -> None:
    documents[-1]["trailer"]["parameter_order"] = ["wrong"]


def _wrong_identity(documents: list[dict[str, Any]]) -> None:
    documents[-1]["trailer"]["posterior_identity_hash"] = "fnv1a64:wrong"


def _wrong_chain_stats(documents: list[dict[str, Any]]) -> None:
    documents[-1]["trailer"]["chains"][0]["draw_count"] = 1


def _unrepresentable_value(documents: list[dict[str, Any]]) -> None:
    documents[1]["values"]["mu"] = 10**400


@pytest.mark.parametrize(
    "mutate",
    [
        _wrong_kind,
        _missing_settings,
        _wrong_chain_topology,
        _ungrouped_draw,
        _wrong_trailer_parameters,
        _wrong_identity,
        _wrong_chain_stats,
        _unrepresentable_value,
    ],
)
def test_portable_posterior_rejects_incomplete_or_inconsistent_lineage(
    mutate: Callable[[list[dict[str, Any]]], None],
) -> None:
    documents = copy.deepcopy(_documents())
    mutate(documents)
    with pytest.raises(PortablePosteriorError):
        _validate(_encode(documents))


def test_real_posterior_fixture_has_portable_authority() -> None:
    parsed = validate_portable_posterior(
        model_bytes=(FIXTURE / "model.ir.json").read_bytes(),
        data_bytes=(FIXTURE / "data.json").read_bytes(),
        posterior_bytes=(FIXTURE / "posterior.ndjson").read_bytes(),
    )
    assert len(parsed.draws) == 600
