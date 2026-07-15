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


def _missing_draw_index(documents: list[dict[str, Any]]) -> None:
    del documents[1]["draw_index"]


def _non_json_number(documents: list[dict[str, Any]]) -> None:
    documents[0]["settings"]["target_accept"] = float("nan")


def _overflowing_number(documents: list[dict[str, Any]]) -> None:
    documents[0]["settings"]["target_accept"] = float("inf")


def _wrong_value_rank(documents: list[dict[str, Any]]) -> None:
    values = documents[1]["values"]["z"]
    documents[1]["values"]["z"] = [values]


def _float_header_seed(documents: list[dict[str, Any]]) -> None:
    documents[0]["seed"] = float(documents[0]["seed"])


def _float_num_warmup(documents: list[dict[str, Any]]) -> None:
    documents[0]["settings"]["num_warmup"] = 300.0


def _float_max_treedepth(documents: list[dict[str, Any]]) -> None:
    documents[0]["settings"]["max_treedepth"] = 10.0


def _float_tree_depth(documents: list[dict[str, Any]]) -> None:
    documents[1]["tree_depth"] = float(documents[1]["tree_depth"])


def _float_divergences(documents: list[dict[str, Any]]) -> None:
    documents[-1]["trailer"]["chains"][0]["divergences"] = 0.0


def _float_header_chains(documents: list[dict[str, Any]]) -> None:
    documents[0]["chains"] = float(documents[0]["chains"])


def _float_draw_seed(documents: list[dict[str, Any]]) -> None:
    documents[1]["seed"] = float(documents[1]["seed"])


def _float_draw_count(documents: list[dict[str, Any]]) -> None:
    documents[1]["draw_count"] = float(documents[1]["draw_count"])


def _float_draw_chain_count(documents: list[dict[str, Any]]) -> None:
    documents[1]["chain_count"] = float(documents[1]["chain_count"])


def _float_draw_chain_order(documents: list[dict[str, Any]]) -> None:
    documents[1]["chain_order"][0] = float(documents[1]["chain_order"][0])


def _trailer_seed_mismatch(documents: list[dict[str, Any]]) -> None:
    documents[-1]["trailer"]["seed"] += 1


def _tree_depth_above_native_limit(documents: list[dict[str, Any]]) -> None:
    documents[1]["tree_depth"] = 21


def _missing_draw_seed(documents: list[dict[str, Any]]) -> None:
    del documents[1]["seed"]


def _missing_draw_count(documents: list[dict[str, Any]]) -> None:
    del documents[1]["draw_count"]


def _missing_draw_chain_metadata(documents: list[dict[str, Any]]) -> None:
    del documents[1]["chain_count"]


def _missing_draw_tree_depth(documents: list[dict[str, Any]]) -> None:
    del documents[1]["tree_depth"]


def _tree_depth_above_declared_max(documents: list[dict[str, Any]]) -> None:
    documents[1]["tree_depth"] = documents[0]["settings"]["max_treedepth"] + 1


def _wrong_divergence_summary(documents: list[dict[str, Any]]) -> None:
    documents[-1]["trailer"]["chains"][0]["divergences"] += 1


def _wrong_treedepth_histogram(documents: list[dict[str, Any]]) -> None:
    documents[-1]["trailer"]["chains"][0]["treedepth_histogram"][0] += 1


def _missing_header_stats_mode(documents: list[dict[str, Any]]) -> None:
    del documents[0]["sample_stats_mode"]


def _missing_draw_stats_mode(documents: list[dict[str, Any]]) -> None:
    del documents[1]["sample_stats_mode"]


def _missing_v2_energy(documents: list[dict[str, Any]]) -> None:
    del documents[1]["energy"]


def _missing_tree_accept(documents: list[dict[str, Any]]) -> None:
    del documents[1]["tree_accept"]


def _missing_step_size(documents: list[dict[str, Any]]) -> None:
    del documents[-1]["trailer"]["chains"][0]["step_size"]


def _missing_mean_accept(documents: list[dict[str, Any]]) -> None:
    del documents[-1]["trailer"]["chains"][0]["mean_accept"]


def _missing_packing(documents: list[dict[str, Any]]) -> None:
    del documents[0]["packing"]


def _missing_target_accept(documents: list[dict[str, Any]]) -> None:
    del documents[0]["settings"]["target_accept"]


def _invalid_target_accept(documents: list[dict[str, Any]]) -> None:
    documents[0]["settings"]["target_accept"] = 1.0


def _invalid_tree_accept(documents: list[dict[str, Any]]) -> None:
    documents[1]["tree_accept"] = 1.1


def _invalid_step_size(documents: list[dict[str, Any]]) -> None:
    documents[-1]["trailer"]["chains"][0]["step_size"] = 0.0


def _invalid_mean_accept(documents: list[dict[str, Any]]) -> None:
    documents[-1]["trailer"]["chains"][0]["mean_accept"] = 1.1


def _wrong_mean_accept(documents: list[dict[str, Any]]) -> None:
    documents[-1]["trailer"]["chains"][0]["mean_accept"] += 0.01


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
        _missing_draw_index,
        _non_json_number,
        _overflowing_number,
        _wrong_value_rank,
        _float_header_seed,
        _float_num_warmup,
        _float_max_treedepth,
        _float_tree_depth,
        _float_divergences,
        _float_header_chains,
        _float_draw_seed,
        _float_draw_count,
        _float_draw_chain_count,
        _float_draw_chain_order,
        _trailer_seed_mismatch,
        _tree_depth_above_native_limit,
        _missing_draw_seed,
        _missing_draw_count,
        _missing_draw_chain_metadata,
        _missing_draw_tree_depth,
        _tree_depth_above_declared_max,
        _wrong_divergence_summary,
        _wrong_treedepth_histogram,
        _missing_header_stats_mode,
        _missing_draw_stats_mode,
        _missing_v2_energy,
        _missing_tree_accept,
        _missing_step_size,
        _missing_mean_accept,
        _missing_packing,
        _missing_target_accept,
        _invalid_target_accept,
        _invalid_tree_accept,
        _invalid_step_size,
        _invalid_mean_accept,
        _wrong_mean_accept,
    ],
)
def test_portable_posterior_rejects_incomplete_or_inconsistent_lineage(
    mutate: Callable[[list[dict[str, Any]]], None],
) -> None:
    documents = copy.deepcopy(_documents())
    mutate(documents)
    with pytest.raises(PortablePosteriorError):
        _validate(_encode(documents))


def test_portable_posterior_rejects_overflowing_standard_number() -> None:
    posterior = (
        (FIXTURE / "posterior.ndjson")
        .read_bytes()
        .replace(b'"target_accept":0.8', b'"target_accept":1e400', 1)
    )
    with pytest.raises(PortablePosteriorError):
        _validate(posterior)


def test_portable_posterior_rejects_oversized_integer_metadata() -> None:
    posterior = (
        (FIXTURE / "posterior.ndjson")
        .read_bytes()
        .replace(b'"target_accept":0.8', b'"target_accept":' + b"9" * 400, 1)
    )
    with pytest.raises(PortablePosteriorError):
        _validate(posterior)


def test_real_posterior_fixture_has_portable_authority() -> None:
    parsed = validate_portable_posterior(
        model_bytes=(FIXTURE / "model.ir.json").read_bytes(),
        data_bytes=(FIXTURE / "data.json").read_bytes(),
        posterior_bytes=(FIXTURE / "posterior.ndjson").read_bytes(),
    )
    assert len(parsed.draws) == 600
