from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from bayescycle._run_artifacts.canonical_data import JsonValue
from bayescycle._run_artifacts.generated_datasets import (
    MAX_GENERATED_ARTIFACT_BYTES,
    MAX_GENERATED_LINE_BYTES,
    GeneratedDatasetsArtifactError,
    parse_generated_datasets,
    verify_generated_datasets,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "playground/tests/fixtures/generated_datasets.v0.ndjson"
MODEL_BYTES = b'{"bayeswire_ir":1,"model":{}}\n'
DESIGN_BYTES = (
    b'{"format":"bayescycle.data.json.v1","variables":'
    b'{"x":{"dtype":"float64","shape":[3],"values":[-1.0,0.0,1.0]}}}\n'
)
FIT_DATA_BYTES = b'{"format":"bayescycle.data.json.v1","variables":{}}\n'
FIXED_BYTES = (
    b'{"format":"bayescycle.data.json.v1","variables":'
    b'{"alpha":{"dtype":"float64","shape":[],"values":[0.5]}}}\n'
)
PARAMETERS_0 = FIXED_BYTES
DATASET_0 = (
    b'{"format":"bayescycle.data.json.v1","variables":'
    b'{"x":{"dtype":"float64","shape":[3],"values":[-1.0,0.0,1.0]},'
    b'"y":{"dtype":"float64","shape":[3],"values":[-0.2,0.5,1.2]}}}\n'
)


def _fixture_bytes() -> bytes:
    return FIXTURE.read_bytes()


def _documents() -> list[dict[str, JsonValue]]:
    return cast(
        list[dict[str, JsonValue]],
        [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()],
    )


def _encode(documents: list[dict[str, JsonValue]]) -> bytes:
    return b"".join(
        json.dumps(document, separators=(",", ":"), allow_nan=False).encode() + b"\n"
        for document in documents
    )


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _posterior_source() -> bytes:
    fingerprint = _sha256(b"bayescycle-model-data-v1\n" + MODEL_BYTES + b"\n" + FIT_DATA_BYTES)
    documents: list[dict[str, object]] = [
        {
            "draws_format": "v0-provisional",
            "artifact_kind": "posterior_draws",
            "artifact_scope": "observed_data_conditioned_parameter_draws",
            "model_data_fingerprint": fingerprint,
            "params": [{"name": "alpha", "shape": [], "coordinate_order": [[]]}],
            "parameter_count": 1,
            "parameter_order": ["alpha"],
            "settings": {"num_draws": 2},
            "chain_count": 1,
            "chain_order": [0],
            "draw_count": 2,
        },
        {
            "draws_format": "v0-provisional",
            "artifact_kind": "posterior_draws",
            "artifact_scope": "observed_data_conditioned_parameter_draws",
            "draw_index": 0,
            "chain": 0,
            "draw": 0,
            "parameter_count": 1,
            "parameter_order": ["alpha"],
            "values": {"alpha": 1.0},
        },
        {
            "draws_format": "v0-provisional",
            "artifact_kind": "posterior_draws",
            "artifact_scope": "observed_data_conditioned_parameter_draws",
            "draw_index": 1,
            "chain": 0,
            "draw": 1,
            "parameter_count": 1,
            "parameter_order": ["alpha"],
            "values": {"alpha": 2.0},
        },
        {
            "trailer": {
                "draws_format": "v0-provisional",
                "artifact_kind": "posterior_draws",
                "artifact_scope": "observed_data_conditioned_parameter_draws",
                "model_data_fingerprint": fingerprint,
                "draws_per_chain": 2,
                "chain_count": 1,
                "chain_order": [0],
                "draw_count": 2,
                "parameter_count": 1,
                "parameter_order": ["alpha"],
                "params": 1,
                "chains": [{"chain": 0, "draw_count": 2}],
            }
        },
    ]
    return b"".join(
        json.dumps(document, separators=(",", ":")).encode() + b"\n" for document in documents
    )


def _source_stream(kind: str) -> bytes:
    documents = _documents()
    header = documents[0]
    trailer = documents[-1]["trailer"]
    assert isinstance(header, dict)
    assert isinstance(trailer, dict)
    source: dict[str, JsonValue]
    lineages: list[dict[str, JsonValue]]
    if kind == "model-prior":
        source = {
            "kind": "model-prior",
            "model_hash": header["generation_model_hash"],
            "authored_provenance": None,
        }
        lineages = [
            {"kind": "model-prior", "source_draw_index": 0},
            {"kind": "model-prior", "source_draw_index": 1},
        ]
    else:
        source = {
            "kind": "posterior",
            "fit_hash": f"sha256:{'1' * 64}",
            "fit_model_hash": header["generation_model_hash"],
            "fit_data_hash": f"sha256:{'2' * 64}",
        }
        lineages = [
            {"kind": "posterior", "source_draw_index": 7, "chain": 1, "draw": 3},
            {"kind": "posterior", "source_draw_index": 2, "chain": 0, "draw": 2},
        ]
    header["parameter_source"] = source
    trailer["parameter_source"] = source
    for index, lineage in enumerate(lineages, start=1):
        draw = documents[index]
        draw["source_lineage"] = lineage
        parameters = draw["parameters"]
        assert isinstance(parameters, dict)
        variables = parameters["variables"]
        assert isinstance(variables, dict)
        alpha = variables["alpha"]
        assert isinstance(alpha, dict)
        alpha["values"] = [float(index)]
    return _encode(documents)


def test_parses_canonical_header_pairs_and_trailer() -> None:
    artifact = parse_generated_datasets(_fixture_bytes())

    assert artifact.generation_model_hash == (
        "sha256:2c8947663a1a49b8e48c52542efc5b94c237365e40d859ee088b9d214845ccd5"
    )
    assert artifact.design_hash == (
        "sha256:02b4040e221b6967267a7e8bcb1692c4b00c9d76e90ed58a7344e5390eaddcc6"
    )
    assert artifact.source_kind == "fixed"
    assert artifact.count == 2
    assert artifact.seed == 7
    assert [draw.draw_index for draw in artifact.draws] == [0, 1]
    assert all(draw.parameters.variable_map()["alpha"].values == (0.5,) for draw in artifact.draws)
    assert artifact.draws[0].dataset.variable_map()["y"].values == (-0.2, 0.5, 1.2)


def test_accepts_varying_model_prior_and_posterior_parameters() -> None:
    for kind in ("model-prior", "posterior"):
        artifact = parse_generated_datasets(_source_stream(kind))
        assert artifact.source_kind == kind
        assert [draw.parameters.variable_map()["alpha"].values for draw in artifact.draws] == [
            (1.0,),
            (2.0,),
        ]
        assert all(draw.source_kind == kind for draw in artifact.draws)


def test_selection_preserves_nested_parameter_and_dataset_bytes() -> None:
    artifact = parse_generated_datasets(_fixture_bytes())

    selection = artifact.select(0)

    assert selection.draw_index == 0
    assert selection.parameters_bytes == PARAMETERS_0
    assert selection.dataset_bytes == DATASET_0
    assert artifact.bytes == _fixture_bytes()
    with pytest.raises(GeneratedDatasetsArtifactError, match="range"):
        artifact.select(2)


def test_verifies_exact_hashes_fixed_values_and_design_prefix() -> None:
    artifact = parse_generated_datasets(_fixture_bytes())
    verify_generated_datasets(
        artifact,
        model_bytes=MODEL_BYTES,
        design_bytes=DESIGN_BYTES,
        fixed_parameters_bytes=FIXED_BYTES,
    )

    with pytest.raises(GeneratedDatasetsArtifactError, match="design hash"):
        verify_generated_datasets(
            artifact,
            model_bytes=MODEL_BYTES,
            design_bytes=b"{}\n",
            fixed_parameters_bytes=FIXED_BYTES,
        )
    with pytest.raises(GeneratedDatasetsArtifactError, match="fixed parameters"):
        verify_generated_datasets(
            artifact,
            model_bytes=MODEL_BYTES,
            design_bytes=DESIGN_BYTES,
            fixed_parameters_bytes=b'{"format":"bayescycle.data.json.v1","variables":{}}\n',
        )


def test_resolver_rejects_output_from_the_wrong_plan() -> None:
    prior = parse_generated_datasets(_source_stream("model-prior"))
    with pytest.raises(GeneratedDatasetsArtifactError, match="source kind"):
        verify_generated_datasets(
            prior,
            model_bytes=MODEL_BYTES,
            design_bytes=DESIGN_BYTES,
            fixed_parameters_bytes=FIXED_BYTES,
            expected_source_kind="fixed",
            expected_count=2,
            expected_seed=7,
        )

    fixed = parse_generated_datasets(_fixture_bytes())
    with pytest.raises(GeneratedDatasetsArtifactError, match="count|seed"):
        verify_generated_datasets(
            fixed,
            model_bytes=MODEL_BYTES,
            design_bytes=DESIGN_BYTES,
            fixed_parameters_bytes=FIXED_BYTES,
            expected_source_kind="fixed",
            expected_count=1,
            expected_seed=8,
        )


def test_resolver_verifies_prior_descriptor_and_posterior_draw_lineage() -> None:
    prior = parse_generated_datasets(_source_stream("model-prior"))
    verify_generated_datasets(
        prior,
        model_bytes=MODEL_BYTES,
        design_bytes=DESIGN_BYTES,
        expected_source_kind="model-prior",
        expected_count=2,
        expected_seed=7,
        model_prior_bytes=MODEL_BYTES,
        authored_provenance=None,
    )
    with pytest.raises(GeneratedDatasetsArtifactError, match="model-prior"):
        verify_generated_datasets(
            prior,
            model_bytes=MODEL_BYTES,
            design_bytes=DESIGN_BYTES,
            expected_source_kind="model-prior",
            expected_count=2,
            expected_seed=7,
            model_prior_bytes=b"other model",
            authored_provenance=None,
        )

    posterior_bytes = _posterior_source()
    documents = _documents()
    source: dict[str, JsonValue] = {
        "kind": "posterior",
        "fit_hash": _sha256(posterior_bytes),
        "fit_model_hash": _sha256(MODEL_BYTES),
        "fit_data_hash": _sha256(FIT_DATA_BYTES),
    }
    documents[0]["parameter_source"] = source
    trailer = documents[-1]["trailer"]
    assert isinstance(trailer, dict)
    trailer["parameter_source"] = source
    for index in range(2):
        draw = documents[index + 1]
        draw["source_lineage"] = {
            "kind": "posterior",
            "source_draw_index": index,
            "chain": 0,
            "draw": index,
        }
        parameters = draw["parameters"]
        assert isinstance(parameters, dict)
        variables = parameters["variables"]
        assert isinstance(variables, dict)
        alpha = variables["alpha"]
        assert isinstance(alpha, dict)
        alpha["values"] = [float(index + 1)]
    posterior_artifact = parse_generated_datasets(_encode(documents))
    verify_generated_datasets(
        posterior_artifact,
        model_bytes=MODEL_BYTES,
        design_bytes=DESIGN_BYTES,
        expected_source_kind="posterior",
        expected_count=2,
        expected_seed=7,
        posterior_bytes=posterior_bytes,
        fit_data_bytes=FIT_DATA_BYTES,
    )
    wrong = _documents()
    wrong[0]["parameter_source"] = source
    wrong_trailer = wrong[-1]["trailer"]
    assert isinstance(wrong_trailer, dict)
    wrong_trailer["parameter_source"] = source
    for index in range(2):
        wrong[index + 1]["source_lineage"] = {
            "kind": "posterior",
            "source_draw_index": index,
            "chain": 0,
            "draw": index,
        }
    with pytest.raises(GeneratedDatasetsArtifactError, match="posterior"):
        verify_generated_datasets(
            parse_generated_datasets(_encode(wrong)),
            model_bytes=MODEL_BYTES,
            design_bytes=DESIGN_BYTES,
            expected_source_kind="posterior",
            expected_count=2,
            expected_seed=7,
            posterior_bytes=posterior_bytes,
            fit_data_bytes=FIT_DATA_BYTES,
        )

    reshaped = json.loads(json.dumps(documents))
    reshaped[0]["parameter_schema"][0]["shape"] = [1]
    for draw in reshaped[1:-1]:
        draw["parameters"]["variables"]["alpha"]["shape"] = [1]
    with pytest.raises(GeneratedDatasetsArtifactError, match="posterior|shape"):
        verify_generated_datasets(
            parse_generated_datasets(_encode(cast(list[dict[str, JsonValue]], reshaped))),
            model_bytes=MODEL_BYTES,
            design_bytes=DESIGN_BYTES,
            expected_source_kind="posterior",
            expected_count=2,
            expected_seed=7,
            posterior_bytes=posterior_bytes,
            fit_data_bytes=FIT_DATA_BYTES,
        )


def test_rejects_malformed_truncated_nonfinite_and_oversized_streams() -> None:
    fixture = _fixture_bytes()
    malformed: list[tuple[bytes, str]] = [
        (b"\n".join(fixture.splitlines()[:-1]) + b"\n", "trailer"),
        (fixture.replace(b'"count":2', b'"count":3', 1), "count"),
        (fixture.replace(b'"draw_index":1', b'"draw_index":3', 1), "draw_index"),
        (fixture.replace(b'"draw_index":0', b'"draw_index":0.0', 1), "draw_index"),
        (fixture.replace(b'"draw_count":2', b'"draw_count":2.0', 1), "draw_count"),
        (fixture.replace(b'"values":[0.5]', b'"values":[NaN]', 1), "finite"),
        (
            fixture.replace(b'"values":[0.5]', b'"values":[' + b"9" * 400 + b"]", 1),
            "finite|range|representable",
        ),
        (b" " * MAX_GENERATED_LINE_BYTES + fixture, "line"),
    ]
    documents = _documents()
    documents[0]["unexpected"] = True
    malformed.append((_encode(documents), "unknown"))
    attacker = b'{"format":"bayescycle.data.json.v1","variables":{}}'
    duplicate = fixture.replace(
        b'"parameters":', b'"parameters":' + attacker + b',"parameters":', 1
    )
    malformed.append((duplicate, "duplicate"))

    for stream, expected in malformed:
        with pytest.raises(GeneratedDatasetsArtifactError, match=expected):
            parse_generated_datasets(stream)

    assert MAX_GENERATED_ARTIFACT_BYTES == 64 * 1024 * 1024
