from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from bayescycle._workflow.generation_plan import (
    AuthoredProvenance,
    Draw,
    FitArtifact,
    FitAssociation,
    Fixed,
    GenerationPlanError,
    JointPredict,
    ModelPrior,
    OutcomesOf,
    PosteriorOf,
    generate_datasets,
    generation_invalidation_key,
    generation_plan_identity,
    parse_generation_plan_document,
    resolve_generation_plan_document,
    serialize_generation_plan,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "playground/tests/fixtures/generation_plan.fixed.v0.json"
DESIGN_SOURCE_FIXTURE = (
    ROOT / "playground/tests/fixtures/generation_plan.fixed.design-source.v0.json"
)
DESIGN_SOURCE_ORDER_FIXTURE = (
    ROOT / "playground/tests/fixtures/generation_plan.fixed.design-source-order.v0.json"
)
MODEL_BYTES = b'{"bayeswire_ir":1,"model":{}}\n'
OTHER_MODEL_BYTES = b'{"bayeswire_ir":1,"model":{"name":"other"}}\n'
DESIGN_BYTES = (
    b'{"format":"bayescycle.data.json.v1","variables":'
    b'{"x":{"dtype":"float64","shape":[3],"values":[-1.0,0.0,1.0]}}}\n'
)
FIXED_BYTES = (
    b'{"format":"bayescycle.data.json.v1","variables":'
    b'{"alpha":{"dtype":"float64","shape":[],"values":[0.5]}}}\n'
)
FIT_DATA_BYTES = b'{"format":"bayescycle.data.json.v1","variables":{}}\n'
POSTERIOR_BYTES = b'{"draws_format":"v0-provisional"}\n'
IDENTITY = "sha256:609db688556c5d8c2f5460b0029c7899e307b14c1aa54fdcf45bac4a38543697"
INVALIDATION_KEY = "sha256:ef642308cab544f3e479d74742988477c38a6f3640cb4777365bb8ff5e433377"


def _fixed_plan() -> Draw:
    return Draw(
        distribution=JointPredict(
            parameters=Fixed(FIXED_BYTES),
            outcomes=OutcomesOf(MODEL_BYTES, DESIGN_BYTES),
        ),
        count=2,
        seed=7,
    )


def test_convenience_equals_draw_of_joint_predict_for_every_source() -> None:
    fit = FitArtifact(MODEL_BYTES, FIT_DATA_BYTES, POSTERIOR_BYTES, FitAssociation.RUNTIME)
    sources = (
        Fixed(FIXED_BYTES),
        ModelPrior(MODEL_BYTES),
        PosteriorOf(fit),
    )
    for source in sources:
        expected = Draw(JointPredict(source, OutcomesOf(MODEL_BYTES, DESIGN_BYTES)), 2, 7)
        assert (
            generate_datasets(
                MODEL_BYTES,
                design=DESIGN_BYTES,
                parameter_source=source,
                count=2,
                seed=7,
            )
            == expected
        )


def test_serialized_fixed_plan_matches_shared_versioned_fixture() -> None:
    plan = _fixed_plan()
    assert serialize_generation_plan(plan) == FIXTURE.read_bytes()

    document = parse_generation_plan_document(FIXTURE.read_bytes())
    assert document.bytes == FIXTURE.read_bytes()
    assert document.identity_hash == IDENTITY
    assert document.invalidation_key == INVALIDATION_KEY
    assert generation_plan_identity(plan) == IDENTITY
    assert generation_invalidation_key(plan) == INVALIDATION_KEY


def test_design_source_matches_shared_fixture_and_resolves_byte_identically() -> None:
    source = {
        "x": "linspace(-2, 2, 3)",
        "stale_μ": "repeat([0], 3)",
    }
    plan = generate_datasets(
        MODEL_BYTES,
        design=DESIGN_BYTES,
        design_source=source,
        parameter_source=Fixed(FIXED_BYTES),
        count=2,
        seed=7,
    )
    source["x"] = "changed after construction"
    fixture = DESIGN_SOURCE_FIXTURE.read_bytes()
    assert serialize_generation_plan(plan) == fixture
    assert parse_generation_plan_document(fixture).bytes == fixture

    resolved = resolve_generation_plan_document(
        fixture,
        model_ir_bytes=MODEL_BYTES,
        design_bytes=DESIGN_BYTES,
        fixed_parameters_bytes=FIXED_BYTES,
    )
    assert resolved.design_source == {
        "x": "linspace(-2, 2, 3)",
        "stale_μ": "repeat([0], 3)",
    }
    assert serialize_generation_plan(resolved) == fixture


@pytest.mark.parametrize(
    "design_source",
    [[], {"": "x"}, {"x": ""}, {"x": 1}, None],
)
def test_rejects_malformed_serialized_design_source(design_source: object) -> None:
    document = json.loads(DESIGN_SOURCE_FIXTURE.read_bytes())
    document["design_source"] = design_source
    encoded = json.dumps(document, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(GenerationPlanError, match="design_source"):
        parse_generation_plan_document(encoded)


def test_design_source_canonicalizes_adversarial_keys_by_code_points() -> None:
    source = {
        "10": "10",
        "2": "2",
        "01": "01",
        "x": "x",
        "1": "1",
        "😀": "astral",
        "！": "high BMP",
    }
    plan = generate_datasets(
        MODEL_BYTES,
        design=DESIGN_BYTES,
        design_source=source,
        parameter_source=Fixed(FIXED_BYTES),
        count=2,
        seed=7,
    )
    fixture = DESIGN_SOURCE_ORDER_FIXTURE.read_bytes()
    assert serialize_generation_plan(plan) == fixture
    assert parse_generation_plan_document(fixture).bytes == fixture


def test_rejects_unsorted_design_source_document() -> None:
    canonical = DESIGN_SOURCE_FIXTURE.read_text(encoding="utf-8")
    source = '"design_source":{"stale_μ":"repeat([0], 3)","x":"linspace(-2, 2, 3)"}'
    unsorted = '"design_source":{"x":"linspace(-2, 2, 3)","stale_μ":"repeat([0], 3)"}'
    with pytest.raises(GenerationPlanError, match="strictly ascending"):
        parse_generation_plan_document(canonical.replace(source, unsorted).encode())


@pytest.mark.parametrize("design_source", [{"\ud800": "x"}, {"x": "\ud800"}])
def test_rejects_non_well_formed_in_memory_design_source(
    design_source: dict[str, str],
) -> None:
    with pytest.raises(GenerationPlanError, match="well-formed Unicode scalar-value"):
        generate_datasets(
            MODEL_BYTES,
            design=DESIGN_BYTES,
            design_source=design_source,
            parameter_source=Fixed(FIXED_BYTES),
        )


@pytest.mark.parametrize(
    "replacement",
    ['"design_source":{"\\ud800":"x"}', '"design_source":{"x":"\\ud800"}'],
)
def test_rejects_non_well_formed_document_design_source(replacement: str) -> None:
    canonical = DESIGN_SOURCE_FIXTURE.read_text(encoding="utf-8")
    source = '"design_source":{"stale_μ":"repeat([0], 3)","x":"linspace(-2, 2, 3)"}'
    with pytest.raises(GenerationPlanError, match="well-formed Unicode scalar-value"):
        parse_generation_plan_document(canonical.replace(source, replacement).encode())


def test_serializer_wraps_residual_unicode_encode_error() -> None:
    plan = _fixed_plan()
    object.__setattr__(plan, "design_source", {"x": "\ud800"})
    with pytest.raises(GenerationPlanError, match="well-formed Unicode"):
        serialize_generation_plan(plan)


def test_serializes_all_exact_parameter_source_variants() -> None:
    provenance = AuthoredProvenance(f"sha256:{'1' * 64}", f"sha256:{'2' * 64}")
    fit = FitArtifact(MODEL_BYTES, FIT_DATA_BYTES, POSTERIOR_BYTES, FitAssociation.PORTABLE)
    sources = (ModelPrior(MODEL_BYTES, provenance), PosteriorOf(fit))

    documents = [
        json.loads(
            serialize_generation_plan(
                generate_datasets(
                    MODEL_BYTES,
                    design=DESIGN_BYTES,
                    parameter_source=source,
                    count=3,
                    seed=11,
                )
            )
        )
        for source in sources
    ]
    assert documents[0]["distribution"]["parameters"] == {
        "kind": "model-prior",
        "model_hash": "sha256:2c8947663a1a49b8e48c52542efc5b94c237365e40d859ee088b9d214845ccd5",
        "authored_provenance": {
            "claimed_source_model_hash": f"sha256:{'1' * 64}",
            "claimed_outcome_model_hash": f"sha256:{'2' * 64}",
        },
    }
    assert documents[1]["distribution"]["parameters"] == {
        "kind": "posterior",
        "fit_hash": "sha256:9cd8922c37ec4ace35caf850100f4230988d0214b1d7dfee1c6015dd7cd48bee",
        "fit_model_hash": "sha256:2c8947663a1a49b8e48c52542efc5b94c237365e40d859ee088b9d214845ccd5",
        "fit_data_hash": "sha256:7657f9e3dcc7ce5eba549ba1641bd0bf3d7b5fc1dceea7b042184bfbc9c63294",
    }


def test_plans_copy_bytes_and_reject_invalid_live_values() -> None:
    mutable = bytearray(FIXED_BYTES)
    source = Fixed(cast(bytes, mutable))
    mutable[0] = ord("!")
    assert source.parameters_bytes == FIXED_BYTES

    with pytest.raises(GenerationPlanError, match="bytes"):
        Fixed(cast(bytes, lambda: None))
    with pytest.raises(GenerationPlanError, match="model"):
        JointPredict(
            ModelPrior(OTHER_MODEL_BYTES),
            OutcomesOf(MODEL_BYTES, DESIGN_BYTES),
        )


@pytest.mark.parametrize(
    ("count", "seed", "message"),
    [(0, 0, "count"), (1001, 0, "count"), (1, -1, "seed"), (1, 2**53, "seed")],
)
def test_rejects_plan_bounds(count: int, seed: int, message: str) -> None:
    with pytest.raises(GenerationPlanError, match=message):
        Draw(
            JointPredict(Fixed(FIXED_BYTES), OutcomesOf(MODEL_BYTES, DESIGN_BYTES)),
            count,
            seed,
        )


def test_rejects_unknown_or_missing_serialized_fields() -> None:
    document = json.loads(FIXTURE.read_bytes())
    mutations = []
    with_unknown = dict(document)
    with_unknown["worker"] = "forbidden"
    mutations.append(with_unknown)
    missing_version = dict(document)
    del missing_version["generation_plan_format"]
    mutations.append(missing_version)
    nested_unknown = json.loads(FIXTURE.read_bytes())
    nested_unknown["distribution"]["outcomes"]["backend"] = "forbidden"
    mutations.append(nested_unknown)

    for mutation in mutations:
        encoded = json.dumps(mutation, separators=(",", ":")).encode() + b"\n"
        with pytest.raises(GenerationPlanError, match="unknown|missing|format"):
            parse_generation_plan_document(encoded)

    nested = b"[" * 65 + b"0" + b"]" * 65
    deep = FIXTURE.read_bytes().replace(
        b'{"kind":"fixed","parameters_hash":',
        b'{"kind":"fixed","parameters_hash":' + nested + b',"ignored":',
    )
    with pytest.raises(GenerationPlanError, match="depth"):
        parse_generation_plan_document(deep)

    duplicate = FIXTURE.read_bytes().replace(b'"kind":"draw"', b'"kind":"wrong","kind":"draw"')
    with pytest.raises(GenerationPlanError, match="duplicate"):
        parse_generation_plan_document(duplicate)


def test_design_source_accepts_slots_named_like_structural_fields() -> None:
    plan = generate_datasets(
        MODEL_BYTES,
        design=DESIGN_BYTES,
        design_source={
            "count": "linspace(-2, 2, 3)",
            "dtype": "int64",
            "seed": "[0]",
            "values": "[1.5]",
        },
        parameter_source=Fixed(FIXED_BYTES),
        count=2,
        seed=7,
    )
    data = serialize_generation_plan(plan)
    assert parse_generation_plan_document(data).bytes == data
