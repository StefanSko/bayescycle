"""Produce-conformance: reference declarations must reproduce the corpus.

The corpus pins the wire format. Every reference model is declared, resolved,
serialized, and compared against the committed document; canonical-bytes
hashes are pinned separately so the canonical encoding cannot drift while the
pretty-printed documents stay equal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from reference_models import ReferenceModelCase, reference_model_cases

from bayeswire.ir import canonical_bytes, meta_from_dict, meta_to_dict

CORPUS_DIR = Path(__file__).parent.parent.parent / "corpus"

REGENERATE_HINT = (
    "Corpus files pin the v1 wire format. If this change is deliberate, run "
    "scripts/regenerate_corpus.py, review the diff, add a spec changelog "
    "entry, and decide whether the format version must change."
)


def _case_ids() -> list[str]:
    return [case.name for case in reference_model_cases()]


@pytest.mark.parametrize("case", reference_model_cases(), ids=_case_ids())
def test_corpus_document_matches_current_encoding(case: ReferenceModelCase) -> None:
    corpus_path = CORPUS_DIR / f"{case.name}.json"

    document = json.loads(corpus_path.read_text(encoding="utf-8"))

    assert meta_to_dict(case.meta) == document, REGENERATE_HINT


@pytest.mark.parametrize("case", reference_model_cases(), ids=_case_ids())
def test_corpus_document_decodes_to_current_metadata(case: ReferenceModelCase) -> None:
    corpus_path = CORPUS_DIR / f"{case.name}.json"

    document = json.loads(corpus_path.read_text(encoding="utf-8"))

    assert meta_from_dict(document) == case.meta, REGENERATE_HINT


def test_corpus_canonical_hashes_are_pinned() -> None:
    recorded = json.loads((CORPUS_DIR / "hashes.json").read_text(encoding="utf-8"))

    current = {
        case.name: hashlib.sha256(canonical_bytes(case.meta)).hexdigest()
        for case in reference_model_cases()
    }

    assert current == recorded, REGENERATE_HINT


@pytest.mark.parametrize("case", reference_model_cases(), ids=_case_ids())
def test_evaluation_fixture_carries_the_corpus_document(case: ReferenceModelCase) -> None:
    fixture_path = CORPUS_DIR / "fixtures" / f"{case.name}.json"

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    assert fixture["name"] == case.name
    assert fixture["ir"] == meta_to_dict(case.meta), REGENERATE_HINT
    assert fixture["data"], "fixtures bundle concrete bind data"
    assert fixture["evaluations"], "fixtures bundle JAX-oracle evaluations"
