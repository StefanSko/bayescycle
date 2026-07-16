from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "normalizes plain scalar vector and matrix JSON",
    "preserves plain variables named format and variables",
    "accepts and preserves every canonical dtype",
    "rejects ragged arrays and invalid canonical values",
    "serializes canonical bytes deterministically",
    "rejects document byte depth and scalar caps with named errors",
    "normalizes a near-cap document in one flat value pass",
    "canonical document round-trips at the scalar limit",
    "normalization preserves proto-named variables byte-exactly",
    "representative normalization matches pinned canonical bytes",
]


def test_document_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "documents")
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
