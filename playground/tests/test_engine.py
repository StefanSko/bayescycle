from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

CASES = [
    "golden diagnose is byte-identical",
    "streams per_draw_v2 in batches",
    "sample yields one output per chain and streams draws",
    "diagnose merges the two chain fits",
    "malformed IR returns a typed engine error",
]


@pytest.mark.parametrize("case_name", CASES)
def test_engine_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "engine", 120_000)
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"engine harness did not report {case_name!r}: {results}"
    assert matching[0]["ok"], f"{case_name}: {matching[0]['error']}"
