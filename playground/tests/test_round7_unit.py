from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

ROUND7_CASES = [
    "scalar designs distinguish counts from continuous values",
    "ordered vector truth is increasing and centered",
]


@pytest.mark.parametrize("case_name", ROUND7_CASES)
def test_round7_unit_suite_case(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "round7", 60_000)
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"round7 harness did not report case {case_name!r}: {results}"
    result = matching[0]
    assert result["ok"], f"{case_name}: {result['error']}"
