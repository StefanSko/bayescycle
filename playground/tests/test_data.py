from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

DATA_CASES = [
    "sniffs comma/semicolon/tab",
    "falls back to comma on ties",
    "parses quoted fields with embedded delimiters",
    "single-column file does not warn; hidden delimiter does",
    "WaffleDivorce.csv: semicolon, 13 columns, 50 rows",
    "standardize gives mean 0 sd 1",
    "importJson validates data documents",
    "requiredInputs reads the IR",
    "bind matches by name, derives scalars, reports gaps",
]


@pytest.mark.parametrize("case_name", DATA_CASES)
def test_data_suite_case(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "data", 60_000)
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"data harness did not report case {case_name!r}: {results}"
    result = matching[0]
    assert result["ok"], f"{case_name}: {result['error']}"
