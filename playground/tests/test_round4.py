from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

ROUND4_CASES = [
    "importJson accepts all additional v1 dtypes",
    "requiredInputs preserves every matrix dimension",
    "example asset URLs retain a deployment path prefix",
]


@pytest.mark.parametrize("case_name", ROUND4_CASES)
def test_round4_suite_case(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "round4", 60_000)
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"round4 harness did not report case {case_name!r}: {results}"
    result = matching[0]
    assert result["ok"], f"{case_name}: {result['error']}"
