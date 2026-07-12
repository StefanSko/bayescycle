from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

DESIGN_CASES = [
    "designDocument builds decorrelated permuted-linspace envelopes",
    "truthDocument builds scalar envelopes",
    "defaultTruth is constraint-aware",
    "renderPriorPredictiveDensity is deterministic",
]


@pytest.mark.parametrize("case_name", DESIGN_CASES)
def test_design_suite_case(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "design", 60_000)
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"design harness did not report case {case_name!r}: {results}"
    result = matching[0]
    assert result["ok"], f"{case_name}: {result['error']}"
