from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

RENDER_CASES = [
    "readDashboardData shapes the fixture",
    "stats primitives",
    "every renderer is deterministic",
    "trank grid draws one panel per parameter",
    "ess×r-hat scatter has the verdict line",
    "precis shows mean and 89% interval",
    "divergence verdict thresholds",
    "density overlay renders observed vs replicates",
    "prior→posterior overlay labels the overlap",
]


@pytest.mark.parametrize("case_name", RENDER_CASES)
def test_render_suite_case(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "render", 120_000)
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"render harness did not report case {case_name!r}: {results}"
    result = matching[0]
    assert result["ok"], f"{case_name}: {result['error']}"
