from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

SCALE_CASES = [
    "app subsampling caps a 2000-replicate density overlay",
    "density overlay accepts 2000 replicates directly",
]


@pytest.mark.parametrize("case_name", SCALE_CASES)
def test_scale_suite_case(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "scale", 120_000)
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"scale harness did not report case {case_name!r}: {results}"
    result = matching[0]
    assert result["ok"], f"{case_name}: {result['error']}"
