from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

SHARE_CASES = [
    "round-trips projects byte-equal",
    "rejects garbage",
    "warn threshold",
]


@pytest.mark.parametrize("case_name", SHARE_CASES)
def test_share_codec_suite_case(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "share", 60_000)
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"share harness did not report case {case_name!r}: {results}"
    result = matching[0]
    assert result["ok"], f"{case_name}: {result['error']}"
