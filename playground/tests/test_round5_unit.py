from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page


def test_round5_unit_suite(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "round5", 60_000)
    matching = [
        result
        for result in results
        if result["name"] == "designDocument emits validated integer scalars"
    ]
    assert matching, f"round5 harness did not report scalar design case: {results}"
    result = matching[0]
    assert result["ok"], f"scalar design document: {result['error']}"
