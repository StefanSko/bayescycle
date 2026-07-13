from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page


def test_round6_unit_suite(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "round6", 60_000)
    matching = [
        result
        for result in results
        if result["name"] == "truthDocument broadcasts resolved vector parameters"
    ]
    assert matching, f"round6 harness did not report vector truth case: {results}"
    result = matching[0]
    assert result["ok"], f"vector truth document: {result['error']}"
