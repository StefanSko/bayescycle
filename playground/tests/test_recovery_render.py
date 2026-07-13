from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page


def test_recovery_summary(
    page: Page, base_url: str, run_suite: Callable[..., list[dict[str, Any]]]
) -> None:
    results = run_suite(page, base_url, "recovery")
    matching = [
        result
        for result in results
        if result["name"] == "recovery summary displays engine-owned facts"
    ]
    assert matching and matching[0]["ok"], matching
