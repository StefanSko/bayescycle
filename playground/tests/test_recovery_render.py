from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page


def test_recovery_summary(
    page: Page, base_url: str, run_suite: Callable[..., list[dict[str, Any]]]
) -> None:
    results = run_suite(page, base_url, "recovery")
    assert [result["name"] for result in results] == [
        "recovery summary displays engine-owned facts",
        "recovery truth map uses dashboard parameter labels",
        "recovery truth indexing preserves vector labels at scale",
    ]
    assert all(result["ok"] for result in results), results
