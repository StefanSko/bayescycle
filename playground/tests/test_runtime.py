from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "runtime returns named diagnostic artifacts",
    "prior predictive ignores sampler-only settings",
    "runtime sampling returns valid merged posterior and progress",
]


def test_runtime_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "runtime", 120_000)
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
