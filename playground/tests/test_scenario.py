from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "scenario target divergence requires an explicit main-model recompile",
    "scenario schema compatibility rejects added prior data slots",
    "recovery truth projection retains shared parameters without mutating the pair",
]


def test_scenario_integrity_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "scenario")
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
