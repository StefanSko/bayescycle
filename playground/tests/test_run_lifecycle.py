from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "invalidating edit aborts a superseded run worker",
    "stale completion after abort cannot mutate state",
    "changed recompile lineage aborts and invalidates prior-model generation",
    "source edit aborts its in-flight compile controller",
]


def test_run_lifecycle_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "run-lifecycle")
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
