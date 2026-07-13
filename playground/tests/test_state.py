from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "source edits invalidate compilation and artifacts",
    "document edits retain compilation but invalidate artifacts",
    "sampler edits invalidate runs but retain compilation",
    "later runs clear stale follow-up notices",
    "successful follow-ups preserve earlier artifacts",
    "a replacement posterior drops artifacts derived from the old fit",
    "stale asynchronous results are ignored",
]


def test_state_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "state")
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
