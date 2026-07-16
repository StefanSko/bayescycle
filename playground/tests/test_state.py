from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "scoped starts reject dependencies changed during preflight",
    "conditioning commit installs fit and artifacts atomically",
    "generation success installs collection selection atomically",
    "scoped attempts preserve successful ancestors on failure",
    "selection invalidates only generated conditioning descendants",
    "observed edits cancel observed fit and posterior generation attempts",
    "observed edits preserve generated-data fit and recovery",
    "inference edits preserve fit and independent generation",
    "replacement fits invalidate posterior-sourced collections",
    "source edits invalidate compilation and artifacts",
    "failed recompile preserves artifacts and fit lineage",
    "successful recompile invalidates prior artifacts",
    "document edits retain compilation but invalidate artifacts",
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
