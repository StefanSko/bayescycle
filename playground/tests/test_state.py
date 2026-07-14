from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "scoped starts reject dependencies changed during preflight",
    "scoped attempts preserve successful ancestors on failure",
    "selection invalidates only generated conditioning descendants",
    "observed edits preserve generated-data fit and recovery",
    "inference edits preserve fit and independent generation",
    "replacement fits invalidate posterior-sourced collections",
    "source edits invalidate compilation and artifacts",
    "document edits retain compilation but invalidate artifacts",
    "sampler edits invalidate runs but retain compilation",
    "generation settings selectively invalidate artifacts and stale runs",
    "generation edits invalidate a completed generated-data fit",
    "generation edits orphan a running generated-data fit",
    "generation edits preserve an independent running sample",
    "sampler edits preserve an independent prior predictive run",
    "predictive draw edits preserve fits and running samples",
    "predictive draw edits orphan running prior generation",
    "settings notices follow their posterior artifacts",
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
