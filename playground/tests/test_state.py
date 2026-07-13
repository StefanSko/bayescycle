from collections.abc import Callable
from typing import Any

import pytest
from playwright.sync_api import Page

CASES = [
    "source edits invalidate compilation and artifacts",
    "document edits retain compilation but invalidate artifacts",
    "stale asynchronous results are ignored",
]


@pytest.mark.parametrize("case_name", CASES)
def test_state_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
    case_name: str,
) -> None:
    results = run_suite(page, base_url, "state")
    matching = [result for result in results if result["name"] == case_name]
    assert matching, f"state harness did not report {case_name!r}: {results}"
    assert matching[0]["ok"], f"{case_name}: {matching[0]['error']}"
