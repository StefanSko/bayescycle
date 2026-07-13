from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page


def test_compiler_lifecycle_contract(
    page: Page, base_url: str, run_suite: Callable[..., list[dict[str, Any]]]
) -> None:
    results = run_suite(page, base_url, "compiler-lifecycle")
    assert len(results) == 6, results
    failures = [result for result in results if not result["ok"]]
    assert not failures, failures
