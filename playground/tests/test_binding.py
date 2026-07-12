from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page


def test_binding_suite(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "binding", 60_000)
    matching = [
        result
        for result in results
        if result["name"] == "bind assignments override name matching and report invalid columns"
    ]
    assert matching, f"binding harness did not report assignment case: {results}"
    result = matching[0]
    assert result["ok"], f"binding assignments: {result['error']}"
