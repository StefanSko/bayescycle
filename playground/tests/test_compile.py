from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "all 10 corpus models match native hashes and golden IR",
    "surfaces bayeswire declaration errors verbatim",
    "requires exactly one model class",
    "compiler executes in a dedicated worker",
    "module poisoning cannot affect the next compiler worker",
    "compiler timeout resets the isolated worker",
]


def test_compile_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "compile", 300_000)
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
