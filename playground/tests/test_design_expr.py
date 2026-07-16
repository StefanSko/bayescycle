from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "evaluates literal linspace and repeat expressions exactly",
    "pins seeded uniform values",
    "pins seeded normal values and default seed",
    "rejects malformed and out-of-vocabulary expressions",
]


def test_design_expression_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "design-expr")
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
