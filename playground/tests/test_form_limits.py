from collections.abc import Callable
from typing import Any

from playwright.sync_api import Page

CASES = [
    "parameter forms stop above two hundred fields",
    "design forms stop above fifty compatible slots",
]


def test_form_limit_contract(
    page: Page,
    base_url: str,
    run_suite: Callable[..., list[dict[str, Any]]],
) -> None:
    results = run_suite(page, base_url, "form-limits")
    assert [result["name"] for result in results] == CASES
    assert all(result["ok"] for result in results), results
